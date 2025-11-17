"""
Intelligent cache warming and prefetching for AI Toolkit DataLoader.

This module provides asynchronous prefetching of batches to GPU, reducing GPU idle time
by preparing the next batch while the current batch is being processed.
"""

import threading
import queue
import torch
from typing import Optional, Any
from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO, FileItemDTO


class DataLoaderPrefetcher:
    """
    Asynchronously prefetches batches from dataloader and moves them to GPU.

    This prefetcher:
    - Runs in a background thread to fetch next N batches
    - Moves batch tensors to GPU using CUDA streams for async transfer
    - Queues pre-warmed batches for the training loop
    - Reduces GPU idle time between training steps

    Example:
        >>> prefetcher = DataLoaderPrefetcher(dataloader, device='cuda', prefetch_batches=2)
        >>> for batch in prefetcher:
        >>>     # batch is already on GPU
        >>>     train_step(batch)

    Args:
        dataloader: PyTorch DataLoader to wrap
        device: Target device (e.g., 'cuda' or 'cuda:0')
        prefetch_batches: Number of batches to prefetch (default: 2)
        enabled: Whether prefetching is enabled (default: True)
    """

    def __init__(
        self,
        dataloader,
        device: str = 'cuda',
        prefetch_batches: int = 2,
        enabled: bool = True,
    ):
        self.dataloader = dataloader
        self.device = torch.device(device) if isinstance(device, str) else device
        self.prefetch_batches = max(1, prefetch_batches)
        self.enabled = enabled and torch.cuda.is_available()

        # Threading primitives
        self.queue = None
        self.thread = None
        self.stop_event = None
        self.exception = None

        # CUDA stream for async transfer
        self.stream = None

    def _move_to_device(self, obj: Any, device: torch.device, stream: Optional[torch.cuda.Stream] = None) -> Any:
        """
        Recursively move tensors to device using the specified CUDA stream.

        Handles:
        - FileItemDTO objects
        - DataLoaderBatchDTO objects
        - Lists and tuples
        - Dictionaries
        - Tensors
        - Other objects (returned as-is)
        """
        if isinstance(obj, torch.Tensor):
            # Move tensor to device non-blocking if using stream
            return obj.to(device, non_blocking=(stream is not None))

        elif isinstance(obj, FileItemDTO):
            # FileItemDTO contains various cached data that may be tensors
            # We need to move relevant tensor attributes
            # Clone the object to avoid modifying the original
            import copy
            obj_copy = copy.copy(obj)

            # Move tensor attributes if they exist
            if hasattr(obj_copy, '_encoded_latent') and obj_copy._encoded_latent is not None:
                obj_copy._encoded_latent = self._move_to_device(obj_copy._encoded_latent, device, stream)

            if hasattr(obj_copy, 'img') and obj_copy.img is not None:
                obj_copy.img = self._move_to_device(obj_copy.img, device, stream)

            if hasattr(obj_copy, 'conditioning') and obj_copy.conditioning is not None:
                obj_copy.conditioning = self._move_to_device(obj_copy.conditioning, device, stream)

            if hasattr(obj_copy, 'clip_image_embeds') and obj_copy.clip_image_embeds is not None:
                if isinstance(obj_copy.clip_image_embeds, dict):
                    obj_copy.clip_image_embeds = {
                        k: self._move_to_device(v, device, stream)
                        for k, v in obj_copy.clip_image_embeds.items()
                    }
                else:
                    obj_copy.clip_image_embeds = self._move_to_device(obj_copy.clip_image_embeds, device, stream)

            # Handle PromptEmbeds (from text_embedding_cache)
            if hasattr(obj_copy, 'prompt_embeds') and obj_copy.prompt_embeds is not None:
                from toolkit.prompt_utils import PromptEmbeds
                if isinstance(obj_copy.prompt_embeds, PromptEmbeds):
                    # Move PromptEmbeds tensors
                    if obj_copy.prompt_embeds.text_embeds is not None:
                        if isinstance(obj_copy.prompt_embeds.text_embeds, (list, tuple)):
                            obj_copy.prompt_embeds.text_embeds = [
                                self._move_to_device(t, device, stream) if t is not None else None
                                for t in obj_copy.prompt_embeds.text_embeds
                            ]
                        else:
                            obj_copy.prompt_embeds.text_embeds = self._move_to_device(
                                obj_copy.prompt_embeds.text_embeds, device, stream
                            )

                    if obj_copy.prompt_embeds.pooled_embeds is not None:
                        obj_copy.prompt_embeds.pooled_embeds = self._move_to_device(
                            obj_copy.prompt_embeds.pooled_embeds, device, stream
                        )

                    if obj_copy.prompt_embeds.attention_mask is not None:
                        obj_copy.prompt_embeds.attention_mask = self._move_to_device(
                            obj_copy.prompt_embeds.attention_mask, device, stream
                        )

            return obj_copy

        elif isinstance(obj, DataLoaderBatchDTO):
            # DataLoaderBatchDTO contains a list of FileItemDTO objects
            import copy
            obj_copy = copy.copy(obj)
            if hasattr(obj_copy, 'file_items') and obj_copy.file_items is not None:
                obj_copy.file_items = [
                    self._move_to_device(item, device, stream)
                    for item in obj_copy.file_items
                ]
            return obj_copy

        elif isinstance(obj, (list, tuple)):
            moved_list = [self._move_to_device(item, device, stream) for item in obj]
            return type(obj)(moved_list)

        elif isinstance(obj, dict):
            return {k: self._move_to_device(v, device, stream) for k, v in obj.items()}

        else:
            # Return as-is for non-tensor objects
            return obj

    def _prefetch_worker(self):
        """
        Background thread worker that prefetches batches to GPU.

        This runs in a separate thread and:
        1. Fetches batches from the dataloader iterator
        2. Moves them to GPU using a CUDA stream
        3. Synchronizes the stream to ensure transfer is complete
        4. Queues the batch for consumption by the main thread
        """
        try:
            # Create a CUDA stream for async transfers
            if self.enabled:
                print("[Prefetch] Creating CUDA stream...")
                import sys
                sys.stdout.flush()
                self.stream = torch.cuda.Stream(device=self.device)
                print(f"[Prefetch] CUDA stream created successfully: {self.stream}")
                sys.stdout.flush()

            print("[Prefetch] Starting to iterate dataloader...")
            sys.stdout.flush()
            for batch in self.dataloader:
                if self.stop_event.is_set():
                    break

                if self.enabled:
                    # Move batch to GPU using dedicated stream
                    with torch.cuda.stream(self.stream):
                        batch_gpu = self._move_to_device(batch, self.device, self.stream)

                    # Wait for transfer to complete before queuing
                    self.stream.synchronize()
                else:
                    # Prefetching disabled, just pass through
                    batch_gpu = batch

                # Queue the batch for main thread
                self.queue.put(batch_gpu)

            # Send sentinel to signal end of iteration
            self.queue.put(None)

        except Exception as e:
            # Store exception to re-raise in main thread
            self.exception = e
            self.queue.put(None)

    def __iter__(self):
        """
        Start the prefetch thread and return iterator.
        """
        # Create queue with size = prefetch_batches
        self.queue = queue.Queue(maxsize=self.prefetch_batches)

        # Create stop event
        self.stop_event = threading.Event()

        # Reset exception
        self.exception = None

        # Start prefetch thread
        self.thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        self.thread.start()

        return self

    def __next__(self):
        """
        Get next prefetched batch from queue.
        """
        # Get batch from queue (blocks if queue is empty)
        batch = self.queue.get()

        # Check for exception in worker thread
        if self.exception is not None:
            raise self.exception

        # Check for end of iteration
        if batch is None:
            # Clean up thread
            self.stop()
            raise StopIteration

        return batch

    def stop(self):
        """
        Stop the prefetch thread and clean up resources.
        """
        if self.thread is not None:
            self.stop_event.set()

            # Drain queue to unblock worker thread
            try:
                while True:
                    self.queue.get_nowait()
            except queue.Empty:
                pass

            # Wait for thread to finish
            self.thread.join(timeout=5.0)
            self.thread = None

        # Clean up CUDA stream
        if self.stream is not None:
            self.stream = None

    def __del__(self):
        """
        Ensure thread is stopped on deletion.
        """
        self.stop()


def wrap_dataloader_with_prefetcher(
    dataloader,
    device: str = 'cuda',
    prefetch_batches: int = 2,
    enabled: bool = True,
):
    """
    Convenience function to wrap a dataloader with prefetching.

    Args:
        dataloader: PyTorch DataLoader to wrap
        device: Target device for prefetching (default: 'cuda')
        prefetch_batches: Number of batches to prefetch (default: 2)
        enabled: Whether to enable prefetching (default: True)

    Returns:
        DataLoaderPrefetcher instance

    Example:
        >>> dataloader = get_dataloader_from_datasets(dataset_options, batch_size=4)
        >>> prefetcher = wrap_dataloader_with_prefetcher(dataloader, prefetch_batches=2)
        >>> for batch in prefetcher:
        >>>     # batch is already on GPU
        >>>     train_step(batch)
    """
    return DataLoaderPrefetcher(
        dataloader=dataloader,
        device=device,
        prefetch_batches=prefetch_batches,
        enabled=enabled,
    )
