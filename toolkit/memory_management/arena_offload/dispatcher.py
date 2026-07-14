"""Generic saved-forward dispatcher over canonical arena storage."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from toolkit.memory_management.immutable_runtime import (
    ImmutableProgram,
    ImmutableRuntimeError,
    ImmutableTransformerRuntime,
    build_program_fingerprint,
)
from toolkit.memory_management.arena_offload.transfer import (
    configure_fetch_runtime,
    free_on_backward,
)


DISPATCHER_GENERATION = "generic-block-dispatcher-v1"


def _in_backward_graph_task() -> bool:
    """True for non-reentrant checkpoint replay inside autograd backward."""
    try:
        return int(torch._C._current_graph_task_id()) >= 0
    except (AttributeError, RuntimeError):
        return False


class OriginalBlockInvoker(torch.nn.Module):
    """Own a selected block while calling its preserved installed forward."""

    def __init__(self, block, saved_forward):
        super().__init__()
        self.block = block
        self._saved_forward = saved_forward

    def forward(self, *args, **kwargs):
        return self._saved_forward(*args, **kwargs)


@dataclass(frozen=True)
class _Replacement:
    state_name: str
    leaf_index: int
    tensor_indices: tuple[int, ...]
    reconstruct: object


class _InstalledDispatcher:
    def __init__(self, executor, index):
        self.executor = executor
        self.index = int(index)

    @torch.compiler.disable
    def __call__(self, *args, **kwargs):
        return self.executor.dispatch(self.index, args, kwargs)


class GenericBlockDispatcherRuntime(ImmutableTransformerRuntime):
    """Reuse residency/transfer policy while preserving ordinary block math."""

    def __init__(
        self,
        model,
        residency,
        *,
        selection,
        depth=3,
        compile_blocks=True,
        compile_dynamic=True,
        compile_dynamic_hints=(),
        protected_training_leaf_keys=(),
        owner_token=None,
    ):
        self.selection = selection
        super().__init__(
            model,
            residency,
            blocks=selection.blocks,
            block_keys=selection.block_keys,
            entries_by_block=selection.entries_by_block,
            depth=depth,
            compile_blocks=compile_blocks,
            compile_dynamic=compile_dynamic,
            compile_dynamic_hints=compile_dynamic_hints,
            protected_training_leaf_keys=protected_training_leaf_keys,
            owner_token=owner_token,
        )
        self._invokers = ()
        self._dispatchers = ()
        self._saved_forwards = ()
        self._replacements = ()

    def _replacement_plan(self, index, invoker):
        abi = self._block_abis[index]
        record = self.residency.arena.block_record(abi.block_key)
        replacements = []
        for leaf_index, leaf_name in enumerate(record.leaf_names):
            spec = record.leaf_spec(leaf_name)
            for substitution in spec.substitutions:
                replacements.append(
                    _Replacement(
                        state_name=(
                            f"block.{leaf_name}.{substitution.name}"
                        ),
                        leaf_index=leaf_index,
                        tensor_indices=substitution.tensor_indices,
                        reconstruct=substitution.reconstruct,
                    )
                )
        try:
            parameter_names = {
                name for name, _value in invoker.named_parameters(remove_duplicate=False)
            }
        except TypeError:
            parameter_names = {name for name, _value in invoker.named_parameters()}
        try:
            buffer_names = {
                name for name, _value in invoker.named_buffers(remove_duplicate=False)
            }
        except TypeError:
            buffer_names = {name for name, _value in invoker.named_buffers()}
        available = parameter_names | buffer_names
        for replacement in replacements:
            if replacement.state_name not in available:
                raise ImmutableRuntimeError(
                    f"installed_replacement_target_missing:{abi.block_key}."
                    f"{replacement.state_name}"
                )
        if len({item.state_name for item in replacements}) != len(replacements):
            raise ImmutableRuntimeError(
                f"installed_replacement_target_conflict:{abi.block_key}"
            )
        return tuple(replacements)

    @staticmethod
    def _build_state(replacements, leaf_args):
        state = {}
        for replacement in replacements:
            tensors = leaf_args[replacement.leaf_index]
            selected = tuple(tensors[index] for index in replacement.tensor_indices)
            state[replacement.state_name] = replacement.reconstruct(selected)
        return state

    def _get_dispatch_kernel(self, index):
        key = (DISPATCHER_GENERATION, int(index))
        existing = self._block_kernels.get(key)
        if existing is not None:
            return existing
        invoker = self._invokers[index]
        replacements = self._replacements[index]

        def kernel(leaf_args, args, kwargs):
            state = self._build_state(replacements, leaf_args)
            return torch.func.functional_call(
                invoker,
                state,
                args,
                kwargs,
                strict=False,
                tie_weights=False,
            )

        if self.compile_blocks:
            kernel = torch.compile(
                kernel,
                mode="default",
                fullgraph=False,
                dynamic=self.compile_dynamic,
            )
        self._block_kernels[key] = kernel
        return kernel

    def _dispatcher_program(self, mode):
        fingerprint = build_program_fingerprint(
            mode,
            self._block_abis,
            architecture_key=DISPATCHER_GENERATION,
            depth=self.depth,
            checkpoint_mode="model" if mode == self.TRAIN else "none",
        )
        return ImmutableProgram(mode=mode, fingerprint=fingerprint, trunk=None)

    def finalize_execution(self):
        if self._finalized:
            return self
        configure_fetch_runtime(depth=self.depth, owner_token=self.owner_token)
        saved = tuple(block.forward for block in self._blocks)
        invokers = tuple(
            OriginalBlockInvoker(block, saved_forward)
            for block, saved_forward in zip(self._blocks, saved, strict=True)
        )
        replacements = tuple(
            self._replacement_plan(index, invoker)
            for index, invoker in enumerate(invokers)
        )
        dispatchers = tuple(
            _InstalledDispatcher(self, index)
            for index in range(len(self._blocks))
        )
        self._saved_forwards = saved
        self._invokers = invokers
        self._replacements = replacements
        self._dispatchers = dispatchers
        installed = []
        try:
            for block, dispatcher in zip(self._blocks, dispatchers, strict=True):
                block.forward = dispatcher
                installed.append(block)
        except BaseException:
            for index, block in enumerate(installed):
                block.forward = saved[index]
            raise
        self._programs = {
            self.TRAIN: self._dispatcher_program(self.TRAIN),
            self.SAMPLE: self._dispatcher_program(self.SAMPLE),
        }
        self._finalization_signature = (DISPATCHER_GENERATION,)
        self._finalized = True
        return self

    def _mark_dispatch_dynamic(self, tensor):
        if not self.compile_blocks or not self.compile_dynamic_hints:
            return
        for dim, lo, hi in self.compile_dynamic_hints:
            size = int(tensor.shape[dim])
            if (lo is not None and size < lo) or (hi is not None and size > hi):
                self._warn_hint_out_of_range(dim, size, lo, hi)
                continue
            torch._dynamo.maybe_mark_dynamic(tensor, int(dim))

    def dispatch(self, index, args, kwargs):
        self._require_finalized()
        if self._active_token is None:
            raise ImmutableRuntimeError(
                f"block_dispatch_outside_transformer_execution:{self._block_abis[index].block_key}"
            )
        if self._active_mode == self.SAMPLE and torch.is_grad_enabled():
            raise ImmutableRuntimeError(
                "immutable_execution_mode_mismatch:active=sample:call=train"
            )
        if not args or not isinstance(args[0], torch.Tensor):
            raise ImmutableRuntimeError(
                f"unsupported_block_arguments:{self._block_abis[index].block_key}"
            )

        source = self._sources.source(index)
        transfer = source.transfer
        token = None
        compact_flat = None
        first = args[0]
        training = self._active_mode == self.TRAIN
        if transfer is not None:
            if training and any(
                (source.block_key, leaf) in self.protected_training_leaf_keys
                for leaf in source.leaf_names
            ):
                raise ImmutableRuntimeError(
                    f"uncheckpointed_block_not_resident:{source.block_key}"
                )
            host = self.residency.arena.block_record(source.block_key).host_flat
            nbytes = int(transfer.compact_nbytes)
            guard = first.reshape(-1)[:1].clone() if training else first
            token = torch.ops.mm.fetch_start_multi_after(
                host,
                source.ranges,
                nbytes,
                guard,
            )
            compact_flat = torch.ops.mm.fetch_wait(token, nbytes)
            if training and torch.is_grad_enabled():
                first = free_on_backward(first, token)
                args = (first, *args[1:])

        leaf_args = source.assemble_leaf_args(self.residency, compact_flat)
        self._mark_dispatch_dynamic(first)
        output = self._get_dispatch_kernel(index)(leaf_args, args, kwargs)
        if token is not None:
            # The first checkpoint pass discards its fetched views, so return
            # that slot after forward. Replay runs inside an autograd graph
            # task; its token is instead released by free_on_backward after
            # the compiled block backward has consumed the substituted state.
            if (
                not training
                or not torch.is_grad_enabled()
                or not _in_backward_graph_task()
            ):
                torch.ops.mm.fetch_free_after(token, output)
        return output

    def close(self):
        if self._sources.active_executions:
            raise ImmutableRuntimeError("cannot_close_during_execution")
        for block, dispatcher, saved in zip(
            self._blocks,
            self._dispatchers,
            self._saved_forwards,
        ):
            if block.forward is dispatcher:
                block.forward = saved
        self._dispatchers = ()
        self._saved_forwards = ()
        self._invokers = ()
        self._replacements = ()
        super().close()


def prepare_block_dispatcher_runtime(
    transformer,
    residency,
    *,
    selection,
    depth=3,
    compile_blocks=True,
    compile_dynamic=True,
    compile_dynamic_hints=(),
    protected_training_leaf_keys=(),
    owner_token=None,
):
    return GenericBlockDispatcherRuntime(
        transformer,
        residency,
        selection=selection,
        depth=depth,
        compile_blocks=compile_blocks,
        compile_dynamic=compile_dynamic,
        compile_dynamic_hints=compile_dynamic_hints,
        protected_training_leaf_keys=protected_training_leaf_keys,
        owner_token=owner_token,
    )
