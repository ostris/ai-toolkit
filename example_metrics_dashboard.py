#!/usr/bin/env python3
"""
Example: Using MetricsCollector for Training Metrics Dashboard

This script demonstrates how to integrate the MetricsCollector
into a training loop for comprehensive metrics tracking and display.

Features demonstrated:
- Real-time metrics collection during training
- Dashboard display at regular intervals
- Memory usage monitoring
- Dataloader throughput tracking
- Timer integration for detailed breakdowns
"""

from toolkit.metrics_collector import MetricsCollector
import time
import random


def example_basic_usage():
    """
    Basic usage example showing essential metrics tracking.
    """
    print("\n" + "="*70)
    print("Example 1: Basic Usage")
    print("="*70)

    # Initialize metrics collector
    metrics = MetricsCollector(
        window_size=100,  # Keep last 100 steps for averaging
        enable_memory_tracking=True,
        enable_throughput_tracking=True,
        enable_dataloader_tracking=True,
    )

    # Training loop
    num_steps = 50
    for step in range(num_steps):
        # Mark start of step
        metrics.start_step()

        # ... your training code here ...
        time.sleep(0.01)  # Simulate training

        # Mark end of step with metrics
        metrics.end_step(
            loss=random.uniform(0.2, 0.5),
            lr=1e-4,
            batch_size=4
        )

        # Print dashboard every 20 steps
        if (step + 1) % 20 == 0:
            metrics.print_dashboard()

    print("\n✓ Basic usage example completed")


def example_with_loss_dict():
    """
    Example showing tracking of multiple loss components.
    """
    print("\n" + "="*70)
    print("Example 2: Tracking Multiple Loss Components")
    print("="*70)

    metrics = MetricsCollector(window_size=50)

    for step in range(30):
        metrics.start_step()

        # ... training code ...
        time.sleep(0.01)

        # Track multiple loss components
        metrics.end_step(
            loss_dict={
                'total_loss': random.uniform(0.3, 0.5),
                'mse_loss': random.uniform(0.1, 0.2),
                'perceptual_loss': random.uniform(0.05, 0.15),
                'kl_loss': random.uniform(0.001, 0.01),
            },
            lr=1e-4 * (0.95 ** step),  # Decaying LR
            batch_size=8
        )

    # Final dashboard
    metrics.print_dashboard()
    print("\n✓ Loss dict example completed")


def example_with_dataloader_tracking():
    """
    Example showing dataloader metrics tracking.
    """
    print("\n" + "="*70)
    print("Example 3: Dataloader Metrics Tracking")
    print("="*70)

    metrics = MetricsCollector(window_size=100)

    for step in range(50):
        # Simulate dataloader fetch
        fetch_start = time.time()
        time.sleep(random.uniform(0.001, 0.01))  # Simulate I/O
        fetch_time = time.time() - fetch_start
        metrics.record_dataloader_fetch(fetch_time)

        # Simulate cache hit/miss
        if random.random() < 0.85:  # 85% hit rate
            metrics.record_cache_hit()
        else:
            metrics.record_cache_miss()

        # Training step
        metrics.start_step()
        time.sleep(0.02)  # Simulate training
        metrics.end_step(
            loss=random.uniform(0.2, 0.4),
            batch_size=random.choice([4, 8, 16])
        )

    metrics.print_dashboard()
    print("\n✓ Dataloader tracking example completed")


def example_with_timer_integration():
    """
    Example showing integration with Timer for detailed breakdowns.
    """
    print("\n" + "="*70)
    print("Example 4: Timer Integration for Detailed Breakdown")
    print("="*70)

    metrics = MetricsCollector(window_size=50)

    # Simulate Timer output
    timing_breakdown = {
        'get_batch': 0.0123,
        'forward_pass': 0.0456,
        'backward_pass': 0.0789,
        'optimizer_step': 0.0234,
        'prepare_latents': 0.0111,
        'encode_prompts': 0.0067,
    }

    # Update timing breakdown from Timer
    metrics.update_timing_breakdown(timing_breakdown)

    # Do training steps
    for step in range(20):
        metrics.start_step()
        time.sleep(0.01)
        metrics.end_step(loss=random.uniform(0.2, 0.4))

    # Print with timing breakdown
    metrics.print_dashboard(include_timing_breakdown=True)
    print("\n✓ Timer integration example completed")


def example_export_and_logging():
    """
    Example showing export functionality for logging.
    """
    print("\n" + "="*70)
    print("Example 5: Export Metrics for Logging")
    print("="*70)

    metrics = MetricsCollector(window_size=50)

    # Run some steps
    for step in range(10):
        metrics.start_step()
        time.sleep(0.01)
        metrics.end_step(
            loss=random.uniform(0.2, 0.4),
            lr=1e-4,
            batch_size=4
        )

    # Export metrics
    exported = metrics.export_to_dict()

    print("\nExported metrics structure:")
    print(f"  Summary: {len(exported['summary'])} metrics")
    print(f"  Step count: {exported['step_count']}")
    print(f"  Total samples: {exported['total_samples_processed']}")
    print(f"  Cache stats: {exported['cache_stats']}")

    # You could save this to a file or send to a logging service
    # import json
    # with open('metrics.json', 'w') as f:
    #     json.dump(exported, f, indent=2)

    print("\n✓ Export and logging example completed")


def example_realistic_training_loop():
    """
    Realistic training loop example with all features.
    """
    print("\n" + "="*70)
    print("Example 6: Realistic Training Loop")
    print("="*70)

    # Initialize
    metrics = MetricsCollector(window_size=100)
    num_epochs = 2
    steps_per_epoch = 25
    dashboard_interval = 10

    print(f"\nTraining for {num_epochs} epochs ({steps_per_epoch} steps/epoch)...")

    global_step = 0
    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")

        for step in range(steps_per_epoch):
            # Dataloader fetch
            fetch_start = time.time()
            time.sleep(random.uniform(0.001, 0.005))
            metrics.record_dataloader_fetch(time.time() - fetch_start)

            # Cache hit/miss
            if random.random() < 0.9:
                metrics.record_cache_hit()
            else:
                metrics.record_cache_miss()

            # Training step
            metrics.start_step()
            time.sleep(random.uniform(0.015, 0.025))  # Simulate variable step time

            metrics.end_step(
                loss_dict={
                    'total_loss': random.uniform(0.5 / (epoch + 1), 1.0 / (epoch + 1)),
                    'mse': random.uniform(0.2 / (epoch + 1), 0.5 / (epoch + 1)),
                },
                lr=1e-4 * (0.98 ** global_step),
                batch_size=random.choice([4, 8])
            )

            global_step += 1

            # Show dashboard periodically
            if global_step % dashboard_interval == 0:
                metrics.print_dashboard()

        # End of epoch summary
        print(f"\n--- End of Epoch {epoch + 1} ---")
        metrics.print_dashboard()

        # Could reset metrics per epoch if desired
        # metrics.reset()

    print("\n✓ Realistic training loop example completed")


def example_integration_code_snippet():
    """
    Show code snippet for integrating into actual training code.
    """
    print("\n" + "="*70)
    print("Example 7: Integration Code Snippet")
    print("="*70)

    print("""
# In your training script initialization:

from toolkit.metrics_collector import MetricsCollector

# Create metrics collector
metrics = MetricsCollector(
    window_size=100,
    enable_memory_tracking=True,
    enable_throughput_tracking=True,
    enable_dataloader_tracking=True,
)

# In your training loop:

for step in range(num_steps):
    # Start step timing
    metrics.start_step()

    # Fetch batch (optional: time it)
    fetch_start = time.time()
    batch = next(dataloader_iterator)
    metrics.record_dataloader_fetch(time.time() - fetch_start)

    # Your training code
    with torch.no_grad():
        # ... prepare data ...

    optimizer.zero_grad()
    with accelerator.accumulate(model):
        loss_dict = train_step(batch)

    # End step with metrics
    metrics.end_step(
        loss_dict=loss_dict,
        lr=optimizer.param_groups[0]['lr'],
        batch_size=len(batch)
    )

    # Optional: update timing breakdown from Timer
    if hasattr(self, 'timer'):
        metrics.update_timing_breakdown(self.timer.timers)

    # Print dashboard periodically
    if step % 100 == 0 and step > 0:
        metrics.print_dashboard(include_timing_breakdown=True)

# At end of training
print("\\nFinal metrics:")
metrics.print_dashboard(include_timing_breakdown=True)

# Export for logging
exported_metrics = metrics.export_to_dict()
# Save to file, log to wandb, etc.
""")

    print("\n✓ Integration code snippet shown")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("MetricsCollector Usage Examples")
    print("="*70)

    example_basic_usage()
    example_with_loss_dict()
    example_with_dataloader_tracking()
    example_with_timer_integration()
    example_export_and_logging()
    example_realistic_training_loop()
    example_integration_code_snippet()

    print("\n" + "="*70)
    print("All examples completed!")
    print("="*70)
    print("\nFor more information, see:")
    print("  - toolkit/metrics_collector.py (source code)")
    print("  - test_metrics_collector.py (test suite)")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
