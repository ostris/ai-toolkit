from toolkit.timer import Timer


def test_print_ignores_cancelled_timer_without_samples():
    timer = Timer()
    printed_timings = []
    timer.add_after_print_hook(printed_timings.append)

    timer.start("cancelled")
    timer.cancel("cancelled")
    timer.print()

    assert printed_timings == [{}]
