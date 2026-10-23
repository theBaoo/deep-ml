def compute_throughput(intervals):
    """
    Compute step-wise, cumulative, and overall tokens-per-second.

    Args:
        intervals: list of (tokens_processed, elapsed_seconds) tuples.

    Returns:
        dict with keys 'step_tps', 'cumulative_tps', 'overall_tps'.
    """
    if not intervals:
        return {'step_tps': [], 'cumulative_tps': [], 'overall_tps': 0.0}

    step_tps = []
    cumulative_tps = []
    overall_tps = 0.0

    all_tokens = 0
    all_times = 0.0
    for (tokens, times) in intervals:
        if times == 0:
            raise ValueError("1")

        all_tokens += tokens
        all_times += times
        step_tps.append(round(tokens / times, 2))
        cumulative_tps.append(round(all_tokens / all_times, 2))

    overall_tps = all_tokens / all_times

    return {'step_tps': step_tps, 'cumulative_tps': cumulative_tps, 'overall_tps': overall_tps}