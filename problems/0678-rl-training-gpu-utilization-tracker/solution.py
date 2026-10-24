def track_gpu_utilization(timestamps_ms: list, gpu_util_pct: list, phase_labels: list) -> dict:
    """
    Analyze GPU utilization during RL training and compute per-phase statistics.
    """
    total_time_ms = timestamps_ms[-1] - timestamps_ms[0]

    if total_time_ms <= 0:
        raise ValueError("Total time must be positive")

    # Aggregate statistics for each phase.
    phase_data = {}

    weighted_util_sum = 0.0

    for i in range(len(gpu_util_pct)):
        duration = timestamps_ms[i + 1] - timestamps_ms[i]
        util = gpu_util_pct[i]
        phase = phase_labels[i]

        weighted_util_sum += duration * util

        if phase not in phase_data:
            phase_data[phase] = {
                "total_time": 0.0,
                "weighted_util": 0.0,
                "wasted_gpu_time": 0.0,
                "first_index": i,
            }

        data = phase_data[phase]
        data["total_time"] += duration
        data["weighted_util"] += duration * util
        data["wasted_gpu_time"] += duration * (100 - util)

    # Overall time-weighted utilization.
    avg_gpu_util_pct = round(weighted_util_sum / total_time_ms, 2)

    # Per-phase statistics, with alphabetically sorted phase names.
    phase_stats = {}

    for phase in sorted(phase_data):
        data = phase_data[phase]
        phase_time = data["total_time"]

        phase_stats[phase] = {
            "total_time_ms": round(phase_time, 2),
            "avg_util_pct": round(data["weighted_util"] / phase_time, 2),
            "time_fraction_pct": round(phase_time / total_time_ms * 100, 2),
        }

    # Find the phase with the most wasted GPU-time.
    # On ties, retain the phase that appeared first.
    bottleneck_phase = max(
        phase_data,
        key=lambda phase: (
            phase_data[phase]["wasted_gpu_time"],
            -phase_data[phase]["first_index"],
        ),
    )

    return {
        "total_time_ms": round(total_time_ms, 2),
        "avg_gpu_util_pct": avg_gpu_util_pct,
        "phase_stats": phase_stats,
        "bottleneck_phase": bottleneck_phase,
        "gpu_idle_fraction_pct": round(100 - avg_gpu_util_pct, 2),
    }