def compute_dilation_factor(tasks: list, capacity: int) -> dict:
    """
    Compute the resource dilation factor for a constrained scheduling problem.
    
    Args:
        tasks: List of dicts with 'duration', 'resources', 'dependencies' keys.
        capacity: Total available resource units.
    
    Returns:
        Dict with 'dilation_factor', 'actual_makespan', 'lower_bound', 'start_times'.
    """
    n = len(tasks)

    # ---------- 1. Critical path ----------
    # dp[i] = task i 完成的最早时间（忽略资源限制）
    dp = [0.0] * n

    # dependencies are task indices. We can compute recursively because
    # the dependency graph is assumed to be a DAG.
    state = [0] * n  # 0 = unvisited, 1 = visiting, 2 = done

    def dfs(i):
        if state[i] == 2:
            return dp[i]
        if state[i] == 1:
            raise ValueError("Dependency graph contains a cycle")

        state[i] = 1

        if tasks[i]["dependencies"]:
            dp[i] = tasks[i]["duration"] + max(
                dfs(dep) for dep in tasks[i]["dependencies"]
            )
        else:
            dp[i] = tasks[i]["duration"]

        state[i] = 2
        return dp[i]

    critical_path = max((dfs(i) for i in range(n)), default=0.0)

    # ---------- 2. Work lower bound ----------
    total_work = sum(
        task["duration"] * task["resources"]
        for task in tasks
    )

    lower_bound = float(max(
        critical_path,
        total_work / capacity
    ))

    # ---------- 3. Greedy list scheduling ----------
    start_times = [None] * n
    finish_times = [None] * n

    scheduled = [False] * n
    completed = [False] * n

    # Currently running tasks: (finish_time, task_index)
    running = []

    current_time = 0.0
    used_resources = 0

    while not all(completed):
        # ---- Process completions ----
        finished_now = []

        for finish_time, i in running:
            if finish_time <= current_time:
                completed[i] = True
                used_resources -= tasks[i]["resources"]
                finished_now.append((finish_time, i))

        if finished_now:
            running = [
                (finish_time, i)
                for finish_time, i in running
                if finish_time > current_time
            ]

        # ---- Schedule ready tasks in index order ----
        scheduled_something = True

        while scheduled_something:
            scheduled_something = False

            for i in range(n):
                if scheduled[i]:
                    continue

                deps = tasks[i]["dependencies"]

                # All dependencies must have completed.
                if not all(completed[d] for d in deps):
                    continue

                resources = tasks[i]["resources"]

                if used_resources + resources > capacity:
                    continue

                # Schedule task i
                scheduled[i] = True
                start_times[i] = current_time

                finish_time = current_time + tasks[i]["duration"]
                finish_times[i] = finish_time

                running.append((finish_time, i))
                used_resources += resources

                scheduled_something = True

        # If everything is completed, we're done.
        if all(completed):
            break

        # Otherwise jump to the next completion event.
        if running:
            current_time = min(
                finish_time for finish_time, _ in running
            )
        else:
            # This should only happen for an invalid dependency graph.
            raise ValueError("Unable to make progress")

    actual_makespan = max(finish_times, default=0.0)

    dilation_factor = (
        actual_makespan / lower_bound
        if lower_bound > 0
        else 0.0
    )

    return {
        "dilation_factor": round(dilation_factor, 4),
        "actual_makespan": round(actual_makespan, 4),
        "lower_bound": round(lower_bound, 4),
        "start_times": [
            round(t, 4) for t in start_times
        ],
    }
