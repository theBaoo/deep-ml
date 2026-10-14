def balance_undersample(data: list) -> list:
    """
    Undersample the majority classes so all classes have the same number of
    samples equal to the minority class count.

    data: list of (sample, label) tuples
    Returns: list of (sample, label) tuples, order-preserving
    """
    from collections import defaultdict
    import random
    gdata = defaultdict(list)
    
    for i, (_, label) in enumerate(data):
        gdata[label].append(i)

    if not gdata:
        return []
    
    min_count = min(len(samples) for samples in gdata.values())

    selected = set()
    for indices in gdata.values():
        selected.update(indices[:min_count])

    return [item for i, item in enumerate(data) if i in selected]
