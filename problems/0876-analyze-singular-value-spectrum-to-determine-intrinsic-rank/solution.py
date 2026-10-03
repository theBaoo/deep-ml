import numpy as np

def suggest_rank(delta_W: np.ndarray, energy_threshold: float) -> int:
	"""
	Return the smallest rank k such that the top-k singular values of delta_W
	capture at least `energy_threshold` of the total squared-singular-value energy.
	"""
	# Your code here
	sv = np.linalg.svd(delta_W, compute_uv=False)
	energy = sv ** 2
	te = energy.sum()

	if te == 0:
		return 0

	ce = np.cumsum(energy)
	tare = energy_threshold * te

	return int(np.searchsorted(ce, tare) + 1)