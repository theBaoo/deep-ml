def huber_loss(y_true, y_pred, delta=1.0):
	"""
	Compute the Huber Loss between true and predicted values.

	Args:
		y_true (float | list[float]): Ground truth values
		y_pred (float | list[float]): Predicted values
		delta (float): Transition threshold between MSE and MAE behavior

	Returns:
		float: Average Huber loss
	"""
	# Your code here
	import numpy as np
	y_true = np.asarray(y_true)
	y_pred = np.asarray(y_pred)
	mask = np.abs(y_true - y_pred) <= delta
	mse = 0.5 * (y_pred - y_true) ** 2 * mask
	mae = delta * (np.abs(y_pred - y_true) - 0.5 * delta) * ~mask
	return (mae + mse).mean().item()