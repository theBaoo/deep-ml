import numpy as np
import math

class ZeroCopyBatchLoader:
    def __init__(self, data: np.ndarray, batch_size: int):
        """Store data in a flat contiguous buffer simulating shared memory."""
        self.data = data # [n_samples, n_features]
        self.batch_size = batch_size
        self.num_batches_ = math.ceil(data.shape[0] / batch_size)

    def num_batches(self) -> int:
        """Return total number of batches."""
        return self.num_batches_

    def get_batch(self, batch_idx: int) -> np.ndarray:
        """Return batch as a zero-copy view into the buffer."""
        return self.data[batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size]

    def is_zero_copy(self, batch_idx: int) -> bool:
        """Check whether the batch shares memory with the buffer."""
        return np.shares_memory(self.get_batch(batch_idx), self.data)

    def get_batch_means(self) -> list:
        """Return list of per-batch mean values, each rounded to 4 decimals."""
        return [round(np.mean(self.get_batch(i)), 4) for i in range(self.num_batches())]

    def write_to_buffer(self, row: int, col: int, value: float) -> None:
        """Write a value directly into the flat buffer at (row, col)."""
        self.data[row, col] = value