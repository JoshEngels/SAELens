from typing import Callable

import einops
import numpy as np
import torch
from tqdm import tqdm


class SyntheticActivationStore:
    def __init__(self, batch_generator: Callable[[], torch.Tensor]):
        self.batch_generator = batch_generator
        self.estimated_norm_scaling_factor = 1.0

    # Returns batches of size (batch_size, 1, d_in)
    def next_batch(self):
        batch = self.batch_generator()
        batch = einops.rearrange(batch, "b d -> b () d")
        return batch

    @torch.no_grad()
    def estimate_norm_scaling_factor(self, n_batches_for_norm_estimate: int = int(1e3)):
        norms_per_batch = []
        assert n_batches_for_norm_estimate > 0
        for _ in tqdm(
            range(n_batches_for_norm_estimate), desc="Estimating norm scaling factor"
        ):
            acts = self.next_batch()
            d_in = acts.shape[-1]
            norms_per_batch.append(acts.norm(dim=-1).mean().item())
        mean_norm = np.mean(norms_per_batch)
        scaling_factor = np.sqrt(d_in) / mean_norm

        return scaling_factor
