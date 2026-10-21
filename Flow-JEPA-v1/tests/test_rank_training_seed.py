"""Regression test for rank-specific stochastic training streams."""

import random

import numpy as np
import torch
import torch.distributed as dist

from utils import RankAwareTrainingSeedCallback


class TrainerStub:
    def __init__(self, global_rank):
        self.global_rank = global_rank


def main():
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 4:
        raise AssertionError(f"expected four ranks, got {world_size}")

    torch.manual_seed(3072)
    model = torch.nn.Linear(8, 8)
    initial_weight = model.weight.detach().clone()

    callback = RankAwareTrainingSeedCallback(3072)
    callback.on_fit_start(TrainerStub(rank), model)
    sample = torch.tensor(
        [random.random(), np.random.random(), torch.rand(()).item()]
    )

    gathered_weights = [torch.empty_like(initial_weight) for _ in range(4)]
    gathered_samples = [torch.empty_like(sample) for _ in range(4)]
    dist.all_gather(gathered_weights, initial_weight)
    dist.all_gather(gathered_samples, sample)

    for weight in gathered_weights[1:]:
        torch.testing.assert_close(weight, gathered_weights[0])
    if len({tuple(value.tolist()) for value in gathered_samples}) != 4:
        raise AssertionError("rank-specific random streams are not unique")

    if rank == 0:
        print("identical initialization and distinct rank RNG streams: PASS")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
