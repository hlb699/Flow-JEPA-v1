"""Four-process regression test for global-batch SIGReg loss and gradients."""

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from module import SIGReg


def main():
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 4:
        raise AssertionError(f"expected four ranks, got {world_size}")

    time_steps, local_batch, dim = 3, 2, 5
    global_batch = local_batch * world_size
    torch.manual_seed(1234)
    full_input = torch.randn(time_steps, global_batch, dim)
    dist.broadcast(full_input, src=0)

    layer = torch.nn.Linear(dim, dim, bias=False)
    dist.broadcast(layer.weight.data, src=0)
    initial_weight = layer.weight.detach().clone()
    ddp_layer = DistributedDataParallel(layer)

    local_input = full_input[
        :, rank * local_batch : (rank + 1) * local_batch
    ]
    torch.manual_seed(5678)
    distributed_loss = SIGReg(knots=5, num_proj=7)(
        ddp_layer(local_input)
    )
    distributed_loss.backward()
    distributed_gradient = ddp_layer.module.weight.grad.detach().clone()

    reference_layer = torch.nn.Linear(dim, dim, bias=False)
    reference_layer.weight.data.copy_(initial_weight)
    torch.manual_seed(5678)
    reference_loss = SIGReg(
        knots=5,
        num_proj=7,
        gather_distributed=False,
    )(
        reference_layer(full_input)
    )
    reference_loss.backward()

    torch.testing.assert_close(distributed_loss, reference_loss)
    torch.testing.assert_close(
        distributed_gradient,
        reference_layer.weight.grad,
        rtol=1e-5,
        atol=1e-6,
    )
    if rank == 0:
        print("four-rank SIGReg loss and gradient equivalence: PASS")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
