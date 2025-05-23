import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datautils import MyTrainDataset

import torch.multiprocessing as mp # Wrapper around python native multiprocessing
from torch.utils.data.distributed import DistributedSampler # middle man takeing input data and distributed it into multiple GPUs
from torch.nn.parallel import DistributedDataParallel as DDP # wrapper around model to make it distributed
from torch.distributed import init_process_group, destroy_process_group # for initializing and destroying the process group

import os

# world_size is the total number of processes in a group
# rank is the unique id of each process in the group [0 to world_size-1]

def ddp_setup(rank, world_size, gpu_ids):
    """
    Args:
        rank: Unique indentifier of each process
        world_size: Total number of processes in the group
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355' # just a free part
    init_process_group(backend="nccl", rank=rank, world_size=world_size) # nccl is NVIDIA's library for collective communication
    torch.cuda.set_device(gpu_ids[rank]) # set the device for each process

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int, 
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.model = DDP(self.model, device_ids=[self.gpu_id]) # wrap the model with DDP

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = F.cross_entropy(output, targets)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        for source, targets in self.train_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self._run_batch(source, targets)

    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = "checkpoint.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0: # only save from one GPU
                self._save_checkpoint(epoch)


def load_train_objs():
    train_set = MyTrainDataset(2048)  # load your dataset
    model = torch.nn.Linear(20, 1)  # load your model
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    return train_set, model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,  # shuffle is handled by DistributedSampler
        sampler=DistributedSampler(dataset),  # distribute the data across GPUs
    )


def main(rank: int, world_size:int, total_epochs: int, save_every: int, batch_size:int, gpu_ids: list):
    ddp_setup(rank, world_size, gpu_ids) # initialize the process group
    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = Trainer(model, train_data, optimizer, gpu_ids[rank], save_every)
    trainer.train(total_epochs)
    destroy_process_group()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()

    gpu_ids = [0, 3]
    world_size = len(gpu_ids) # number of GPUs available
    print(f"Number of GPUs: {world_size}")
    
    mp.spawn(
        main, 
        args=(world_size, args.total_epochs, args.save_every, args.batch_size, gpu_ids),
        nprocs=world_size,
        join=True,
    )