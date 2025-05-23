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

def ddp_setup():
    init_process_group(backend="nccl") # torch run will set the environment variables for us
    

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        save_every: int, 
        snapshot_path: str,
    ) -> None:
        self.gpu_id = int(os.environ["LOCAL_RANK"]) # get the local rank from the environment variable
        self.model = model.to(self.gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.epochs_run = 0
        if os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(snapshot_path)
        self.model = DDP(self.model, device_ids=[self.gpu_id]) # wrap the model with DDP

    def _load_snapshot(self, snapshot_path):
        # a snapshot is a dictionary containing the model state and the number of epochs run
        snapshot =  torch.load(snapshot_path)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        self.optimizer.load_state_dict(snapshot["OPTIMIZER_STATE"])
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")
        self.model.train() # set the model to training mode

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

    def _save_snapshot(self, epoch):
        snapshot = {}
        snapshot["MODEL_STATE"] = self.model.module.state_dict()
        snapshot["EPOCHS_RUN"] = epoch
        snapshot["OPTIMIZER_STATE"] = self.optimizer.state_dict()
        PATH = "snapshot.pt"
        torch.save(snapshot, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs): # to continue training from a snapshot, set max_epochs to a higher number
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0: # only save from one GPU
                self._save_snapshot(epoch)


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


def main(total_epochs: int, save_every: int, snapshot_path: str="snapshot.pt"):
    ddp_setup() # initialize the process group
    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size=32)
    trainer = Trainer(model, train_data, optimizer, save_every, snapshot_path=snapshot_path)
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
    
    main(args.total_epochs, args.save_every) # CUDA_VISIBLE_DEVICES=0,3 
    #torchrun --nproc_per_node=2 multi_gpu_torchrun.py 50 10 --batch_size 32