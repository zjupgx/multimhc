import argparse
import numpy as np
import re
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, DataLoader
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from pathlib import Path
aa_idx = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9, 'M': 10,
          'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19, 'X': 20}

script_path = Path(__file__).parent.absolute()
edge_index = np.load(Path.joinpath(script_path, 'models/edge_index.npy'))
edge_index = torch.tensor(edge_index)
re_hla = re.compile(r'(HLA-[ABC])(\d{2}\:\d{2})')
with open(Path.joinpath(script_path, 'models/MHC_pseudo.dat')) as f:
    lines = f.read().splitlines()
hla_dic = {}
for line in lines[1:]:
    hla, seq = line.split()
    if re_hla.match(hla):
        hla = '*'.join(re_hla.match(hla).groups())
        hla_dic[hla] = seq


def build_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input', metavar='input.csv',
                        help='Input file for prediction with columns "peptide" and "HLA".', required=True)
    parser.add_argument('-o', '--output', metavar='output.csv',
                        help='Output file for results')
    parser.add_argument(
        '-d', '--device',
        choices=['auto', 'cpu', 'gpu'],
        default='auto',
        help='Device to use: auto (default), cpu, or gpu'
    )
    return parser


def resolve_device(device_arg):
    if device_arg == 'cpu':
        return torch.device('cpu'), False

    if device_arg == 'gpu':
        if not torch.cuda.is_available():
            raise RuntimeError(
                'GPU was requested (--device gpu) but CUDA is not available. '
                'Please check running enviroment.'
            )
        return torch.device('cuda:0'), True

    # auto
    if torch.cuda.is_available():
        return torch.device('cuda:0'), True
    else:
        return torch.device('cpu'), False


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) /
            float(max(1, num_training_steps - num_warmup_steps))
        )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


class Combine_dataset(Dataset):
    def __init__(self, gnn_data, rnn_data):
        self.gnn_x = gnn_data
        self.rnn_x = torch.tensor(rnn_data)
        assert len(gnn_data) == len(rnn_data)

    def __len__(self):
        return len(self.gnn_x)

    def __getitem__(self, index):
        data = {'gnn_input': self.gnn_x[index],
                'rnn_input': self.rnn_x[index], }
        return data


class BipartiteData(Data):
    def __init__(self, edge_index=None, x_s=None, x_t=None):
        super().__init__()
        self.edge_index = edge_index
        self.x_s = x_s
        self.x_t = x_t

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index':
            return torch.tensor([[self.x_s.size(0)], [self.x_t.size(0)]])
        else:
            return super().__inc__(key, value, *args, **kwargs)


def parse_data(seq_idx):
    res_idx = seq_idx
    x_s = res_idx[:11]
    x_t = res_idx[11:]
    data = BipartiteData(torch.LongTensor(edge_index),
                         torch.LongTensor(x_s), torch.LongTensor(x_t))
    return data


class Process():
    def __init__(self, seqs, labels=None, seq_len=49):
        self.seqs = seqs
        self.labels = labels
        self.seq_len = seq_len

    @staticmethod
    def pad_seq(seq, seq_len):
        pep = seq[:11]
        hla = seq[11:]
        seq = hla+pep
        if len(seq) < seq_len:
            padding_len = seq_len - len(seq)
            padding_seq = 'X'*padding_len
            seq = seq + padding_seq
        else:
            seq = seq[:seq_len]
        assert len(seq) == seq_len
        return seq

    def seqs_aa2idx(self):
        seqs_idx = []
        for seq in self.seqs:
            seq = self.pad_seq(seq, seq_len=self.seq_len)
            seqs_idx.append([aa_idx[aa] for aa in seq])
        return torch.LongTensor(seqs_idx)


class Fusion_datamodule(pl.LightningDataModule):
    def __init__(self, batch_size, seqs):
        super().__init__()
        self.batch_size = batch_size
        self.seqs = seqs

    def prepare_data(self):
        gnn_val = []
        for seq in self.seqs:
            seq_idx = [aa_idx[aa] for aa in seq]
            gnn_data = parse_data(seq_idx)
            gnn_val.append(gnn_data)

        process = Process(self.seqs)
        rnn_val = process.seqs_aa2idx()
        self.val_dataset = Combine_dataset(gnn_data=gnn_val, rnn_data=rnn_val)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
