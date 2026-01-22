import argparse
import torch
import numpy as np
import pandas as pd
from models import Fusion_PL
from utils import Fusion_datamodule, hla_dic, re_hla
import yaml
import sys
import time
import logging
import random
import os
from pathlib import Path
from pytorch_lightning import Trainer
from copy import deepcopy
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
script_path = Path(__file__).parent.absolute()


def seed_torch(seed=3407):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def process_data(df):
    df['length'] = df['peptide'].map(lambda x: len(x))
    cond = df['length'].isin([8, 9, 10, 11])
    df = df[cond]
    seqs = []
    for pep, hla in zip(df['peptide'], df['HLA']):
        if hla not in hla_dic.keys():
            logging.warning(f'{hla} is not in the training data.')
            continue
        diff_len = 11 - len(pep)
        pep += diff_len*'X'
        if re_hla.match(hla):
            hla = '*'.join(re_hla.match(hla).groups())
        hla_seq = hla_dic[hla]
        seq = pep.strip() + hla_seq.strip()
        seqs.append(seq)
    return seqs


def predict(device=None, argv=sys.argv[1:]):
    args = parser.parse_args(argv)
    input_file = args.input
    input_df = pd.read_csv(input_file)
    output_df = deepcopy(input_df)
    seqs = process_data(input_df)
    trainer = Trainer(accelerator="gpu" if torch.cuda.is_available() else "cpu",
                      enable_progress_bar=False,
                      logger=False)
    fold_prob_ls = []
    bsz = 256
    for fold_idx in range(1, 6):
        datamod = Fusion_datamodule(batch_size=bsz, seqs=seqs)
        datamod.prepare_data()
        test_loader = datamod.val_dataloader()
        hparam_f = Path.joinpath(script_path, 'models/hparams.yaml')
        with open(hparam_f) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        model = Fusion_PL(config=config, device=device)
        ckpt_file = Path.joinpath(
            script_path, f'models/model_fold{fold_idx}.ckpt')
        test_results = trainer.predict(model, dataloaders=test_loader, ckpt_path=ckpt_file)
        fold_prob_ls.append(torch.cat(test_results, dim=0).numpy())

    probs = np.mean(fold_prob_ls, axis=0)
    output_df['Prediction Score'] = probs
    if not args.output:
        now = time.strftime("%Y%m%d-%H_%M_%S", time.localtime(time.time()))
        output_file = f'{now}_predictions.csv'
        logging.warning(
            f'Output file is not specified, the results will be saved in {output_file}')
    else:
        output_file = args.output
    output_df.to_csv(output_file, index=False, float_format='%.6f')
    return output_df


if __name__ == '__main__':
    seed_torch()
    cuda_num = 0
    use_cuda = False
    DEVICE = torch.device(
        f'cuda:{cuda_num}') if use_cuda else torch.device('cpu')
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input', metavar='input.csv',
                        help='Input file for prediction with columns "peptide" and "HLA".', required=True)
    parser.add_argument('-o', '--output', metavar='output.csv',
                        help='Output file for results')
    predict(device=DEVICE)
