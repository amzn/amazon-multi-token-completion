import argparse
import time
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import csv

from dataclasses import dataclass, field, asdict
from pytorch_lightning.core.memory import ModelSummary
from pytorch_lightning.loggers import TensorBoardLogger
from torch import no_grad
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM, Trainer, BertTokenizerFast, TrainingArguments, BertTokenizer, BertForMaskedLM, \
    HfArgumentParser, RobertaForMaskedLM, BertForSequenceClassification
import datasets
import os
from tqdm import tqdm
import pandas as pd
import torch
import pytorch_lightning as pl
import torch
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from typing import Dict


def create_mapping(dataset, model_name, dataset_path, model_path, dataset_suffix):
    tok: BertTokenizerFast = AutoTokenizer.from_pretrained(model_name)
    seen_dataset = datasets.DatasetDict({k: v for k, v in dataset.items() if k in ['train', 'dev', 'test']})
    seen_dataset = seen_dataset.map(
        lambda samples: {
            'span': [s if s is not None else '|'.join(n.split('|')[:-1]) for s, n in zip(samples['span'], samples['np'])]},
        batched=True, num_proc=8)
    vocab = set(seen_dataset['train'].unique('span')) | set(seen_dataset['dev'].unique('span')) | set(seen_dataset['test'].unique('span'))
    mapping = tok.vocab
    cur_id = len(mapping)
    for v in tqdm(vocab):
        if v not in mapping:
            mapping[v] = cur_id
            cur_id += 1
    seen_dataset = seen_dataset.map(lambda samples: {'labels': [mapping[v] for v in samples['span']]}, batched=True, num_proc=16)

    seen_dataset.save_to_disk(dataset_path)
    model: BertForMaskedLM = AutoModelForMaskedLM.from_pretrained(model_name)
    model.resize_token_embeddings(len(mapping))
    model.save_pretrained(model_path)

    pd.Series(mapping).to_csv(f'matrix_plugin/ext_dec_map_{model_name.replace("/", "_")}{dataset_suffix}.csv', index_label='phrase', header=['id'])
    return model, seen_dataset, mapping


class MatrixDecoder(pl.LightningModule):
    def __init__(self, config, model):
        super().__init__()

        self.save_hyperparameters(config)
        self.config = config
        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, input_features):
        return self.model(input_features)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config['lr'])

    def step(self, batch):
        y = self.forward(batch['input_features'])
        loss = self.criterion(y, batch['labels'])
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log('val_loss', loss, prog_bar=True)
        return loss


def train(config: Dict, transformer_matrix: BertOnlyMLMHead, dataset: datasets.DatasetDict):
    trainer = pl.Trainer(
        gpus=config['num_gpus']
        , logger=TensorBoardLogger(save_dir=os.getcwd(), version=config['version'], name='matrix_plugin_results')
        , max_epochs=config['epochs']
        , val_check_interval=config['val_check_interval']
        , gradient_clip_val=1
    )

    model = MatrixDecoder(config, transformer_matrix)
    dataset.set_format(type='pt', columns=['input_features', 'labels'])
    trainer.fit(model,
                DataLoader(dataset['train'], config['batch_size'], num_workers=16, shuffle=True),
                DataLoader(dataset['dev'].select(range(config['dev_size'])), 32, num_workers=16))

    return trainer.checkpoint_callback.best_k_models[0]


class Logger:
    def __init__(self, logfile):
        print("output to:", logfile)
        self.logfile = logfile
        self.fp = open(logfile, 'w')

    def print(self, output):
        print(output)
        self.fp.write(str(output) + '\n')

    def close(self):
        print("output printed to:", self.logfile)
        self.fp.close()


@torch.no_grad()
def test(matrix, dataset: datasets.DatasetDict, mapping: Dict, tokenizer: BertTokenizerFast, ckpt_path, log=False):
    test_model = MatrixDecoder.load_from_checkpoint(ckpt_path, model=matrix).cuda().eval()
    test_dataset = seen_dataset['test']
    rev_map = {v: k for k, v in mapping.items()}
    acc_at_to_check = [1, 2, 3, 5, 10, 20, 50]
    found, total = defaultdict(int), 0
    PRINT_EVERY = 20

    logger = Logger(Path(ckpt_path).parent.parent / 'test.log')
    if log:
        top5_fp = open(Path(ckpt_path).parent.parent / 'top5_res.csv', 'w')
        writer = csv.DictWriter(top5_fp, fieldnames=['top5_match', 'masked', 'span', 'top5_results'])
        writer.writeheader()

    BATCH_SZ = 128
    loader = DataLoader(test_dataset, batch_size=BATCH_SZ, shuffle=False)
    for irow, row in enumerate(tqdm(loader)):
        results = torch.topk(test_model.forward(torch.stack(row['input_features']).type(torch.FloatTensor).T.cuda()), k=max(acc_at_to_check), dim=-1)
        text_res = [[rev_map[int(i)] for i in res] for res in results.indices]

        total += len(row['span'])
        spans = tokenizer.batch_decode(tokenizer(row['span'])['input_ids'], skip_special_tokens=True)
        for i, span in enumerate(spans):
            if log:
                writer.writerow(
                    dict(top5_match='V' if span in text_res[i][:5] else 'X', masked=row['masked_text'][i], span=span, top5_results=text_res[i][:5]))

            for acc_at in acc_at_to_check:
                if span in text_res[i][:acc_at]:
                    found[acc_at] += 1

        if irow % PRINT_EVERY == 0:
            for acc_at in acc_at_to_check:
                logger.print(f"accuracy at {acc_at} is: {found[acc_at] / total:.2%} ({found[acc_at]} out of {total})")

    for acc_at in acc_at_to_check:
        logger.print(f"accuracy at {acc_at} is: {found[acc_at] / total:.2%} ({found[acc_at]} out of {total})")

    if log:
        top5_fp.close()


@torch.no_grad()
def benchmark(matrix, model, model_name, tokenizer: BertTokenizerFast, ckpt_path):

    test_df = pd.read_csv('data/28_6_dataset/seen_test_set.csv', na_filter=False)
    sample_test = test_df['masked_text'].iloc[:1000]
    if 'roberta' in model_name:
        sample_test = sample_test.str.replace('[MASK]', '<mask>')

    print("starting benchmark of 1000 samples")
    tic = time.perf_counter()
    for masked in sample_test:
        context_ids = tokenizer.encode_plus(masked, add_special_tokens=True, return_tensors='pt').to('cuda')
        results = torch.topk(model(**context_ids)['logits'], k=10, dim=-1)
    toc = time.perf_counter()
    print(f"inferencing the first 1000 test examples took: {toc - tic:0.4f} seconds")
    print(f"Avg single inference time is {(toc - tic) / 1000:0.4f} seconds")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='bert-base-cased')
    parser.add_argument('--version', type=str, default='version_0')
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--dataset_name', type=str, default='wiki_pub')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--val_check_interval', type=float, default=0.25)
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--dev_size', type=int, default=20_000)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--no_log', action='store_true')
    parser.add_argument('--benchmark', action='store_true')
    args = parser.parse_args()
    model_name = args.model
    args.log = not args.no_log
    config = vars(args)

    dataset_name = args.dataset_name
    dataset_suffix = '' if dataset_name == 'wiki' else f'_{dataset_name}'

    if args.benchmark:
        model = AutoModelForMaskedLM.from_pretrained(model_name).cuda()
        tok = AutoTokenizer.from_pretrained(model_name)
        ckpt = args.ckpt
        model = AutoModelForMaskedLM.from_pretrained(model_name).cuda()
        model.resize_token_embeddings(208915)
        matrix = model.lm_head if 'roberta' in model_name else model.cls
        matrix.eval()
        test_model = MatrixDecoder.load_from_checkpoint(ckpt, model=matrix).cuda().eval()

        if 'roberta' in model_name:
            # RobertaForMaskedLM
            model.lm_head = matrix
        else:
            # BertForMaskedLM
            model.cls = matrix

        print(ckpt)
        benchmark(matrix, model.eval(), model_name, tok, ckpt)
        exit(0)

    model_path = f'matrix_plugin/ext_dec_{model_name}{dataset_suffix}'
    dataset_path = f'data/matrix_plugin/seen_dataset_{model_name}{dataset_suffix}/'

    if args.force or not os.path.exists(model_path) or not os.path.exists(dataset_path):
        if args.input_path is not None:
            input_path = args.input_path
        elif 'roberta' in model_name:
            input_path = f'{HOME_DIR}/MultiTokenCompletionData/input_data_{model_name}{dataset_suffix}'
        elif 'spanbert' in model_name:
            input_path = f'input_data_{model_name}'
        else:
            input_path = f'{HOME_DIR}/MultiTokenCompletionData/input_data_cased'

        dataset = datasets.load_from_disk(input_path)

        print('increasing vocab')
        model, seen_dataset, mapping = create_mapping(dataset, model_name, dataset_path, model_path, dataset_suffix)
    else:
        print('loading vocab')
        model = AutoModelForMaskedLM.from_pretrained(model_name)
        seen_dataset = datasets.load_from_disk(dataset_path)
        mapping = pd.read_csv(f'matrix_plugin/ext_dec_map_{model_name.replace("/", "_")}{dataset_suffix}.csv', na_filter=False).set_index('phrase')[
            'id'].to_dict()
        model: BertForMaskedLM = AutoModelForMaskedLM.from_pretrained(model_path)

    if 'roberta' in model_name:
        # RobertaForMaskedLM
        matrix = model.lm_head
    else:
        # BertForMaskedLM
        matrix = model.cls

    tok = AutoTokenizer.from_pretrained(model_name)

    if args.test:
        matrix.eval()
        ckpt = args.ckpt
        if ckpt is None:
            from glob import glob

            ckpt = sorted(glob(f'matrix_plugin_results/{args.version}/checkpoints/*.ckpt'))[-1]
        print(ckpt)
        test(matrix, seen_dataset, mapping, tok, ckpt, args.log)
    else:
        # detach output embedding matrix so it will train
        model.get_output_embeddings().weight = torch.nn.Parameter(model.get_output_embeddings().weight.clone())

        best_ckpt = train(config, matrix, seen_dataset)
        matrix.eval()
        test(matrix, seen_dataset, mapping, tok, best_ckpt, args.log)
