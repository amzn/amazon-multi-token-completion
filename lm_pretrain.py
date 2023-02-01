import os

from pytorch_lightning.loggers import TensorBoardLogger

from configuration import DATA_PATH
from mtc_model import Seq2Seq, get_transformer
from pytorch_lightning.callbacks import LearningRateMonitor
import torch
from torch.nn.utils.rnn import pad_sequence
import datasets
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
import predict
import numpy as np

DATA_SZ = 100_000_000

p = dict(
    batch_size=128,
    hidden_size=768,
    num_layers=2,
    dropout=0.2,

    learning_rate=0.0005,
    warmup_learning_rate=1e-8,
    optimizer_patience=3,
    optimizer_warmup_steps=3,
    optimizer_factor=0.25,

    freeze_bert=False,
    teacher_forcing=1,
    split_layers=False,
    context_as_hidden=False,
    context_as_input=False,
    use_prev_token=False,
    use_positions=False,
    lm_pretrain=True,
    model='roberta-base'
)

_, tokenizer = get_transformer(p['model'])
p['output_size'] = len(tokenizer)


def parse_data():
    data: datasets.Dataset = datasets.concatenate_datasets(
        [datasets.load_from_disk(f'data/{dname}_sents_dataset/')['train'] for dname in ['books', 'wiki']])
    data.shuffle(seed=42).select(range(DATA_SZ)).save_to_disk('data/pretraining_data')


def pad_collate(batch, add_eos=True):
    lens = [len(tokenizer.tokenize(v['text'])) for v in batch]
    median = int(np.median(np.array(lens)))
    yy = [[tokenizer.bos_token_id,
           *tokenizer(v['text'], add_special_tokens=False, truncation=True, max_length=median)['input_ids']] for v in batch]
    if add_eos:
        yy = [[*y, tokenizer.eos_token_id] if len(y) <= median else [*y] for y in
              yy]  # sentence ended, add eos, <= because median is computed without bos token
    y_lens = [len(y) for y in yy]
    yy_pad = pad_sequence(
        [torch.tensor(y, dtype=torch.long) for y in yy],
        batch_first=True,
        padding_value=0)  # [batch_size, padded_y]
    return torch.zeros([len(batch), p['hidden_size']]), yy_pad, y_lens, None


def train(config=p, num_epochs=3, num_gpus=1, tune=False):
    config['epochs'] = num_epochs

    lr_monitor = LearningRateMonitor()
    callbacks = [lr_monitor]
    if tune:
        callbacks.extend([TuneReportCallback({"loss": "val_loss"}, on="validation_end")])
    trainer = Trainer(
        logger=TensorBoardLogger(save_dir=os.getcwd(), version=2000, name='lm_pretrain_roberta'),
        gpus=num_gpus,
        max_epochs=num_epochs,
        val_check_interval=3000,
        gradient_clip_val=0.25,
        callbacks=callbacks
    )

    model = Seq2Seq(config)

    freeze_bert = config['freeze_bert']
    if freeze_bert:
        for name, param in model.decoder.embedding.named_parameters():
            param.requires_grad = False
        for name, param in model.decoder.linear.named_parameters():
            param.requires_grad = False

    input_features = datasets.load_from_disk(f'{DATA_PATH}/pretraining_data')
    input_features_train = input_features.select(range(99_900_000))
    input_features_val = input_features.select(range(99_900_000, 100_000_000))

    trainer.fit(model,
                DataLoader(input_features_train, config['batch_size'], num_workers=8, shuffle=True,
                           collate_fn=lambda b: pad_collate(b, add_eos=True)
                           ),
                val_dataloaders=DataLoader(input_features_val, config['batch_size'], num_workers=8,
                                           collate_fn=lambda b: pad_collate(b, add_eos=True)
                                           )
                )


def test_manual():
    while True:
        sent = input('write a sentence prefix: ')
        if sent == "exit":
            break
        res = predict.generate_lm("./lm_pretrain_roberta/epoch=0-step=779999.ckpt",
                                  sent, num_layers=p['num_layers'], hidden_size=p['hidden_size'])
        print("The suggested completion is:         ", " ".join(res))


if __name__ == '__main__':
    # parse_data()
    train(num_epochs=1)
    # test_manual()
