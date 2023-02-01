import argparse
import pandas as pd
from collections import defaultdict
from configuration import DATA_PATH, HOME_DIR
import ast

import torch
import datasets
import random
from pytorch_lightning.loggers import TensorBoardLogger
from s3fs import S3FileSystem

from torch import nn, no_grad
from lr_scheduler.warmup_reduce_lr_on_plateau_scheduler import WarmupReduceLROnPlateauScheduler
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from transformers import AutoTokenizer, BertForMaskedLM, BertTokenizer, PreTrainedTokenizerFast, pipeline, \
    default_data_collator, BertModel, AutoModelForMaskedLM, RobertaForMaskedLM, AutoModel
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import os
from tqdm import tqdm
from generate import Generator, generate

from ray.tune.integration.pytorch_lightning import TuneReportCallback

p = dict(
    batch_size=128,
    # max_epochs=3,
    hidden_size=768,
    # hidden_size=768 * 4,
    num_layers=2,
    dropout=0.2,
    learning_rate=0.001,
    freeze_bert=False,
    teacher_forcing=0.5,
    split_layers=False,
    context_as_hidden=True,
    # context_as_input=False,
    context_as_input=True,
    use_prev_token=True,
    use_positions=True,
    translation=False,
    complex_translation=False,
    model='bert-base-cased',
    # model='roberta-base',
    load_from_pretrain=False,
    epochs=10,
)

version = f'pretrained_model_{p["model"]}'

case_sens = 'cased'


def get_transformer(model_name):
    bert: BertForMaskedLM = AutoModelForMaskedLM.from_pretrained(model_name)
    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(model_name)

    tokenizer.add_tokens(['[SOS]', '[EOS]'], special_tokens=True)
    tokenizer.bos_token = '[SOS]'
    tokenizer.eos_token = '[EOS]'

    bert.resize_token_embeddings(len(tokenizer))
    return bert, tokenizer


bert, tokenizer = get_transformer(p["model"])
p['output_size'] = len(tokenizer)


def set_transformer(model_name):
    global bert, tokenizer
    bert, tokenizer = get_transformer(model_name)
    p['output_size'] = len(tokenizer)


def encode_with_sos(text):
    return tokenizer.encode(f'[SOS] {text} [EOS]', add_special_tokens=False)


def replace_with_sos(ids):
    return tokenizer.encode(f'[SOS] {tokenizer.decode(ids, skip_special_tokens=True)} [EOS]', add_special_tokens=False)


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


class SplitEmbedding(nn.Module):
    def __init__(self):
        super(SplitEmbedding, self).__init__()
        self.bert_embedding = bert.bert.embeddings
        self.extra_embedding = nn.Embedding(2, bert.bert.embeddings.word_embeddings.embedding_dim)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None,
                past_key_values_length=0):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.bert_embedding.position_ids[:,
                           past_key_values_length: seq_length + past_key_values_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.bert_embedding.position_ids.device)

        if inputs_embeds is None:
            batch = input_ids
            mask = batch >= self.bert_embedding.word_embeddings.num_embeddings

            pretrained_batch = batch.clone()
            pretrained_batch[mask] = 0

            embedded_batch = self.bert_embedding.word_embeddings(pretrained_batch)

            # Every token without representation has to be brought into appropriate range
            batch -= self.bert_embedding.word_embeddings.num_embeddings
            # Zero out the ones which already have pretrained embedding
            batch[~mask] = 0
            non_pretrained_embedded_batch = self.extra_embedding(batch)

            # And finally change appropriate tokens from placeholder embedding created by
            # pretrained into trainable embeddings.
            embedded_batch[mask] = non_pretrained_embedded_batch[mask]
            inputs_embeds = embedded_batch
        token_type_embeddings = self.bert_embedding.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.bert_embedding.position_embedding_type == "absolute":
            position_embeddings = self.bert_embedding.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.bert_embedding.LayerNorm(embeddings)
        embeddings = self.bert_embedding.dropout(embeddings)

        return embeddings


class RNNDecoder(nn.Module):
    def __init__(self,
                 hidden_size,
                 num_layers,
                 dropout,
                 output_size,
                 embedding_size,
                 split_layers,
                 context_as_hidden,
                 context_as_input,
                 model_name,
                 config={}):
        super(RNNDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.output_size = output_size
        self.embedding_size = embedding_size
        self.split_layers = split_layers
        self.context_as_hidden = context_as_hidden
        self.context_as_input = context_as_input

        input_sz = embedding_size * 2 if self.context_as_input else embedding_size

        self.rnn = nn.GRU(input_size=input_sz,
                          hidden_size=hidden_size,
                          num_layers=num_layers
                          , dropout=dropout
                          )
        self.transform = nn.Linear(embedding_size * 2 + hidden_size, hidden_size)
        self.act = nn.functional.gelu
        bert, _ = get_transformer(model_name)
        if 'roberta' in model_name:
            self.linear = bert.lm_head
        else:
            self.linear = bert.cls

        if config.get('init_embeddings', False):
            bert.init_weights()
            print('INIT WEIGHTS')

        if self.split_layers:
            self.embedding = SplitEmbedding()
            self.linear_extra = nn.Linear(bert.cls.predictions.decoder.in_features, 2, bias=False)
            self.bias = nn.Parameter(torch.zeros(2))
            self.linear_extra.bias = self.bias
        elif 'roberta' in model_name:
            self.embedding = bert.roberta.embeddings
        else:
            self.embedding = bert.bert.embeddings

    def forward(self, input, hidden, context, position_ids=None):
        # input = [batch size] or [batch_size, seq_len]
        # position_ids = [batch size]
        # hidden = [num_layers, batch size, embedding size]
        # context = [1, batch size, embedding size]

        if input.dim() == 1:
            input = input.unsqueeze(1)  # [batch size, 1 (seq len)]
        if position_ids is not None:
            position_ids = position_ids.unsqueeze(1)  # [1, batch size]

        embedded = self.embedding(input, position_ids=position_ids).permute(1, 0, 2)  # [1, batch, emb size]
        if self.context_as_input:
            emb_con = torch.cat((embedded, context), dim=-1)
        else:
            emb_con = embedded

        output, hidden = self.rnn(emb_con, hidden)  # TODO: check adding layerNorm and skip connection
        # output = [seq len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]

        decoder_output = torch.cat((output.squeeze(0), context.squeeze(0), embedded.squeeze(0)),
                                   dim=1)  # [batch size, emb dim + hid dim * 2]

        output = self.transform(decoder_output)  # [batch size, hid dim]
        output = self.act(output)  # [batch size, hid dim]

        if self.split_layers:
            prediction = self.linear.predictions.transform(output)
            prediction = torch.cat((self.linear.predictions.decoder(prediction), self.linear_extra(prediction)), dim=-1)
        else:
            prediction = self.linear(output)  # [batch size, output dim]

        return prediction, hidden


class Seq2Seq(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.save_hyperparameters(config)

        self.learning_rate = config['learning_rate']
        self.warmup_learning_rate = config.get('warmup_learning_rate', 1e-8)
        self.num_layers = config['num_layers']
        self.teacher_forcing = config['teacher_forcing']
        self.context_as_hidden = config.get('context_as_hidden', False)

        self.optimizer_patience = config.get('optimizer_patience', 2)
        self.optimizer_warmup_steps = config.get('optimizer_warmup_steps', 2)
        self.optimizer_factor = config.get('optimizer_factor', 0.2)
        self.lm_pretrain = config.get('lm_pretrain', False)

        if config.get('translation', False):
            if config.get('complex_translation', False):
                self.translation = nn.Sequential(nn.Linear(config['hidden_size'], config['hidden_size'] * 4), nn.GELU(),
                                                 nn.LayerNorm(config['hidden_size'] * 4),
                                                 nn.Linear(config['hidden_size'] * 4, config['hidden_size']))
            else:
                self.translation = nn.Linear(config['hidden_size'], config['hidden_size'], bias=False)
                torch.nn.init.eye_(self.translation.weight)

        self.decoder = RNNDecoder(
            hidden_size=config['hidden_size'],
            num_layers=self.num_layers,
            dropout=config['dropout'],
            output_size=config['output_size'],
            embedding_size=768,
            split_layers=config['split_layers'],
            context_as_hidden=self.context_as_hidden,
            context_as_input=config.get('context_as_input', not self.context_as_hidden),
            model_name=config['model'],
            config=config
        )

        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, encoder_output, trg, teacher_forcing_ratio=0, position_ids=None):
        # encoder_output = [batch size, embedding size]
        # trg = [batch size, trg len]
        # position_ids = [batch size]

        encoder_output = encoder_output.unsqueeze(0)  # [1, batch size, embedding size]
        if self.hparams.get('translation', False):
            encoder_output = self.translation(encoder_output)

        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_size

        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size, device='cuda')

        # context also used as the initial hidden state of the decoder
        hidden = self.get_init_hidden(batch_size, encoder_output)

        # first input to the decoder is the <sos> tokens
        input = trg[:, 0]  # [batch size]

        for t in range(1, trg_len):
            # insert input token embedding, previous hidden state and the context state
            # receive output tensor (predictions) and new hidden state
            output, hidden = self.decoder(input, hidden, encoder_output, position_ids)
            if self.lm_pretrain:
                hidden = repackage_hidden(hidden)
            # place predictions in a tensor holding predictions for each token
            outputs[t] = output

            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1 = output.argmax(1)
            input = trg[:, t] if teacher_force else top1
            if position_ids is not None:
                position_ids = torch.add(position_ids, 1)
            # input = output.argmax(1)

        return outputs

    def configure_optimizers(self):
        if not self.lm_pretrain:
            return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        else:
            optimizer = torch.optim.Adam(self.parameters(), self.warmup_learning_rate)
            scheduler = WarmupReduceLROnPlateauScheduler(
                optimizer,
                init_lr=self.warmup_learning_rate,
                peak_lr=self.learning_rate,
                warmup_steps=self.optimizer_warmup_steps,  # this is steps of the scheduler not training steps!
                patience=self.optimizer_patience,
                factor=self.optimizer_factor,
            )
            lr_dict = {
                # REQUIRED: The scheduler instance
                'scheduler': scheduler,
                # The unit of the scheduler's step size, could also be 'step'.
                # 'epoch' updates the scheduler on epoch end whereas 'step'
                # updates it after a optimizer update.
                'interval': 'step',
                # How many epochs/steps should pass between calls to
                # `scheduler.step()`. 1 corresponds to updating the learning
                # rate after every epoch/step.
                'frequency': 3000,
                # Metric to to monitor for schedulers like `ReduceLROnPlateau`
                'monitor': 'val_loss',
                # If set to `True`, will enforce that the value specified 'monitor'
                # is available when the scheduler is updated, thus stopping
                # training if not found. If set to `False`, it will only produce a warning
                'strict': True,
                # If using the `LearningRateMonitor` callback to monitor the
                # learning rate progress, this keyword can be used to specify
                # a custom logged name
                'name': None,
                # to get monitored value
                'reduce_on_plateau': True,
            }
            return {"optimizer": optimizer, "lr_scheduler": lr_dict}

    def training_step(self, batch, batch_idx):
        x, y_padded, y_lens, positions = batch
        # y_padded = [batch_size, trg len]

        y_hat = self(x, y_padded, teacher_forcing_ratio=self.teacher_forcing,
                     position_ids=positions)  # [trg len, batch size, output dim]

        y_padded = y_padded.permute(1, 0)  # [trg_len, batch size]

        output_dim = y_hat.shape[-1]

        y_hat = y_hat[1:].view(-1, output_dim)  # [(trg len - 1) * batch size, output dim]
        y_padded = y_padded[1:].contiguous().view(-1)  # [(trg len - 1) * batch size]

        loss = self.criterion(y_hat, y_padded)
        # result = pl.TrainResult(loss)
        # result.log('train_loss', loss)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.lm_pretrain:
            x, y_padded, y_lens, positions = batch
            y_hat = self(x, y_padded, teacher_forcing_ratio=1,
                         position_ids=positions)  # [trg len, batch size, output dim]
            y_padded = y_padded.permute(1, 0)  # [trg_len, batch size]
            output_dim = y_hat.shape[-1]
            y_hat = y_hat[1:].view(-1, output_dim)  # [(trg len - 1) * batch size, output dim]
            y_padded = y_padded[1:].contiguous().view(-1)  # [(trg len - 1) * batch size]
            loss = self.criterion(y_hat, y_padded)
            self.log('val_loss', loss, prog_bar=True)
            return loss
        else:
            x, orig_y_padded, y_lens, positions = batch
            # orig_y_padded = [batch_size, trg len]

            y_hat = self(x, orig_y_padded, teacher_forcing_ratio=0,
                         position_ids=positions)  # [trg len, batch size, output dim]

            y_padded = orig_y_padded.permute(1, 0)  # [trg_len, batch size]

            output_dim = y_hat.shape[-1]
            y_hat = y_hat[1:].view(-1, output_dim)  # [(trg len - 1) * batch size, output dim]

            y_padded = y_padded[1:].contiguous().view(-1)  # [(trg len - 1) * batch size]

            loss = self.criterion(y_hat, y_padded)
            self.log('val_loss', loss, prog_bar=True)

            batch_size = orig_y_padded.shape[0]
            hidden = self.get_init_hidden(batch_size, x)
            input = orig_y_padded[:, 0:1]  # [batch size, 1]
            x = x.unsqueeze(0)
            NUM_RES = 5
            res = generate(self, bert.config, tokenizer, input, hidden, context=x, position_ids=positions, num_beams=NUM_RES)
            spans = tokenizer.batch_decode(orig_y_padded, skip_special_tokens=True)
            found = 0
            for i, span in enumerate(spans):
                if span in res[i * NUM_RES: i * (NUM_RES + 1)]:
                    found += 1
            self.log('suc@5', found / batch_size, on_step=False, on_epoch=True, prog_bar=True)
            return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        result = pl.EvalResult()
        result.log('test_loss', loss)
        return result

    def get_init_hidden(self, batch_size, encoder_output=None):
        if self.context_as_hidden:
            hidden = encoder_output.repeat(self.decoder.num_layers, 1, 1)  # [1, batch size, embedding size]
        else:
            hidden = torch.zeros(self.num_layers, batch_size, self.decoder.hidden_size,
                                 device='cuda')  # [1, batch size, embedding size]
        return hidden


def parse_data(dataset_name='wiki', dataset_suffix=''):
    bert: BertModel = AutoModel.from_pretrained(p['model']).cuda()
    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(p['model'])
    if dataset_name == 'wiki_pub':
        s3_prefix = f'{DATA_PATH}/MultiTokenCompletion/mlm_dataset/publication_data'
        train_dataset = datasets.DatasetDict(
            {k: datasets.Dataset.from_pandas(pd.read_csv(f'{s3_prefix}/{k}_set.csv', na_filter=False)) for k in ['train', 'dev', 'test']})
    elif dataset_name == 'wiki_unseen':
        train_dataset = datasets.load_dataset('csv',
                                              data_files={
                                                  'train': 'data/28_6_dataset/train_set.csv',
                                                  'dev': 'data/28_6_dataset/seen_dev_set.csv',
                                                  'test': 'data/28_6_dataset/seen_test_set.csv',
                                              }, na_filter=False)
    elif dataset_name == 'pubmed':
        s3_prefix = f'{DATA_PATH}/MultiTokenCompletion/pubmed_dataset_freq_50'
        train_dataset = datasets.DatasetDict(
            {k: datasets.Dataset.from_pandas(pd.read_csv(f'{s3_prefix}/pubmed_{k}_set.csv', na_filter=False)) for k in ['train', 'dev', 'test']})
    elif dataset_name == 'pubmed_ext':
        s3_prefix = f'{DATA_PATH}/MultiTokenCompletion/pubmed_mtc_dataset'
        train_dataset = datasets.DatasetDict(
            {k: datasets.Dataset.from_pandas(pd.read_csv(f'{s3_prefix}/{k}_set.csv', na_filter=False)) for k in ['train', 'dev', 'test']})
    elif dataset_name == 'pubmed_large':
        s3_prefix = f'{DATA_PATH}/MultiTokenCompletion/pubmed_mtc_dataset_large_92K_dataset'
        train_dataset = datasets.DatasetDict(
            {k: datasets.Dataset.from_pandas(pd.read_csv(f'{s3_prefix}/{k}_set.csv', na_filter=False)) for k in ['train', 'dev', 'test']})
    elif dataset_name == 'pubmed_largespec':
        s3_prefix = f'{DATA_PATH}/MultiTokenCompletion/pubmed_mtc_dataset_large_without_reg'
        train_dataset = datasets.DatasetDict(
            {k: datasets.Dataset.from_pandas(pd.read_csv(f'{s3_prefix}/{k}_set.csv', na_filter=False)) for k in ['train', 'dev', 'test']})
    else:
        raise Exception(f'parsing of dataset {dataset_name} is not defined')

    MASK_TOKEN = tokenizer.mask_token

    if 'roberta' in p['model']:
        train_dataset = train_dataset.map(lambda masked_text: {'masked_text': [v.replace('[MASK]', MASK_TOKEN) for v in masked_text]},
                                          batched=True,
                                          num_proc=16,
                                          input_columns=['masked_text'])

    MASK_TOKEN_ID = tokenizer.mask_token_id

    def get_mask_loc(examples):
        return {'mask_loc': [v.index(MASK_TOKEN_ID) for v in tokenizer(examples['masked_text'], padding=True)['input_ids']]}

    print(train_dataset)
    dataset_w_maskloc = train_dataset.map(
        get_mask_loc,
        batched=True,
        load_from_cache_file=True,
        remove_columns=['nps', 'range', 'span_lower', 'row_num']
    )

    dataset_w_maskloc128 = dataset_w_maskloc.filter(
        lambda ex: ex['mask_loc'] < 126,
        num_proc=16,
        load_from_cache_file=True
    )

    dataset_w_maskloc128.save_to_disk(f'data/preprocessed_data_{p["model"]}{dataset_suffix}')
    dataset_w_maskloc128 = datasets.load_from_disk(f'data/preprocessed_data_{p["model"]}{dataset_suffix}')

    print("DATASET", dataset_w_maskloc128)

    def to_features(samples):
        masked_text = samples['masked_text']
        mask_loc = samples['mask_loc']
        tokenized = tokenizer(masked_text, return_tensors='pt', padding='max_length', truncation=True,
                              max_length=128)
        with torch.no_grad():
            features = bert(**tokenized.to(device='cuda'))[0].cpu()

        return {'input_features': [f[l].numpy().tolist() for f, l in zip(features, mask_loc)],
                'prev_token': [int(tokenized['input_ids'][i][l - 1]) for i, l in enumerate(mask_loc)]
                }

    def to_toks(samples):
        masked_text = samples['masked_text']
        mask_loc = samples['mask_loc']
        tokenized = tokenizer(masked_text, return_tensors='pt', padding='max_length', truncation=True, max_length=128)

        return {'prev_token': [tokenized['input_ids'][i][l - 1] for i, l in enumerate(mask_loc)]
                }

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    input_features = dataset_w_maskloc128.map(
        to_features,
        batched=True,
        batch_size=256,
    )
    input_features.save_to_disk(f'data/input_data_{p["model"]}{dataset_suffix}')


def pad_collate(batch, use_prev_token=False, use_positions=False):
    xx = [v['input_features'] for v in batch]
    positions = []
    for v in batch:
        v['prev'] = v['prev_token'] if use_prev_token else tokenizer.bos_token_id
        if use_positions:
            positions.append(v['mask_loc'])
    span_name = 'span_lower' if case_sens == 'uncased' else 'span'
    yy = [[v['prev'], *tokenizer(str(v[span_name]), add_special_tokens=False)['input_ids'],
           tokenizer.eos_token_id] for v in batch]

    y_lens = [len(y) for y in yy]
    yy_pad = pad_sequence(
        [torch.tensor(y, dtype=torch.long) for y in yy],
        batch_first=True,
        padding_value=0)  # [batch_size, padded_y]

    return torch.Tensor(xx), yy_pad, y_lens, torch.LongTensor(positions) if use_positions else None


seed_everything(1)


@no_grad()
def eval_test(trainer, dataset, acc_at_to_check=(1, 2, 3, 5, 10)):
    found, total = defaultdict(int), 0
    model: Seq2Seq = trainer.lightning_module.eval()
    test_dataset = dataset['test']
    for row in tqdm(test_dataset):
        results = generate(model, bert.config, tokenizer,
                           [row['prev_token'] if model.hparams['use_prev_token'] else tokenizer.bos_token_id],
                           model.get_init_hidden(model.hparams['batch_size'], row['input_features']),
                           context=row['input_features'], position_ids=row['mask_loc'],
                           num_beams=max(acc_at_to_check))

        total += 1
        span_name = 'span'
        span = tokenizer.decode(tokenizer.encode(str(row[span_name])), skip_special_tokens=True)
        for acc_at in acc_at_to_check:
            if span in results[:acc_at]:
                found[acc_at] += 1

    acc = {k: v / total for k, v in found.items()}
    return acc


def train(config=p, input_features=None, num_gpus=1, tune=False):
    # csv_logger = CSVLogger('./', name='lstm', version='0'),

    # if not config['split_layers']:
    #     bert.resize_token_embeddings(len(tokenizer))
    config['output_size'] = len(AutoTokenizer.from_pretrained(config['model'])) + 2
    config['case'] = case_sens

    callbacks = None
    if tune:
        callbacks = [TuneReportCallback({"loss": "val_loss"}, on="validation_end")]

    trainer = Trainer(
        gpus=num_gpus
        , logger=TensorBoardLogger(save_dir=os.getcwd(), version=version, name='eacl2022_models')
        , max_epochs=config['epochs']
        , val_check_interval=config['val_check_interval']
        , gradient_clip_val=1
        , callbacks=callbacks
    )

    if config['ckpt'] is not None:
        model = Seq2Seq.load_from_checkpoint(config['ckpt'])
    elif config['load_from_pretrain']:
        config['context_as_hidden'] = True
        config['context_as_input'] = False
        if config['model'].startswith('roberta'):
            model = Seq2Seq.load_from_checkpoint(
                'lm_pretrain_roberta/epoch=0-step=779999.ckpt', config=config,
                strict=False)
        else:
            model = Seq2Seq.load_from_checkpoint(
                '../MultiTokenCompletion/lm_pretrain/version_1999/checkpoints/epoch=0-step=713999.ckpt', config=config,
                strict=False)
    else:
        model = Seq2Seq(config)

    freeze_bert = config['freeze_bert']
    if freeze_bert:
        for name, param in model.decoder.embedding.named_parameters():
            param.requires_grad = False
        for name, param in model.decoder.linear.named_parameters():
            param.requires_grad = False

    # freeze decoder
    if p['translation']:
        for name, param in model.decoder.named_parameters():
            param.requires_grad = False

    use_prev_token = config.get('use_prev_token', False)
    use_positions = config.get('use_positions', False)

    trainer.fit(model,
                DataLoader(input_features['train'], config['batch_size'], num_workers=16, shuffle=True,
                           collate_fn=lambda b: pad_collate(b, use_prev_token, use_positions)
                           ),
                DataLoader(input_features['test'].select(range(config['dev_size'])), 32, num_workers=16,
                           collate_fn=lambda b: pad_collate(b, use_prev_token, use_positions)))
    return trainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str)
    parser.add_argument("--model", type=str, default='bert-base-cased')
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--input_path", type=str, default=None)
    parser.add_argument("--freeze_bert", type=ast.literal_eval, default=False)  # Hack: python3.6 doesn't support bool natively
    parser.add_argument("--split_layers", type=ast.literal_eval, default=False)  # Hack: python3.6 doesn't support bool natively
    parser.add_argument("--context_as_hidden", type=ast.literal_eval, default=True)  # Hack: python3.6 doesn't support bool natively
    parser.add_argument("--use_prev_token", type=ast.literal_eval, default=True)  # Hack: python3.6 doesn't support bool natively
    parser.add_argument("--use_positions", type=ast.literal_eval, default=True)  # Hack: python3.6 doesn't support bool natively
    parser.add_argument("--load_from_pretrain", type=ast.literal_eval, default=True)  # Hack: python3.6 doesn't support bool natively
    parser.add_argument("--context_as_input", type=ast.literal_eval, default=False)  # Hack: python3.6 doesn't support bool natively
    parser.add_argument("--translation", type=ast.literal_eval, default=False)  # Hack: python3.6 doesn't support bool natively
    parser.add_argument("--complex_translation", type=ast.literal_eval, default=False)  # Hack: python3.6 doesn't support bool natively
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--dev_size", type=int, default=20000)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--teacher_forcing", type=float, default=0.5)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--val_check_interval", type=float, default=0.2)
    parser.add_argument("--init_embeddings", action='store_true')
    parser.add_argument("--dataset_name", type=str, default='wiki_pub')
    parser.add_argument("--data_only", type=ast.literal_eval, default=False)
    args = parser.parse_args()
    args_dict = vars(args)
    print(args_dict)

    set_transformer(args.model)

    for k, v in args_dict.items():
        if v is not None:
            p[k] = v
    if args.version is not None:
        version = args.version
    dataset_name = args.dataset_name
    dataset_suffix = '' if dataset_name == 'wiki' else f'_{dataset_name}'

    if args.input_path is not None:
        input_path = args.input_path
    elif 'roberta' in p['model'] or 'spanbert' in p['model'].lower():
        input_path = f'{HOME_DIR}/MultiTokenCompletionData/input_data_{p["model"]}{dataset_suffix}'
    else:
        input_path = f'data/input_data_{p["model"]}{dataset_suffix}'

    print("input_path")

    if not os.path.exists(input_path):
        print(f"PATH {input_path} wasn't found, creating data")
        parse_data(dataset_name, dataset_suffix)

    if not args.data_only:
        fs = S3FileSystem() if input_path.startswith('s3://') else None
        input_features = datasets.load_from_disk(input_path, fs=fs)
        print(input_features)

        train(args_dict, input_features)
