from generation import Generation
from mtc_model import Seq2Seq, p as params, tokenizer, bert, get_transformer
from transformers import pipeline, AutoTokenizer
import torch.nn.functional as F

import torch
from queue import PriorityQueue
import operator

from pytorch_lightning import Trainer, seed_everything
import argparse
import readline
import questionary
from termcolor import colored
from lm_pretrain import pad_collate
from translation_model import EncDecTranslation

nlp = pipeline("feature-extraction", model='bert-base-uncased', tokenizer='bert-base-uncased', device=0)


def get_latest_ckpt():
    from glob import glob
    from regex import match
    ckpt = sorted(glob('lightning_logs/version*/checkpoints/last.ckpt'),
                  key=lambda x: int(match(r'.*/version_([0-9]+)/.*', x)[1]))[-1]
    return ckpt


def get_ckpt_version(version, dir='lightning_logs'):
    from glob import glob
    from regex import match
    ckpt = sorted(glob(f'{dir}/version_{version}/checkpoints/last.ckpt'),
                  key=lambda x: int(match(r'.*/version_([0-9]+)/.*', x)[1]))[-1]
    return ckpt


def generate_until_eos(model, hidden, input, context, max_size):
    tokens = torch.Tensor()
    with torch.no_grad():
        while True:
            predicted, hidden = model.decoder.forward(input, hidden, context)
            softmax_output = F.softmax(predicted, 1)
            log_prob, indexes = torch.topk(softmax_output, k=1)
            tokens = torch.cat([tokens, indexes.cpu()], 1)
            input = indexes.cuda()
            if tokens.size()[1] == max_size:
                break
    tokens = tokens.int()
    return tokens


def generate_lm(ckpt, prefix, num_layers, hidden_size):
    model = Generation.load_from_checkpoint(ckpt).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model.hparams['model'])
    model.eval()
    v = {'text': prefix}
    _, yy_pad, _, _ = pad_collate([v], add_eos=False)

    def get_hidden_for_prefix(model, token_seq):
        hidden = torch.zeros(num_layers, 1, hidden_size, device='cuda')
        context = torch.zeros(1, 1, hidden_size, device='cuda')
        with torch.no_grad():
            for token in token_seq:
                decoder_input = torch.tensor([token]).cuda()
                _, hidden = model.decoder.forward(decoder_input, hidden, context)
        return hidden

    hidden_for_generation = get_hidden_for_prefix(model, yy_pad[0][:-1])  # without last token which will be the input for the generation
    input_for_generation = torch.tensor([yy_pad[0][-1]]).cuda()
    context_for_generation = torch.zeros(1, 1, hidden_size, device='cuda')
    output = generate_until_eos(model, hidden_for_generation, input_for_generation, context_for_generation, max_size=20)
    return tokenizer.batch_decode(output, skip_special_tokens=True)


def generate(text, k=1, model: Seq2Seq = None,
             ckpt=None, mask_feature=None, mask_loc=None,
             bert=None, tokenizer=None, limit_func=None):
    if model is None:
        model = Generation.load_from_checkpoint(ckpt or get_latest_ckpt()).cuda()

    tokens_out = tokenizer(text, return_tensors='pt')
    tokens = tokens_out['input_ids'].numpy().tolist()[0]
    if mask_feature is None:
        if tokenizer.mask_token_id not in tokens:
            print("ERROR: couldn't find MASK token.")
            return []

        mask_loc = tokens.index(tokenizer.mask_token_id)
        if mask_loc >= 128:
            print("ERROR: string too long. Skipping.")
            return []

        features = bert(**tokens_out.to(device='cuda'))[0][0].cpu()
        mask_feature = features[mask_loc]

    num_beams = k

    position_ids = None
    if model.hparams.get('use_positions', False):
        position_ids = torch.tensor([mask_loc] * num_beams).cuda()

    t0 = torch.tensor(mask_feature).unsqueeze(0).unsqueeze(0).cuda()
    if model.hparams.get('translation', False):
        t0 = model.translation(t0)
    t0 = t0.repeat(1, num_beams, 1)

    hidden = model.get_init_hidden(num_beams, t0)
    prev = tokens[mask_loc - 1] if model.hparams.get('use_prev_token', False) else tokenizer.bos_token_id
    # prev = tokenizer.bos_token_id
    decoder_input = torch.tensor([prev]).unsqueeze(0).cuda()
    output = model.generate(input_ids=decoder_input,
                            bos_token_id=tokenizer.bos_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            hidden=hidden,
                            context=t0,
                            position_ids=position_ids,
                            num_beams=num_beams,
                            num_return_sequences=num_beams,
                            prefix_allowed_tokens_fn=limit_func
                            )
    return tokenizer.batch_decode(output[:, 1:], skip_special_tokens=True)

