from mtc_model import get_transformer, Seq2Seq
from predict import generate, Generation, get_ckpt_version, get_latest_ckpt
import torch
import pandas as pd
import argparse
import os
import time

NUM_RESULTS = 10
PRINT_EVERY = 500


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
def bench_generate(text, k, model, bert, tokenizer):
    tokens_out = tokenizer(text, return_tensors='pt')
    tokens = tokens_out['input_ids'].numpy().tolist()[0]
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
                            num_return_sequences=num_beams
                            )
    return tokenizer.batch_decode(output[:, 1:], skip_special_tokens=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--ckpt', type=str)
    parser.add_argument('--log', action='store_true')
    args = parser.parse_args()

    ckpt = args.ckpt
    print('ckpt:', ckpt)
    if ckpt is None:
        if args.version is not None:
            ckpt = get_ckpt_version(args.version)
            print(f'using {ckpt}')
        else:
            ckpt = get_latest_ckpt()
            print(f'No checkpoint provided taking the latest: {ckpt}')

    # Loading model
    model = Generation.load_from_checkpoint(ckpt).cuda().eval()
    model_name = model.hparams['model']
    bert, tokenizer = get_transformer(model_name)

    if model_name.startswith('roberta'):
        bert = bert.roberta
    else:
        bert = bert.bert

    test_df = pd.read_csv('<PATH>/seen_test_set.csv', na_filter=False)
    sample_test = test_df['masked_text'].iloc[:1000]

    # bert to GPU
    bert = bert.cuda().eval()

    print("starting benchmark of 1000 samples")
    tic = time.perf_counter()
    for masked in sample_test:
        if model_name.startswith('roberta'):
            masked = masked.replace('[MASK]', '<mask>')
        results = bench_generate(masked, k=NUM_RESULTS, model=model, bert=bert, tokenizer=tokenizer)
    toc = time.perf_counter()
    print(f"inferencing the first 1000 test examples took: {toc - tic:0.4f} seconds")
    print(f"Avg single inference time is {(toc - tic) / 1000:0.4f} seconds")
