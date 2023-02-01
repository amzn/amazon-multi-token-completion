from configuration import HOME_DIR
from mtc_model import get_transformer
import datasets
from predict import generate, Generation, get_ckpt_version, get_latest_ckpt
from collections import defaultdict
from tqdm import tqdm
import argparse
import os

acc_at_to_check = [1, 2, 3, 5, 10, 20, 50]
NUM_RESULTS = max(acc_at_to_check)
PRINT_EVERY = 500


def my_allowed_function(batch_id, input_ids):
    return list(prefix2allowed_next_token_dic.get(tuple(input_ids[1:].tolist()), []))


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--ckpt', type=str)
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--log', action='store_true')
    parser.add_argument('--unseen', action='store_true')
    parser.add_argument('--dev', action='store_true')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('-v', '--version', type=int)
    parser.add_argument('--case', type=str, default="cased")
    parser.add_argument('--limit_vocab', action='store_true')
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

    logger = Logger(
        f'{os.path.dirname(os.path.dirname(ckpt))}/{"unseen" if args.unseen else "seen"}_{"dev" if args.dev else "test"}{"_all" if args.all else ""}.out')

    # Loading model
    model = Generation.load_from_checkpoint(ckpt).cuda()
    model_name = model.hparams['model']
    bert, tokenizer = get_transformer(model_name)

    found, total = defaultdict(int), 0

    # datafile = f'data/{"unseen" if args.unseen else "seen"}_{"dev" if args.dev else "test"}_set.csv'
    dataset_name = f'{"unseen_" if args.unseen else ""}{"dev" if args.dev else "test"}'

    if model_name.startswith('roberta'):
        input_path = f'{HOME_DIR}/MultiTokenCompletionData/input_data_{model_name}'
        bert = bert.roberta
    else:
        if 'spanbert' in model_name.lower():
            input_path = f'{HOME_DIR}/MultiTokenCompletionData/input_data_{model_name}'
        else:
            input_path = f'{HOME_DIR}/MultiTokenCompletionData/input_data_{args.case}'
        bert = bert.bert

    if args.dataset_path is None:
        dataset = datasets.load_from_disk(f'{HOME_DIR}/MultiTokenCompletionData/input_data_{args.case}')
    else:
        dataset = datasets.load_from_disk(args.dataset_path)
    test_dataset: datasets.Dataset = dataset[dataset_name]

    # bert to GPU
    bert = bert.cuda()

    if args.limit_vocab:
        vocab = set(test_dataset['span'])
        allowed_token_sequences = [v + [tokenizer.eos_token_id] for v in
                                   tokenizer.batch_encode_plus(list(vocab), add_special_tokens=False)['input_ids']]
        prefix2allowed_next_token_dic = defaultdict(set)
        for seq in allowed_token_sequences:
            for prefix_idx in range(len(seq)):
                prefix2allowed_next_token_dic[tuple(seq[:prefix_idx])].add(seq[prefix_idx])
        print("amount of possible prefixes is: ", len(prefix2allowed_next_token_dic))

    if not args.all:
        test_dataset = test_dataset.shuffle(42).select(range(5000))

    for row in tqdm(test_dataset):
        results = generate(row['masked_text'], k=NUM_RESULTS, model=model, mask_feature=None,
                           mask_loc=row['mask_loc'], bert=bert, tokenizer=tokenizer,
                           limit_func=my_allowed_function if args.limit_vocab else None)

        total += 1
        span_name = 'span_lower' if args.case == 'uncased' else 'span'
        span = tokenizer.decode(tokenizer.encode(str(row[span_name])), skip_special_tokens=True)
        for acc_at in acc_at_to_check:
            if span in results[:acc_at]:
                found[acc_at] += 1
        if args.log:
            logger.print({'success@10': 'V' if span in results[:10] else 'X',
                          'span': span,
                          'masked_text': row['masked_text'],
                          'top10': results[:10]})

        if total % PRINT_EVERY == 0:
            for acc_at in acc_at_to_check:
                logger.print(f"accuracy at {acc_at} is: {found[acc_at] / total:.2%} ({found[acc_at]} out of {total})")

    for acc_at in acc_at_to_check:
        logger.print(f"accuracy at {acc_at} is: {found[acc_at] / total:.2%} ({found[acc_at]} out of {total})")

    if args.log:
        logger.close()
