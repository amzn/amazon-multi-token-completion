from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration
import torch
import pandas as pd
from collections import defaultdict
import time

from configuration import DATA_PATH


def _filter(output, end_token='<extra_id_1>'):
    # The first token is <unk> (inidex at 0) and the second token is <extra_id_0> (indexed at 32099)
    _txt = t5_tokenizer.decode(output[2:], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    if end_token in _txt:
        _end_token_index = _txt.index(end_token)
        return _txt[:_end_token_index].lower()
    else:
        return _txt.lower()


def get_top_k_predictions(input_ids, topk):
    if limit_vocab:
        outputs = t5_mlm.generate(input_ids=input_ids,
                                  num_beams=topk, num_return_sequences=topk,
                                  max_length= 3 + 10,
                                  eos_token_id=32098,
                                  pad_token_id=32098,
                                  forced_eos_token_id=32098,
                                  prefix_allowed_tokens_fn=my_allowed_function
                                  )
    else:
        outputs = t5_mlm.generate(input_ids=input_ids,
                                  num_beams=topk, num_return_sequences=topk,
                                  max_length= 3 + 10,
                                  eos_token_id=32098,
                                  pad_token_id=32098,
                                  forced_eos_token_id=32098,
                                  )
    results = list(map(_filter, outputs))
    return results

def report_progress(current, gap_to_report, total, current_single_token, didnt, found, acc_at_to_check, found_single_token):
    if current % gap_to_report == 0 and current != 0:
        print("using model ", T5_PATH, " running on ", INPUT_DATA_DIR)
        if limit_vocab:
            print("limiting vocab to vocab of size ", len(allowed_vocab_token_sequences))
        print("finished ", current, " out of ", total, " couldn't handle ", didnt)
        print(time.localtime())
        for acc_at in acc_at_to_check:
            print("accuracy at ", str(acc_at), " is:", (100 * found[acc_at]) / (current - didnt), "(",
                  found[acc_at], " out of ", (current - didnt), ")")
            if current_single_token > 0:
                print("accuracy at ", str(acc_at), " on single tokens only is:", (100 * found_single_token[acc_at]) / current_single_token, "(",
                      found_single_token[acc_at], " out of ", current_single_token, ")")


def eval_mlm_predictions(path_to_load, path_to_save, acc_at_to_check):
    data = pd.read_csv(path_to_load)
    data = data.sample(frac=1)
    found, found_single_token, total_terms = defaultdict(int), defaultdict(int), 0
    did_not_handle, current_single_token = 0, 0
    results, correct, is_single_token = [], [], []
    for ix, row in data.iterrows():
        report_progress(current=total_terms, gap_to_report=500, total=len(data),
                        current_single_token=current_single_token,
                        didnt=did_not_handle, found=found, acc_at_to_check=acc_at_to_check,
                        found_single_token=found_single_token)
        if len(t5_tokenizer.tokenize(row['masked_text'])) > 128:
            did_not_handle += 1
            results.append("")
            correct.append(0)
            is_single_token.append(False)
            continue
        total_terms += 1
        label = str(row['span']).lower()
        if len(t5_tokenizer.tokenize(label)) == 1:
            current_single_token += 1
            is_single_token.append(True)
        else:
            is_single_token.append(False)
        masked = row['masked_text'].replace("[MASK]", "<extra_id_0>")
        context_ids = t5_tokenizer.encode_plus(masked, add_special_tokens=True, return_tensors='pt')['input_ids'].to(device)
        top_predictions = get_top_k_predictions(context_ids, acc_at_to_check[-1])
        for acc_at in acc_at_to_check:
            if label.strip() in top_predictions[:acc_at]:
                found[acc_at] += 1
                if len(t5_tokenizer.tokenize(label)) == 1:
                    found_single_token[acc_at] += 1
            if acc_at == acc_at_to_check[-1]:
                if label.strip() in top_predictions[:acc_at]:
                    correct.append(1)
                else:
                    correct.append(0)
        results.append(top_predictions)
    report_progress(current=total_terms - did_not_handle, gap_to_report=1, total=total_terms,
                    current_single_token=current_single_token, didnt=did_not_handle,
                    found=found, acc_at_to_check=acc_at_to_check, found_single_token=found_single_token)
    new_data = pd.DataFrame(
        {'masked': data['masked_text'][:len(results)], "span": data['span'][:len(results)], 'results': results,
         'correct_at_' + str(acc_at_to_check[-1]): correct, 'is_single_token':is_single_token})
    new_data.to_csv(path_to_save, index=False)
    print("saved results to ", OUTPUT_RESULTS_FILE)


def measure_time():
    for model_name in ['t5-base', 't5-3b']:
        T5_PATH = model_name
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        t5_tokenizer = T5Tokenizer.from_pretrained(T5_PATH)
        t5_config = T5Config.from_pretrained(T5_PATH)
        print("loading model")
        t5_mlm = T5ForConditionalGeneration.from_pretrained(T5_PATH, config=t5_config).to(device)
        t5_mlm.eval()
        print("finished loading model ", model_name)
        for num_of_examples in [500,1000]:
            data = pd.read_csv(f"{DATA_PATH}/MultiTokenCompletion/pubmed_dataset_freq_50/pubmed_test_set.csv").head(num_of_examples)
            with torch.no_grad():
                tic = time.perf_counter()
                for ix, row in data.iterrows():
                    masked = row['masked_text'].replace("[MASK]", "<extra_id_0>")
                    context_ids = t5_tokenizer.encode_plus(masked, add_special_tokens=True, return_tensors='pt')[
                        'input_ids'].to(device)
                    outputs = t5_mlm.generate(input_ids=context_ids,
                                              num_beams=10, num_return_sequences=10,
                                              max_length=3 + 10,
                                              eos_token_id=32098,
                                              pad_token_id=32098,
                                              forced_eos_token_id=32098,
                                              )
                toc = time.perf_counter()
            print("For model ", model_name, " inferencing the first ", num_of_examples, " test examples took: "  f" {toc - tic:0.4f} seconds")
            print(f"Avg single inference time is {(toc - tic)/num_of_examples:0.4f} seconds")


def get_completion_token_sequences_for_t5(allowed_sequences):
    # [:-1] since encode adds space in the end, 0 since T5 starts completions with 0, 32099 and 32098 is eos:
    allowed_token_sequences = [[0, 32099] + t5_tokenizer.encode(s)[:-1] + [32098] for s in allowed_sequences]
    return [token_list for token_list in allowed_token_sequences if not "<unk>" in t5_tokenizer.decode(token_list)]


def get_prefix2allowed_tokens_dic(allowed_token_sequences):
    dic = defaultdict(set)
    for seq in allowed_token_sequences:
        for prefix_idx in range(len(seq)):
            dic[tuple(seq[:prefix_idx])].add(seq[prefix_idx])
    print("amount of possible prefixes is: ", len(dic))
    return dic


def load_vocab(data_file):
    df = pd.read_csv(data_file)
    vocab = df['span'].unique().tolist()
    return [str(i) for i in vocab]


def my_allowed_function(batch_id, input_ids):
    return list(prefix2allowed_next_token_dic[tuple(input_ids.tolist())])

if __name__ == '__main__':
    # measure_time()
    # params for specific run
    T5_PATH = 't5-base'  # "t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"
    acc_at_to_check = [1, 3, 5, 10, 50]
    INPUT_DATA_DIR = f"{DATA_PATH}/MultiTokenCompletion/mlm_dataset/publication_data/"
    OUTPUT_RESULTS_FILE = f"{DATA_PATH}/MultiTokenCompletion/t5_results/t5_base_mlm_dataset_publication_data_results_limited.csv"
    limit_vocab = True

    # initialization
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t5_tokenizer = T5Tokenizer.from_pretrained(T5_PATH)
    t5_config = T5Config.from_pretrained(T5_PATH)
    print("loading model")
    t5_mlm = T5ForConditionalGeneration.from_pretrained(T5_PATH, config=t5_config).to(device)
    print("finished loading model")

    if limit_vocab:
        # preparing the prefix to allowed tokens dictionary for limited completion
        vocabulary = load_vocab(INPUT_DATA_DIR + "unmasked_data.csv")
        allowed_vocab_token_sequences = get_completion_token_sequences_for_t5(vocabulary)
        print("vocabulary size is: ", len(allowed_vocab_token_sequences))
        prefix2allowed_next_token_dic = get_prefix2allowed_tokens_dic(allowed_vocab_token_sequences)

    # run completions
    eval_mlm_predictions(INPUT_DATA_DIR + "test_set.csv", OUTPUT_RESULTS_FILE, acc_at_to_check=acc_at_to_check)
