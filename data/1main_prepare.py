from datasets import load_dataset
import spacy
import json
from random import choice

from spacy.language import Language
from glob import glob
from configuration import DATA_PATH

SAMPLE = int(1e6)

name_to_dataset = {
    'books': ['bookcorpus'],
    'wiki': ['wikipedia', '20200501.en']
}


@Language.component('set_custom_boundaries')
def set_custom_boundaries(doc):
    for token in doc[:-1]:
        if '\n' in token.text:
            doc[token.i + 1].is_sent_start = True
    return doc


def get_masked_npchunks(id, sent):
    nlp = spacy.load('en_core_web_sm')
    nlp.add_pipe('set_custom_boundaries', first=True)
    doc = nlp(sent) if type(sent) == str else sent

    # Extract NP chunks
    doc_chunks = []
    for np in doc.noun_chunks:
        prefix = doc[:np.start].text
        suffix = doc[np.end:].text.rstrip("\n")

        res = {
            'doc_id': id, 'sentence': doc.text.rstrip("\n"),
            'nchunk': np.text,
            'single_mask': f'{prefix} [MASK]{np[-1].whitespace_}{suffix}',
            'range': str([np.start_char, np.end_char])
        }
        doc_chunks.append(res)
    return doc_chunks


def get_masked_npchunks_sent(id, sent):
    sents = get_filtered_sents(sent)

    sent_id = choice(range(len(sents)))
    res_list = get_masked_npchunks(id, sents[sent_id])
    return [{**res, 'sent_id': sent_id} for res in res_list]


def get_filtered_sents(nlp, s):
    doc = nlp(s)
    sents = [s.text.strip() for s in doc.sents if 5 <= len(s) <= 100 and not s.text.startswith('Category:') and 'VERB' in [t.pos_ for t in s]]
    if sents and 'may refer to:' in sents[0]:
        sents = []
    return sents


def samples_to_sentences(samples, inds):
    nlp = spacy.load('en_core_web_sm')
    nlp.add_pipe('set_custom_boundaries', first=True)
    with open(f'{DATA_PATH}/data/wiki_sents/{dname}_sentences{inds[0]}_{inds[-1]}.jsonl', 'w') as fp:
        for i, s in zip(inds, samples['text']):
            sents = get_filtered_sents(nlp, s)
            for sent in sents:
                fp.write(json.dumps({'ind': i, 'text': sent}))
                fp.write('\n')


for dname in name_to_dataset.keys():
    dataset = load_dataset(*name_to_dataset[dname])['train']
    dataset.map(samples_to_sentences, with_indices=True, batched=True, batch_size=len(dataset) // 100, num_proc=16)

    dataset = load_dataset('json', data_files={'train': glob(f'data/wiki_sents/{dname}_sentences*.jsonl')})
    dataset.save_to_disk(f'data/{dname}_sents_dataset/')
