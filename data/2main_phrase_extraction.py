# this code is based on https://github.com/explosion/sense2vec/blob/b80070cd9de13be86732947ca9f78524c6668019/sense2vec/util.py#L105
from typing import Union, List, Tuple, Set
from spacy.tokens import Doc, Token, Span
from spacy.util import filter_spans
from datasets import load_dataset, load_from_disk
import re
import spacy
from configuration import DATA_PATH


def load_mt_dataset(dname):
    if dname == 'books':
        # return load_dataset('bookcorpus')
        return load_from_disk(f'{DATA_PATH}/data/books_sents_dataset/')
    elif dname == 'wiki':
        return load_from_disk(f'{DATA_PATH}/data/wiki_sents_dataset/')


def get_noun_phrases(doc: Doc) -> List[Span]:
    """Compile a list of noun phrases in sense2vec's format (without
    determiners). Separated out to make it easier to customize, e.g. for
    languages that don't implement a noun_chunks iterator out-of-the-box, or
    use different label schemes.
    doc (Doc): The Doc to get noun phrases from.
    RETURNS (list): The noun phrases as a list of Span objects.
    """
    trim_labels = ("advmod", "amod", "compound")
    spans = []
    if doc.has_annotation("DEP"):
        for np in doc.noun_chunks:
            while len(np) > 1 and np[0].dep_ not in trim_labels:
                np = np[1:]
            spans.append(np)
    return spans


def get_phrases(doc: Doc) -> Tuple[List[Span], List[Span]]:
    """Compile a list of sense2vec phrases based on a processed Doc: named
    entities and noun chunks without determiners.
    doc (Doc): The Doc to get phrases from.
    RETURNS (list): The phrases as a list of Span objects.
    """
    spans = []
    ents = list(doc.ents)
    ent_words: Set[str] = set()
    for span in ents:
        ent_words.update(token.i for token in span)
    for np in get_noun_phrases(doc):
        # Prefer entities over noun chunks if there's overlap
        if not any(w.i in ent_words for w in np):
            spans.append(np)
    return spans, ents


def make_key(word: str, sense: str) -> str:
    """Create a key from a word and sense, e.g. "usage_example|NOUN".
    word (unicode): The word.
    sense (unicode): The sense.
    RETURNS (unicode): The key.
    """
    text = re.sub(r"\s", "_", word)
    return text + "|" + sense


def extract_phrases(doc: Doc) -> Tuple[str, str]:
    """Extract noun phrases and named-entities from sentence
    """
    spans, ents = get_phrases(doc)
    spans = filter_spans(spans)
    span_ranges = [f'[{v.start_char},{v.end_char}]' for v in spans]
    spans = [make_key(v.text, 'NP') for v in spans]
    ents = filter_spans(ents)
    ent_ranges = [f'[{v.start_char},{v.end_char}]' for v in ents]
    ents = [make_key(v.text, 'ENT') for v in ents]

    return " ".join(spans + ents), " ".join(span_ranges + ent_ranges)


def sents_to_np(sents):
    nlp = spacy.load('en_core_web_sm')
    nps = []
    ranges = []
    for s in sents['text']:
        np, range = extract_phrases(nlp(s))
        nps.append(np)
        ranges.append(range)
    return {'nps': nps, 'np_ranges': ranges}


dname = 'wiki'
dataset = load_mt_dataset(dname)['train']

np_dataset_dir = f"{DATA_PATH}/data/{dname}_sents_np_dataset_fixed"
dataset.map(sents_to_np, batched=True, batch_size=100_000, num_proc=16).save_to_disk(np_dataset_dir)
