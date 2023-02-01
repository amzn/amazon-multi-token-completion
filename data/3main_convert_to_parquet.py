from datasets import load_from_disk
import pyarrow.parquet as pq
import dask.dataframe as dd
from configuration import DATA_PATH

# Convert huggingface dataset to parquet chunked files that can be consumed by SPARK
for dname in ['books', 'wiki']:
    np_dataset_dir = f"{DATA_PATH}/data/{dname}_sents_np_dataset_fixed"
    np_dataset = load_from_disk(np_dataset_dir)
    pq.write_table(np_dataset.data, f'{DATA_PATH}/data/{dname}_sents_np_dataset_fixed.parquet')

    df = dd.read_parquet(f'{DATA_PATH}/data/{dname}_sents_np_dataset_fixed.parquet', chunksize="100MB")
    df = df.repartition(partition_size="100MB")
    dd.to_parquet(df, f'{DATA_PATH}/data/{dname}_sents_np_dataset_chunked_fixed/')

# Sample 50 sentences for all phrases using an EMR:
"""
WITH nps_table as (
(select regexp_replace(regexp_replace(npr['0'], '[|](NP|ENT)', ''), '_', ' ') as span, regexp_replace(regexp_replace(lower(npr['0']), '[|](np|ent)', ''), '_', ' ') as span_lower, npr['0'] as np, npr['1'] as range, text, nps from books LATERAL VIEW explode(arrays_zip(split(nps, ' '), split(np_ranges, ' '))) tbl AS npr where nps != '')
union all
(select regexp_replace(regexp_replace(npr['0'], '[|](NP|ENT)', ''), '_', ' ') as span, regexp_replace(regexp_replace(lower(npr['0']), '[|](np|ent)', ''), '_', ' ') as span_lower, npr['0'] as np, npr['1'] as range, text, nps from wiki LATERAL VIEW explode(arrays_zip(split(nps, ' '), split(np_ranges, ' '))) tbl AS npr where nps != '')
),
dup_sentences as (
select span, span_lower, np, range, text, nps, row_number() over (partition by text order by rand()) as row_num from nps_table
),
distinct_sentences as (
select span, span_lower, np, range, text, nps from dup_sentences where row_num = 1
),
nps_shuffled as (
select *, row_number() over (partition by span_lower order by rand()) as row_num from distinct_sentences
)
select * from nps_shuffled where row_num <= 50 order by span_lower, row_num;
"""

# Filter to only sentence with at last 100 freq with an EMR:
"""
create temporary view top_phrases_w_spaces as (
select *, regexp_replace(phrase, '_', ' ') as span from top_phrases
);

cache table top_phrases_w_spaces;

select sentence_samples.*, top.freq, top.num_of_missing_tokens, top.num_of_words from sentence_samples, top_phrases_w_spaces as top where top.span = sentence_samples.span order by freq DESC, phrase, row_num;
"""
