## MTC: Multi Token Completion

This package provides code for the paper *Simple and Effective Multi-Token Completion from Masked Language Models*.

Paper: [Simple and Effective Multi-Token Completion from Masked Language Models](https://aclanthology.org/2023.findings-eacl.179/)


Bibtex entry:
```
@inproceedings{DBLP:conf/eacl/KalinskyKLG23,
  author       = {Oren Kalinsky and
                  Guy Kushilevitz and
                  Alexander Libov and
                  Yoav Goldberg},
  title        = {Simple and Effective Multi-Token Completion from Masked Language Models},
  booktitle    = {Findings of the Association for Computational Linguistics: {EACL}
                  2023, Dubrovnik, Croatia, May 2-6, 2023},
  pages        = {2311--2324},
  publisher    = {Association for Computational Linguistics},
  year         = {2023},
  url          = {https://aclanthology.org/2023.findings-eacl.179},
  timestamp    = {Mon, 08 May 2023 14:38:37 +0200},
  biburl       = {https://dblp.org/rec/conf/eacl/KalinskyKLG23.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```


## Steps to run the code
### Data
Download data from [here](https://registry.opendata.aws/multi-token-completion/). 

### Configuration
* pip install -r requirements.txt
* pip install transformers==4.5.1
* Update home_dir and data_path in `configuration.py`

### Training

* __RNN decoder__ - run `mtc_model.py --input_path data/input_data_bert-base-cased_wiki_pub/`
* __EMAT decoder__ - run `matrix_plugin.py --input_path data/input_data_bert-base-cased_wiki_pub/`

### Testing

* __RNN decoder__ - run `test.py --dataset_path data/input_data_bert-base-cased_wiki_pub/ --ckpt <CHECKPOINT_PATH> --all`
* __EMAT decoder__ - run `matrix_plugin.py --input_path data/input_data_bert-base-cased_wiki_pub/  --ckpt <CHECKPOINT_PATH> --test`
* __T5 baseline__ - run `T5_constrained_generation.py`

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.
