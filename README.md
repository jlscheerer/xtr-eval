# Baseline Evaluation of Google DeepMind's XTR

------------

> We build on the code provided by Google DeepMind to evaluate XTR. This evaluation serves as the baseline for the _highly optimized_ [XTR/WARP](https://github.com/jlscheerer/xtr-warp) retrieval engine.

## Installation

xtr-eval requires Python 3.8+, PyTorch 1.9+ and Tensorflow 2.8.2 and uses the [Hugging Face Transformers](https://github.com/huggingface/transformers) library.
We evaluate XTR using the [`XTR_base` checkpoint](https://huggingface.co/google/xtr-base-en) provided on Hugging Face.


It is strongly recommended to create a [conda environment](https://docs.anaconda.com/anaconda/install/linux/#installation) using the commands below. We include the corresponding environment file (`environment.yml`).

```sh
conda activate xtr-eval
source ./scripts/build_indexes.sh
```

### Environment Setup
To construct indexes and perform retrieval, define the following values in a `config.yml` file in the repository root:
```yaml
BEIR_COLLECTION_PATH: "..."
LOTTE_COLLECTION_PATH: "..."
```

- `BEIR_COLLECTION_PATH`: Designates the path to the datasets of the [BEIR Benchmark](https://github.com/beir-cellar/beir).
- `LOTTE_COLLECTION_PATH`: Specifies the path to the [LoTTE dataset](https://github.com/stanford-futuredata/ColBERT/blob/main/LoTTE.md).

#### BEIR Benchmark

To download and extract a dataset from the BEIR Benchmark use the [`extract_collection.py`](https://github.com/jlscheerer/xtr-warp/blob/main/utility/extract_collection.py) script provided in [XTR/WARP](https://github.com/jlscheerer/xtr-warp):

```sh
python utility/extract_collection.py -d ${dataset} -i "${BEIR_COLLECTION_PATH}" -s test
```

Replace `${dataset}` with the desired dataset name as specified [here](https://github.com/beir-cellar/beir?tab=readme-ov-file#beers-available-datasets).

#### LoTTE Dataset

1. Download the LoTTE dataset files from [here](https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/lotte.tar.gz).
2. Extract the files manually to the directory specified in `LOTTE_COLLECTION_PATH`.