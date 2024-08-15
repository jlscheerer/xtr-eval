import argparse

from xtr.config import XTRScaNNIndexConfig, XTRFAISSIndexConfig, XTRBruteForceIndexConfig
from xtr.datasets import LoTTE, LoTTEDataset, BEIR, BEIRDataset

from utility.build_index import build_index

def parse_dataset(collection, dataset, split, type_):
    if collection is None or dataset is None or split is None:
        return None
    if collection == "beir":
        return BEIRDataset(dataset=BEIR(dataset), datasplit=split)
    if type_ is None:
        type_ = "search"
    return LoTTEDataset(dataset=LoTTE(dataset), datasplit=split, type_=type_)

def get_dataset(parser, args):
    dataset = parse_dataset(collection=args.collection, dataset=args.dataset, split=args.split, type_=args.type)
    if dataset is None:
        parser.error("invalid dataset specified.")
    return dataset

def parse_index_config(index):
    if index == "scann":
        return XTRScaNNIndexConfig()
    elif index == "faiss":
        return XTRFAISSIndexConfig()
    elif index == "bruteforce":
        return XTRBruteForceIndexConfig()
    return None

def get_index_config(parser, args):
    index_config = parse_index_config(index=args.index)
    if index_config is None:
        parser.error("invalid index config specified.")
    return index_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="xtr-eval",
        description="Evaluation Tool for DeepMind's XTR"
    )

    parser.add_argument("mode", choices=["index"], nargs=1)
    parser.add_argument("-c", "--collection", choices=["beir", "lotte"])
    parser.add_argument("-d", "--dataset")
    parser.add_argument("-t", "--type", choices=["search", "forum"])
    parser.add_argument("-s", "--split", choices=["train", "test", "dev"])
    parser.add_argument("-i", "--index", choices=["scann", "faiss", "bruteforce"])
    parser.add_argument("-m", "--max_num_tokens", type=int)
    args = parser.parse_args()

    assert len(args.mode) == 1
    mode = args.mode[0]

    if mode == "index":
        dataset, index_config = get_dataset(parser, args), get_index_config(parser, args)
        build_index(dataset, index_config, max_num_tokens=args.max_num_tokens)
    else: raise AssertionError