import os
# Enforces CPU-only execution of torch
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Configure environment to ensure single-threaded execution.
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"]= "1"
os.environ["OMP_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)

from xtr.datasets import BEIR, BEIRDataset, LoTTE, LoTTEDataset
from xtr.config import XTRConfig, XTRModel, XTRScaNNIndexConfig, XTRBruteForceIndexConfig, XTRFAISSIndexConfig
from xtr.utils import xtr_tracker, canonical_index_name
from xtr.modeling.xtr import XTR
from xtr.modeling.xtr_opt import XTROpt

import json
from datetime import datetime

NUM_RUNS_PER_EXPERIMENT = 3

def current_time_str():
    return datetime.today().strftime('%Y-%m-%d-%H-%M-%S')

def xtr_eval_latency(dataset, index_config, document_top_k, token_top_k, opt):
    index_name = canonical_index_name(dataset=dataset, index_config=index_config)
    config = XTRConfig(index_name=index_name, model=XTRModel.BASE_EN, index_config=index_config, override=False)
    if not opt:
        xtr = XTR(config=config, collection=dataset.collection, device=torch.device("cpu"))
    else:
        xtr = XTROpt(config=config, collection=dataset.collection, device=torch.device("cpu"))
    tracker = xtr_tracker(name=index_name, opt=opt)
    rankings = xtr.retrieve_docs(dataset.queries, document_top_k=document_top_k, token_top_k=token_top_k, tracker=tracker)
    return tracker, dataset.eval(rankings)

def xtr_run_configuration(dataset, index_config, document_top_k, token_top_k, opt):
    tracker, metrics = xtr_eval_latency(dataset, index_config, document_top_k, token_top_k, opt)
    configuration = {"dataset": dataset.name, "index": index_config.name,
                     "document_top_k": document_top_k, "token_top_k": token_top_k}
    return {
        "config": configuration,
        "metrics": metrics,
        "tracker": tracker.as_dict()
    }

def xtr_run_configurations(datasets, index_configs, document_top_k, token_top_k_values, opt, label):
    ctime = current_time_str()
    os.makedirs("results", exist_ok=True)
    filename = os.path.join("results", f"run_{label}_{ctime}.json")
    results = []
    for dataset in datasets:
        for index_config in index_configs:
            for token_top_k in token_top_k_values:
                results.append(xtr_run_configuration(dataset, index_config, document_top_k=document_top_k, token_top_k=token_top_k, opt=opt))
                with open(filename, "w") as file:
                    json.dump(results, file)