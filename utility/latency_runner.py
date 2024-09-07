import os
# Enforces CPU-only execution of torch
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Configure environment to ensure single-threaded execution.
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"]= "1"
os.environ["OMP_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)

from utility.executor_utils import read_subprocess_inputs, publish_subprocess_results
from utility.runner_utils import make_index_config, make_dataset

from xtr.config import XTRConfig, XTRModel
from xtr.utils import xtr_tracker, canonical_index_name
from xtr.modeling.xtr import XTR

if __name__ == "__main__":
    config, params = read_subprocess_inputs()
    index_config, dataset = make_index_config(config), make_dataset(config)

    index_name = canonical_index_name(dataset=dataset, index_config=index_config)
    xtr_config = XTRConfig(index_name=index_name, model=XTRModel.BASE_EN, index_config=index_config, override=False)
    xtr = XTR(config=xtr_config, collection=dataset.collection, device=torch.device("cpu"))
    tracker = xtr_tracker(name=index_name)
    rankings = xtr.retrieve_docs(dataset.queries, document_top_k=config["document_top_k"],
                                 token_top_k=config["token_top_k"], tracker=tracker)
    metrics = dataset.eval(rankings)
    
    publish_subprocess_results({"tracker": tracker.as_dict(), "metrics": metrics})
