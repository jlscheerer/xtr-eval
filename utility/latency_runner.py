import os
# Enforces CPU-only execution of torch
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from utility.executor_utils import read_subprocess_inputs, publish_subprocess_results

import psutil

if __name__ == "__main__":
    config, params = read_subprocess_inputs()

    num_threads = config["num_threads"]

    proc = psutil.Process()
    if "cpu_affinity" in params:
        # Set the cpu_affinity, e.g., [0, 1] for CPUs #0 and #1
        # Reference: https://psutil.readthedocs.io/en/latest/#psutil.Process.cpu_affinity
        proc.cpu_affinity(params["cpu_affinity"])

    # Configure environment to ensure *correct* number of threads.
    os.environ["MKL_NUM_THREADS"] = str(num_threads)
    os.environ["NUMEXPR_NUM_THREADS"]= str(num_threads)
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(num_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(num_threads)

    os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
    os.environ["KMP_AFFINITY"] = "disabled"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

    import torch
    torch.set_num_threads(num_threads)

    from utility.runner_utils import make_index_config, make_dataset

    from xtr.config import XTRConfig, XTRModel
    from xtr.utils import xtr_tracker, canonical_index_name
    from xtr.modeling.xtr import XTR

    index_config, dataset = make_index_config(config), make_dataset(config)

    index_name = canonical_index_name(dataset=dataset, index_config=index_config)
    xtr_config = XTRConfig(index_name=index_name, model=XTRModel.BASE_EN, index_config=index_config, override=False)
    if not config["optimized"]:
        from xtr.modeling.xtr import XTR
        xtr = XTR(config=xtr_config, collection=dataset.collection, device=torch.device("cpu"))
    else:
        from xtr.modeling.xtr_opt import XTROpt
        xtr = XTROpt(config=xtr_config, collection=dataset.collection, device=torch.device("cpu"))
    xtr.set_num_threads(num_threads)
    tracker = xtr_tracker(name=index_name)
    rankings = xtr.retrieve_docs(dataset.queries, document_top_k=config["document_top_k"],
                                 token_top_k=config["token_top_k"], tracker=tracker)
    metrics = dataset.eval(rankings)
    
    publish_subprocess_results({"tracker": tracker.as_dict(), "metrics": metrics})
