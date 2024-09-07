from xtr.config import XTRIndexConfig
from xtr.datasets.dataset import Dataset
from xtr.utils.tracker import ExecutionTracker

def canonical_index_name(dataset: Dataset, index_config: XTRIndexConfig):
    return f"{dataset.name}.{index_config.name}"

def xtr_tracker(name: str):
    return ExecutionTracker(name=name, steps=["Query Encoding", "search_batched", "enumerate_scores", "Estimate Missing Similarity",
                                              "get_did2scores", "add_ems", "get_final_score", "sort_scores"])