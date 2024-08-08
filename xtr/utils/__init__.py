from xtr.config import XTRIndexConfig
from xtr.datasets.dataset import Dataset
from xtr.utils.tracker import ExecutionTracker

def canonical_index_name(dataset: Dataset, index_config: XTRIndexConfig):
    return f"{dataset.name}.{index_config.name}"

def xtr_tracker(name: str):
    return ExecutionTracker(name=name, steps=["Query Encoding", "Candidate Generation",
                                              "Estimate Missing Similarity", "Aggregate Scores"])