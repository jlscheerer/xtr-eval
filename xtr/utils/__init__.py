from xtr.config import XTRIndexConfig
from xtr.datasets.dataset import Dataset

def canonical_index_name(dataset: Dataset, index_config: XTRIndexConfig):
    return f"{dataset.name}.{index_config.name}"