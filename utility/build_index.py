from typing import Optional

from xtr.config import XTRConfig, XTRIndexConfig, XTRModel
from xtr.datasets.dataset import Dataset
from xtr.modeling.xtr import XTR
from xtr.utils import canonical_index_name

def build_index(dataset: Dataset, index_config: XTRIndexConfig, max_num_tokens: Optional[int] = None, override: bool=False):
    index_name = canonical_index_name(dataset=dataset, index_config=index_config)
    config = XTRConfig(index_name=index_name, model=XTRModel.BASE_EN, index_config=index_config, max_num_tokens=max_num_tokens, override=override)
    _ = XTR(config=config, collection=dataset.collection)