from xtr.config import XTRScaNNIndexConfig
from xtr.datasets import LoTTE, LoTTEDataset, BEIR, BEIRDataset
from xtr.utils import canonical_index_name

dataset = BEIRDataset(dataset=BEIR.SCIFACT, datasplit="test")
index_config = XTRScaNNIndexConfig()

print(canonical_index_name(dataset=dataset, index_config=index_config))