from abc import ABC
from dataclasses import dataclass
from enum import Enum

MAX_SEQ_LEN = 512
TOKEN_EMBED_DIM = 128

class XTRModel(Enum):
    BASE_EN = 1
    BASE_EN_TPU = 2
    BASE_MULTILINGUAL = 3
    BASE_MULTILINGUAL_TPU = 4
    XXL_EN = 5
    XXL_EN_TPU = 6
    XXL_MULTILINGUAL = 7
    XXL_MULTILINGUAL_TPU = 8

class XTRIndexType(Enum):
    SCANN = 1
    FAISS = 2
    BRUTE_FORCE = 3

class XTRIndexConfig(ABC):
    def __init__(self, type_: XTRIndexType):
        self.type_ = type_

    @property
    def index_type(self):
        return self.type_

class XTRScaNNIndexConfig(XTRIndexConfig):
    def __init__(self):
        super().__init__(XTRIndexType.SCANN)

class XTRFAISSIndexConfig(XTRIndexConfig):
    def __init__(self):
        super().__init__(XTRIndexType.FAISS)

class XTRBruteForceIndexConfig(XTRIndexConfig):
    def __init__(self):
        super().__init__(XTRIndexType.BRUTE_FORCE)

@dataclass
class XTRConfig:
    model: XTRModel
    index_config: XTRIndexConfig

    max_seq_len: int = MAX_SEQ_LEN
    token_embed_dim: int = TOKEN_EMBED_DIM

    @property
    def index_type(self):
        return self.index_config.index_type
