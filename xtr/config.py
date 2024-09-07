import os
import abc
from typing import Optional
from dataclasses import dataclass
from enum import Enum

DEFAULT_INDEX_PATH = "~/data/xtr-eval/indexes"

TOKEN_EMBED_DIM = 128
QUERY_MAXLEN = 32
DOC_MAXLEN = 512
MAX_SEQ_LEN = 512

class XTRModel(Enum):
    BASE_EN = 1
    BASE_EN_TPU = 2
    BASE_MULTILINGUAL = 3
    BASE_MULTILINGUAL_TPU = 4
    XXL_EN = 5
    XXL_EN_TPU = 6
    XXL_MULTILINGUAL = 7
    XXL_MULTILINGUAL_TPU = 8

TPU_MODELS = [XTRModel.BASE_EN_TPU, XTRModel.BASE_MULTILINGUAL_TPU, XTRModel.XXL_EN_TPU, XTRModel.XXL_MULTILINGUAL_TPU]
MULTILINGUAL_MODELS = [XTRModel.BASE_MULTILINGUAL, XTRModel.BASE_MULTILINGUAL_TPU, XTRModel.XXL_MULTILINGUAL, XTRModel.XXL_MULTILINGUAL_TPU]

class XTRIndexType(Enum):
    SCANN = 1
    FAISS = 2
    BRUTE_FORCE = 3

class XTRIndexConfig(abc.ABC):
    def __init__(self, type_: XTRIndexType):
        self.type_ = type_

    @property
    def name(self):
        return self._name()

    @property
    def index_type(self):
        return self.type_

    def _name(self):
        return str(self.type_)

    @abc.abstractmethod
    def is_compatible_with(self, other):
        ...

class XTRScaNNIndexConfig(XTRIndexConfig):
    # Default parameters taken from https://github.com/google-deepmind/xtr/blob/main/xtr_evaluation_on_beir_miracl.ipynb
    def __init__(self, *, num_neighbors: int = 10, max_num_leaves: int = 2000, num_leaves_to_search: int = 100,
                 max_training_sample_size: int = 250000, dimensions_per_block: int = 1, anisotropic_quantization_threshold: float = 0.1,
                 leaves_to_search: int = 100, pre_reorder_num_neighbors: int = 100):
        super().__init__(XTRIndexType.SCANN)
        self.num_neighbors = num_neighbors
        self.max_num_leaves = max_num_leaves
        self.num_leaves_to_search = num_leaves_to_search
        self.max_training_sample_size = max_training_sample_size
        self.dimensions_per_block = dimensions_per_block
        self.anisotropic_quantization_threshold = anisotropic_quantization_threshold

        # Parameters for `XTR.retrieve_docs`
        self.leaves_to_search = leaves_to_search
        self.pre_reorder_num_neighbors = pre_reorder_num_neighbors

    def is_compatible_with(self, other):
        if not isinstance(other, XTRScaNNIndexConfig):
            return False
        if self.num_neighbors != other.num_neighbors:
            return False
        if self.max_num_leaves != other.max_num_leaves:
            return False
        if self.num_leaves_to_search != other.num_leaves_to_search:
            return False
        if self.max_training_sample_size != other.max_training_sample_size:
            return False
        if self.dimensions_per_block != other.dimensions_per_block:
            return False
        if self.anisotropic_quantization_threshold != other.anisotropic_quantization_threshold:
            return False
        return True

class XTRFAISSIndexConfig(XTRIndexConfig):
    # Default parameters taken from https://github.com/google-deepmind/xtr/blob/main/xtr_evaluation_on_beir_miracl.ipynb
    def __init__(self, *, num_clusters: int = 50, code_size: int = 64, nbits_per_idx: int = 4,
                 opq_matrix_niter: int = 10):
        super().__init__(XTRIndexType.FAISS)
        self.num_clusters = num_clusters
        self.code_size = code_size
        self.nbits_per_idx = nbits_per_idx
        self.opq_matrix_niter = opq_matrix_niter

    def is_compatible_with(self, other):
        if not isinstance(other, XTRFAISSIndexConfig):
            return False
        if self.num_clusters != other.num_clusters:
            return False
        if self.code_size != other.code_size:
            return False
        if self.nbits_per_idx != other.nbits_per_idx:
            return False
        if self.opq_matrix_niter != other.opq_matrix_niter:
            return False
        return True

class XTRBruteForceIndexConfig(XTRIndexConfig):
    def __init__(self):
        super().__init__(XTRIndexType.BRUTE_FORCE)

    def is_compatible_with(self, other):
        return isinstance(other, XTRBruteForceIndexConfig)

@dataclass
class XTRConfig:
    index_name: str
    model: XTRModel
    index_config: XTRIndexConfig

    index_path: str = DEFAULT_INDEX_PATH

    token_embed_dim: int = TOKEN_EMBED_DIM
    query_maxlen: int = QUERY_MAXLEN
    doc_maxlen: int = DOC_MAXLEN
    max_seq_len: int = MAX_SEQ_LEN

    # Default parameters taken from https://github.com/google-deepmind/xtr/blob/main/xtr_evaluation_on_beir_miracl.ipynb
    token_top_k: int = 100
    document_top_k: int = 40_000

    build_batch_size: int = 32
    override: bool = False

    max_num_tokens: Optional[int] = None

    @property
    def index_type(self):
        return self.index_config.index_type

    @property
    def path(self):
        return os.path.expanduser(os.path.join(self.index_path, self.index_name))

    def is_multilingual(self):
        return self.model in MULTILINGUAL_MODELS

    def is_tpu(self):
        return self.model in TPU_MODELS

    def is_compatible_with(self, other):
        if self.index_name != other.index_name:
            return False
        if self.model != other.model:
            return False
        if not self.index_config.is_compatible_with(other.index_config):
            return False

        if self.token_embed_dim != other.token_embed_dim:
            return False
        if self.query_maxlen != other.query_maxlen:
            return False
        if self.doc_maxlen != other.doc_maxlen:
            return False
        if self.max_seq_len != other.max_seq_len:
            return False

        if self.token_top_k != other.token_top_k:
            return False
        if self.document_top_k != other.document_top_k:
            return False
        return True