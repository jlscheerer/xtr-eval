import abc
from typing import List

from tqdm import tqdm

from nltk.tokenize import sent_tokenize
import re

class Collection(abc.ABC):
    def __init__(self):
        super().__init__()

    @abc.abstractclassmethod
    def __getitem__(self, document_id):
        ...

    @abc.abstractclassmethod
    def __len__(self):
        ...

    @abc.abstractclassmethod
    def enumerate_batches(self, batch_size):
        ...

    @staticmethod
    def cast(collection):
        if isinstance(collection, str):
            return BasicCollection([collection])
        if isinstance(collection, List):
            return BasicCollection(collection)
        return collection

    def index_map(self):
        return None

class BasicCollection(Collection):
    def __init__(self, documents: List[str]):
        super().__init__()
        self.documents = documents

    def __getitem__(self, document_id):
        return self.documents[document_id]

    def __len__(self):
        return len(self.documents)

    def enumerate_batches(self, batch_size):
        for batch_idx in tqdm(range(0, len(self.documents), batch_size)):
            yield batch_idx, self.documents[batch_idx:batch_idx+batch_size]

class MappedCollection(BasicCollection):
    def __init__(self, documents: List[str], keys: List[str]):
        super().__init__(documents=documents)
        self.keys = keys

    def index_map(self):
        return self.keys

class SentenceChunkedCollection(Collection):
    def __init__(self, document):
        self.document = document
        self.basic_collection = None

    def __getitem__(self, document_id):
        self._materialize_collection_if_required()
        return self.basic_collection[document_id]

    def __len__(self):
        self._materialize_collection_if_required()
        return len(self.basic_collection)

    def enumerate_batches(self, batch_size):
        self._materialize_collection_if_required()
        return self.basic_collection.enumerate_batches(batch_size=batch_size)

    def _materialize_collection_if_required(self):
        if self.basic_collection is not None:
            return
        doc = re.sub(r'\[\d+\]', '', self.document)
        # Single-sentence chunks.
        chunks = [chunk.lower() for chunk in sent_tokenize(doc)]
        self.basic_collection = BasicCollection(chunks)

    def __getstate__(self):
        return self.document

    def __setstate__(self, document):
        self.document = document
        self.basic_collection = None