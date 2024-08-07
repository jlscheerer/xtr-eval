import abc
from typing import Literal, Optional, Union

from xtr.data.collection import Collection
from xtr.data.qas import Qas
from xtr.data.qrels import Qrels
from xtr.data.queries import Queries

Datasplit = Union[Literal["dev"], Literal["test"]]

class Dataset(abc.ABC):
    def __init__(self):
        super().__init__()
        self._collection: Optional[Collection] = None
        self._queries: Optional[Queries] = None
        self._expected: Optional[Union[Qrels, Qas]] = None

    @property
    def name(self):
        return self._name()

    @property
    def collection(self):
        if self._collection is None:
            self._collection, self._queries, self._expected = self._load()
        assert self._collection is not None
        return self._collection

    @property
    def queries(self):
        if self._queries is None:
            self._collection, self._queries, self._expected = self._load()
        assert self._queries is not None
        return self._queries

    def eval(self, rankings):
        if self._expected is None:
            self._collection, self._queries, self._expected = self._load()
        assert self._expected is not None
        self._eval(self._expected, rankings)

    @abc.abstractclassmethod
    def _name(self):
        ...

    @abc.abstractclassmethod
    def _load(self):
        ...

    @abc.abstractclassmethod
    def _eval(self, expected, rankings):
        ...