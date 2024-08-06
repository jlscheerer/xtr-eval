import abc
from typing import List

class Queries(abc.ABC):
    def __init__(self):
        super().__init__()

    @abc.abstractclassmethod
    def items(self):
        ...

    @staticmethod
    def cast(queries):
        if isinstance(queries, str):
            return BasicQueries([queries])
        if isinstance(queries, List):
            return BasicQueries(queries)
        return queries

class BasicQueries(Queries):
    def __init__(self, queries: List[str]):
        super().__init__()
        self.queries = queries

    def items(self):
        return self.queries