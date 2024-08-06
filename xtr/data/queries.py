import abc
from typing import Union, List, Dict

class Queries(abc.ABC):
    def __init__(self):
        super().__init__()

    @abc.abstractclassmethod
    def __iter__(self):
        ...

    @staticmethod
    def cast(queries):
        if isinstance(queries, str):
            return BasicQueries([queries])
        if isinstance(queries, List) or isinstance(queries, Dict):
            return BasicQueries(queries)
        return queries

class BasicQueries(Queries):
    def __init__(self, queries: Union[Dict[str, str], List[str]]):
        super().__init__()
        if isinstance(queries, List):
            self.queries = {idx: query for idx, query in enumerate(queries)}
        else: self.queries = queries

    def __iter__(self):
        return self.queries.items().__iter__()