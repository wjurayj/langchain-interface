"""
"""
from typing import List, Text, Any, Dict, Optional
from rank_bm25 import BM25Okapi
from overrides import overrides
from .example_selector import ExampleSelector


@ExampleSelector.register("bm25-example-selector")
class BM25ExampleSelector(ExampleSelector):
    """
    """
    def __init__(
        self,
        num_retrieve: int
    ):
        super().__init__()
        self.num_retrieve = num_retrieve
        self._pool = []
        self._pool_updated = False
        self.database: Optional[BM25Okapi] = None # lazy load database

    @overrides
    def add_example(self, example: Dict[Text, Text]) -> Any:
        self._pool.append(example)
        self._pool_updated = True
    
    @overrides
    def select_examples(self, input_variables: Dict[Text, Text]) -> List[dict]:
        if self._pool_updated or self.database is None:
            self.database = BM25Okapi([item['input'].split(" ") for item in self._pool])
            self.fact_map = {item['input']: idx for idx, item in enumerate(self._pool)}
            self._pool_updated = False

        tokenized_query = input_variables['input'].split(" ")
        top_matchings = self.database.get_top_n(tokenized_query, [item['input'] for item in self._pool], self.num_retrieve)

        return [self._pool[self.fact_map[str(match)]] for match in top_matchings]