"""
"""
from typing import List, Text, Any, Dict
from overrides import overrides
from .example_selector import ExampleSelector
from langchain_core.example_selectors import BaseExampleSelector


@ExampleSelector.register("static-and-dynamic-example-selector")
class StaticAndDynamicExampleSelector(ExampleSelector):
    """
    """
    def __init__(
        self,
        dynamic_selector: BaseExampleSelector,
        num_static: int
    ):
        """Generate constant example selector.
        """
        super().__init__()
        self.dynamic_selector = dynamic_selector
        self.num_static = num_static
        self._static_pool = []
        
    @overrides
    def add_example(self, example: Dict[Text, Text]) -> Any:
        if len(self._static_pool) < self.num_static:
            self._static_pool.append(example)
        else:
            self.dynamic_selector.add_example(example)
        # self._pool.append(example)
    
    @overrides
    def select_examples(self, input_variables: Dict[Text, Text]) -> List[dict]:
        return self._static_pool + self.dynamic_selector.select_examples(input_variables)