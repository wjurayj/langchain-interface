"""
"""
from typing import List, Text, Any, Dict
from overrides import overrides
from ..instances.instance import LLMQueryInstance
from .example_selector import ExampleSelector


@ExampleSelector.register("constant-example-selector")
class ConstantExampleSelector(ExampleSelector):
    """
    """
    def __init__(
        self,
    ):
        """Generate constant example selector.
        """
        super().__init__()
        self._pool = []
        
    @overrides
    def add_example(self, example: Dict[Text, Text]) -> Any:
        self._pool.append(example)
    
    @overrides
    def select_examples(self, input_variables: Dict[Text, Text]) -> List[dict]:
        return self._pool