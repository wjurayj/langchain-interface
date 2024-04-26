"""Create scorers that gives factuality judgment scores to a given claim and evidence pair.
"""
import abc
from registrable import Registrable
from typing import Union, List
from ..instances.instance import LLMQueryInstance


class Interface(Registrable):
    """
    """
    def __init__(self):
        super().__init__()
        
    @abc.abstractmethod
    async def __call__(self, instances: List[LLMQueryInstance]) -> List[Dict[Text, Any]]:
        raise NotImplementedError
