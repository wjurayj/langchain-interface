"""Define the dataloading instance that can be used.
"""
from typing import Text, List, TypeVar, Union, Tuple, Dict, Text, Any, Optional
import abc
from dataclasses import dataclass


@dataclass(frozen=True, eq=True)
class Instance(abc.ABC):
    """
    """
    def __iter__(self):
        for field in self.__dataclass_fields__.keys():
            yield (field, getattr(self, field))


@dataclass(frozen=True, eq=True)
class LLMQueryInstance(Instance):
    """All custom LLMQueryInstance should inherit
    from this very basic instance, so that it is compatible
    with the given scorer.
    """
    id: int
    input: Text
    output: Optional[Union[int, float, Text]] = None

    def __hash__(self) -> int:
        pass