"""Define the dataloading instance that can be used.
"""
from typing import (
    Text,
    List,
    TypeVar,
    Union,
    Tuple,
    Dict,
    Text,
    Any,
    Optional
)
import abc
from dataclasses import dataclass


@dataclass(frozen=True, eq=True)
class Instance(abc.ABC):
    """
    """
    def __iter__(self):
        for field in self.__dataclass_fields__.keys():
            yield (field, getattr(self, field))
            
    def to_dict(self) -> Dict[Text, Any]:
        """
        """
        return {k: v for k, v in self}


@dataclass(frozen=True, eq=True)
class LLMResponse(Instance):
    """All custom LLMQueryInstance should inherit
    from this very basic instance, so that it is compatible
    with the given scorer.
    """
    messages: Text

    def __hash__(self) -> int:
        return hash(self.id) + hash(self.raw)
    
    def __str__(self) -> Text:
        return self.messages
    
    def __dict__(self) -> Dict[Text, Any]:
        return self.to_dict()