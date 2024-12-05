"""Define the dataloading instance that can be used.
"""

from typing import (
    Text,
    Mapping,
    Sequence,
    List,
    TypeVar,
    Union,
    Tuple,
    Dict,
    Text,
    Any,
    Optional,
)
import abc
from dataclasses import dataclass


def _to_dict(element) -> Union[Dict[Text, Any], List[Dict[Text, Any]]]:
    if isinstance(element, Mapping):
        return {k: _to_dict(v) for k, v in element.items()}

    elif isinstance(element, Sequence) and not isinstance(element, str):
        return [_to_dict(e) for e in element]

    elif isinstance(element, Instance):
        return element.to_dict()

    else:
        return element


@dataclass(frozen=True, eq=True)
class Instance(abc.ABC):
    """ """

    def __iter__(self):
        for field in self.__dataclass_fields__.keys():
            yield (field, getattr(self, field))

    def to_dict(self) -> Dict[Text, Any]:
        """ """
        return {k: _to_dict(v) for k, v in self}


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
