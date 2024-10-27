from typing import Annotated, List, Union, TypeVar
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from ..instances.instance import Instance


_T = TypeVar('_T')


def append(item_list: Union[List[_T], List[List[_T]]], value: Union[_T, List[_T]]) -> List[_T]:
    if not isinstance(value, list):
        item_list.append(value)
        
    else:
        item_list.extend(value)
            
    return item_list


def revise(item_list: _T, value: _T) -> _T:
    return value


def keyupdate(item_dict: dict, value: dict) -> dict:
    return {**item_dict, **value}


class BaseState(TypedDict):
    responses: Annotated[list, append]


class WithInputState(BaseState):
    inputs: Annotated[list, append]
    
class WithTagState(BaseState):
    tags: Annotated[list, append]