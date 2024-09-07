from typing import Annotated, List, Union
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from ..instances.instance import Instance


def append(item_list: Union[List[Instance], List[List[Instance]]], value: Union[Instance, List[Instance]]) -> List[Instance]:
    
    if isinstance(value, Instance):
        item_list.append(value)
        
    else:
        assert len(item_list) == 0 or len(value) == len(item_list), "Length of the value should be the same as the item_list."
        if len(item_list) == 0:
            item_list = [[] for _ in range(len(value))]
        for idx, item in enumerate(item_list):
            item.append(value[idx])
            
    return item_list


class BaseState(TypedDict):
    responses: Annotated[list, append]