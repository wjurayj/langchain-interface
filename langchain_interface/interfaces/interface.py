"""Define general interface for langgraph execution.
"""

import abc
import asyncio
from registrable import Registrable
from typing import (
    Union,
    List,
    Dict,
    Text,
    Any,
    Callable,
    Optional,
    Iterable,
    AsyncGenerator,
    Awaitable,
    Tuple
)
from langchain_core.runnables.config import RunnableConfig
from langchain_core.runnables.base import Runnable
from langchain_core.language_models.base import BaseLanguageModel
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph import Graph
from tqdm import tqdm
from ..states.base_states import BaseState
from ..instances.instance import Instance, LLMResponse


class Interface(Registrable, abc.ABC):
    
    def __init__(self):
        super().__init__()
    
    @abc.abstractmethod
    def get_runnable(
        self,
        llm: BaseLanguageModel,
    ) -> Runnable:
        # Important: We return a graph that should be compiled by the end user
        raise NotImplementedError