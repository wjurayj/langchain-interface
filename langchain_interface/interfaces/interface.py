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
    Optional,
    Iterable,
    AsyncGenerator,
    Awaitable,
    Tuple
)
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from tqdm import tqdm
from ..states.base_states import BaseState
from ..instances.instance import LLMQueryInstance, LLMResponse


class Interface(Registrable):
    """
    """
    def __init__(
        self,
        lang_graph: CompiledStateGraph,
        runnable_config: RunnableConfig
    ):
        """
        """
        super().__init__()
        self._graph = lang_graph
        self._runnable_config = runnable_config
        
    def _parse_input(self, instances: List[LLMQueryInstance]) -> List[BaseState]:
        """
        """
        return [BaseState(responses=ins) for ins in instances]
    
    def _parse_output(self, batched_state: BaseState) -> LLMResponse:
        """
        """
        
        return [bs['responses'][-1] for bs in batched_state]
        
    def __call__(
        self, instances: List[LLMQueryInstance],
    ) -> List[LLMResponse]:

        # instances = [self._parse_input(ins) for ins in instances]
        instances = self._parse_input(instances)
        batched_state = self._graph.batch(instances, config=self._runnable_config)
        
        return self._parse_output(batched_state)