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
from langchain_openai import ChatOpenAI
from langchain_core.runnables.config import RunnableConfig
from ..states.base_states import BaseState
from ..instances.instance import Instance, LLMQueryInstance, LLMResponse


class Step(Registrable):
    """
    """
    def __init__(
        self,
        llm_chain_creator,
        model_name: Text,
        max_tokens: Optional[int] = -1,
        temperature: float = 0,
        top_p: float = 1,
        base_url: Optional[Text] = None,
        api_key: Optional[Text] = None,
        model_kwargs: Dict[Text, Any] = {},
        max_concurrency: int = 4,
    ):
        """
        """
        super().__init__()
        
        self.model_name = model_name
        self.base_url = base_url
        self.temperature = temperature
        self.top_p = top_p

        self.model_kwargs = model_kwargs
        self._additional_params = {}

        if api_key is not None:
            self._additional_params["api_key"] = api_key

        self._llm = ChatOpenAI(
            temperature=temperature,
            top_p=top_p,
            model=model_name,
            model_kwargs=self.model_kwargs,
            max_tokens=max_tokens,
            verbose=True,
            base_url=self.base_url,
            **self._additional_params
        )
        self._runnable_config = RunnableConfig(
            max_concurrency=max_concurrency,
        )

        self._llm_chain = llm_chain_creator()
        
    def __call__(
        self, params: Union[Dict[Text, Any], List[Dict[Text, Any]]]
    ) -> Union[List[LLMResponse], LLMResponse]:
        """ """
        
        return self._call_chain(params)
    
    def _call_chain(
        self,
        params: Union[Dict[Text, Any], List[Dict[Text, Any]]]
    ) -> Union[LLMResponse, List[LLMResponse]]:
        """
        """
        
        if isinstance(params, list):
            return self._llm_chain.batch(params)
        
        return self._llm_chain.invoke(params)