"""Define general interface for langgraph execution.
"""

import abc
import asyncio
from registrable import Registrable
from typing import (
    Callable,
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
# from langchain_openai import ChatOpenAI
# from langchain_core.runnables.config import RunnableConfig
from langchain_core.runnables.base import Runnable
from langchain_core.language_models.base import BaseLanguageModel
from ..example_selectors import ExampleSelector
from ..states.base_states import BaseState
from ..instances.instance import Instance, LLMResponse


class Step(Registrable, abc.ABC):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def get_prompt_template(self) -> Runnable:
        """ """
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_output_parser(self) -> Runnable:
        """ """
        raise NotImplementedError   
    
    def chain_llm(self, llm: BaseLanguageModel) -> Runnable:
        """ Provided an LLM, chain it with the current step's prompt template and output parser. """
        return self.get_prompt_template() | llm | self.get_output_parser()
    
    def induce_stated_callable(
        self, 
        llm: BaseLanguageModel,
        parse_input: Callable[[BaseState], Dict[Text, Any]],
        parse_output: Union[Callable[[LLMResponse], Dict[Text, Any]], Callable[[LLMResponse, BaseState], Dict[Text, Any]]]
    ) -> Callable[[BaseState], BaseState]:
        """ """
        
        chained_runnable = self.chain_llm(llm)

        # TODO: whether we want to passdown type hints
        def _callable(state):
            """ """
            inputs = parse_input(state)
            output = chained_runnable.invoke(inputs)
            
            # check whether parse_output takes a single argument or two
            # if takes two, pass the state as the second argument
            return parse_output(output) if len(parse_output.__code__.co_varnames) == 1 else parse_output(output, state)
        
        return _callable
    
    
class FewShotStep(Step):
    def __init__(
        self,
        example_selector: Optional[ExampleSelector] = None
    ):
        super().__init__()
        self._example_selector = example_selector