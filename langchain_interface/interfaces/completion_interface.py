"""We also implement a completion interface that uses vLLM
generation as backbone.
"""
from typing import (
    Union, Text, List, Dict, Optional,
    Callable, Any
)
from registrable import Lazy
from overrides import overrides
from dataclasses import asdict
from langchain_community.llms.vllm import VLLM
from langchain_core.prompts import (
    PromptTemplate,
    FewShotPromptTemplate,
)
import pprint
from langchain_core.output_parsers import StrOutputParser
from ..instances.instance import LLMQueryInstance
from .interface import Interface
from ..example_selectors.example_selector import ExampleSelector


@Interface.register("completion-interface")
class CompletionInterface(Interface):
    def __init__(
        self,
        model_name: Text,
        batch_size: int,
        max_tokens: int,
        input_variables: List[Text],
        inst_prompt: Text,
        question_prompt: Optional[Text] = None,
        example_selector: Optional[ExampleSelector] = None,
        example_prompt_construction: Optional[Dict[Text, Any]] = None,
        input_parser: Optional[Callable[[LLMQueryInstance], Dict[Text, Text]]] = lambda x: {k: str(v) for k, v in asdict(x).items() if not k.endswith("hash")},
        output_parser: Optional[Callable[[Text], Any]] = lambda x: float(x.strip()),
        temperature: float = 0,
        top_k: int = -1,
        top_p: float = 1,
        stop: Optional[List[Text]] = ["\n\n", "Claim:"],
        vllm_kwargs: Dict[Text, Any] = {}
    ):
        super().__init__()

        self.model_name = model_name
        self.batch_size = batch_size
        self.vllm_kwargs = vllm_kwargs
        
        self.input_variables = input_variables
        self.input_parser = input_parser
        self.max_tokens = max_tokens

        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p

        self.stop = stop

        self.llm = VLLM(
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            model=model_name,
            batch_size=batch_size,
            vllm_kwargs=self.vllm_kwargs,
            max_new_tokens=max_tokens,
            stop=stop
        )

        self.inst_prompt = inst_prompt
        self.question_prompt = question_prompt

        self.example_selector = example_selector
        self.example_prompt = PromptTemplate(
            **example_prompt_construction
        ) if example_prompt_construction is not None else None
        
        self.output_parser = output_parser
        
        if self.example_selector is None:
            assert self.output_example_prompt is None, "If example_selector is None, example_prompt should be None as well."
        
        self.llm_chain = self.create_chain()
            
    def create_chain(self):
        """Create a callable chain for the model.
        """
        
        if self.example_selector is None:
            prompt_template = PromptTemplate.from_template(
                self.inst_prompt + '\n' + self.question_prompt,
                input_variables=self.input_variables
            )
        else:
            prompt_template = FewShotPromptTemplate(
                example_prompt=self.example_prompt,
                example_selector=self.example_selector,
                prefix=self.inst_prompt,
                suffix=self.question_prompt,
                input_variables=self.input_variables,
            )

        builtin_parser = StrOutputParser()

        # pprint.pp((prompt_template.invoke({"input": "[pseudo input]", "output": "- [pseudo output]"})))
        return (prompt_template | self.llm | builtin_parser)
        

    @overrides
    def __call__(self, instances: List[LLMQueryInstance]) -> List[Any]:
        """
        """
        instances = [self.input_parser(ins) for ins in instances]
        
        return [self.output_parser(item) for item in self.llm_chain.batch(instances, stop=self.stop)]