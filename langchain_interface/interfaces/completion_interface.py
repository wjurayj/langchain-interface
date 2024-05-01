"""We also implement a completion interface that uses vLLM
generation as backbone.
"""

from typing import Union, Text, List, Dict, Optional, Callable, Any
from logging import getLogger
from registrable import Lazy
from overrides import overrides
from dataclasses import asdict
from langchain_community.llms.vllm import VLLMOpenAI
from langchain_core.runnables.config import RunnableConfig
from langchain_core.prompts import (
    PromptTemplate,
    FewShotPromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser
from ..instances.instance import LLMQueryInstance
from .interface import Interface
from ..example_selectors.example_selector import ExampleSelector


logger = getLogger(__name__)


@Interface.register("completion-interface")
class CompletionInterface(Interface):
    def __init__(
        self,
        model_name: Text,
        batch_size: int,
        instruction_prompt: Text,
        max_tokens: Optional[int] = -1,
        input_example_prompt: Optional[Text] = None,
        output_example_prompt: Optional[Text] = None,
        example_selector: Optional[ExampleSelector] = None,
        input_parser: Optional[
            Callable[[LLMQueryInstance], Dict[Text, Text]]
        ] = lambda x: {
            k: str(v) for k, v in asdict(x).items() if not k.endswith("hash")
        },
        output_parser: Optional[Callable[[Text], Dict[Text, Any]]] = lambda x: float(
            x.strip()
        ),
        temperature: float = 0,
        top_k: int = -1,
        top_p: float = 1,
        stop: Optional[List[Text]] = [],
        base_url: Optional[Text] = None,
        api_key: Optional[Text] = None,
        model_kwargs: Dict[Text, Any] = {},
        max_concurrency: int = 4,
    ):
        """
        key difference is that instruction_prompt is Text, not List[Text]
        """

        self.model_name = model_name
        self.batch_size = batch_size
        self.model_kwargs = model_kwargs

        self.input_parser = input_parser
        self.max_tokens = max_tokens

        self.temperature = temperature
        self.top_k = top_k

        if self.top_k != -1:
            logger.warning(
                "top_k is not supported in VLLM OpenAI API of LangChain. Ignoring top_k."
            )
        self.top_p = top_p

        self.stop = stop
        self.base_url = base_url

        self.additional_kwargs = {}

        if api_key is not None:
            self.additional_kwargs["api_key"] = api_key

        # self.llm = VLLM(
        #     temperature=self.temperature,
        #     top_k=self.top_k,
        #     top_p=self.top_p,
        #     model=model_name,
        #     batch_size=batch_size,
        #     vllm_kwargs=self.vllm_kwargs,
        #     max_new_tokens=max_tokens,
        #     stop=stop
        # )
        self.llm = VLLMOpenAI(
            temperature=self.temperature,
            top_p=self.top_p,
            # top_k=self.top_k,  # For some reason top_k is already supplied in the model_kwargs
            model_name=model_name,
            batch_size=batch_size,
            max_tokens=self.max_tokens,
            base_url=self.base_url,
            model_kwargs=self.model_kwargs,
            **self.additional_kwargs
        )

        runnable_config = RunnableConfig(
            max_concurrency=max_concurrency,
        )

        self.instruction_prompt = instruction_prompt
        self.input_example_prompt = input_example_prompt
        self.output_example_prompt = output_example_prompt

        self.example_selector = example_selector
        self.example_prompt = (
            PromptTemplate.from_template(
                template="\n".join(
                    self.input_example_prompt, self.output_example_prompt
                ),
            )
            if output_example_prompt is not None
            else None
        )

        if self.example_prompt is not None:
            assert (
                self.example_selector is not None
            ), "Example selector must be provided if example prompt is provided."

        self.output_parser = output_parser
        llm_chain = self.create_chain()
        
        super().__init__(llm_chain=llm_chain, runnable_config=runnable_config)

    def create_chain(self):
        """Create a callable chain for the model."""

        if self.example_prompt is None:
            prompt_template = PromptTemplate.from_template(
                self.instruction_prompt + "\n" + self.input_example_prompt,
            )
        else:
            prompt_template = FewShotPromptTemplate(
                example_prompt=self.example_prompt,
                example_selector=self.example_selector,
                prefix=self.instruction_prompt,
                suffix=self.input_example_prompt,
            )

        builtin_parser = StrOutputParser()

        # pprint.pp((prompt_template.invoke({"input": "[pseudo input]", "output": "- [pseudo output]"})))
        return prompt_template | self.llm | builtin_parser