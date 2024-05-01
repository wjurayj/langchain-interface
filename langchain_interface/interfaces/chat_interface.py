"""Prompt-based scorer for the model.
"""

from typing import Union, Text, List, Dict, Optional, Callable, Any
from dataclasses import asdict
from overrides import overrides
from langchain_openai import ChatOpenAI
from tqdm import tqdm

# from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_core.runnables.config import RunnableConfig
from langchain.prompts import (
    ChatPromptTemplate,
    ChatMessagePromptTemplate,
    FewShotChatMessagePromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser

# from ..example_selectors.example_selector import ExampleSelector
from ..example_selectors.example_selector import ExampleSelector
from .interface import Interface
from ..instances.instance import LLMQueryInstance


@Interface.register("chat-interface")
class ChatInterface(Interface):
    def __init__(
        self,
        model_name: Text,
        batch_size: int,
        max_tokens: Optional[int] = -1,
        system_message: Optional[Text] = None,
        instruction_prompt: Optional[List[Text]] = None,
        example_selector: Optional[ExampleSelector] = None,
        input_example_prompt: Optional[Text] = None,
        output_example_prompt: Optional[Text] = None,
        input_parser: Optional[
            Callable[[LLMQueryInstance], Dict[Text, Text]]
        ] = lambda x: {
            k: str(v) for k, v in asdict(x).items() if not k.endswith("hash")
        },
        output_parser: Optional[Callable[[Text], Any]] = lambda x: float(x.strip()),
        temperature: float = 0,
        top_k: int = -1,
        top_p: float = 1,
        base_url: Optional[Text] = None,
        api_key: Optional[Text] = None,
        model_kwargs: Dict[Text, Any] = {},
        max_concurrency: int = 4,
    ):
        """Instruction prompt will always begin with the
        human prompt message, and alternate until the end.
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.base_url = base_url
        # self.llm = ChatOpenAI(
        #     model_name=model_name,
        #     batch_size=batch_size,
        # )
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p

        self.model_kwargs = model_kwargs
        self._additional_params = {}

        if api_key is not None:
            self._additional_params["api_key"] = api_key

        self.llm = ChatOpenAI(
            temperature=temperature,
            # top_k=top_k,
            # top_p=top_p,
            model=model_name,
            model_kwargs=self.model_kwargs,
            max_tokens=max_tokens,
            verbose=True,
            base_url=self.base_url,
            **self._additional_params
        )
        runnable_config = RunnableConfig(
            max_concurrency=max_concurrency,
        )

        self.system_message = (
            [] if system_message is None else [("system", system_message)]
        )
        if instruction_prompt is not None:
            assert (
                len(instruction_prompt) % 2 == 0
            ), "Instruction prompt should be a list of even length."
            self.instruction_prompt = [
                ("human" if idx % 2 == 0 else "ai", msg)
                for idx, msg in enumerate(instruction_prompt)
            ]
        else:
            self.instruction_prompt = []
        self.input_parser = input_parser

        self.input_example_prompt = input_example_prompt
        self.output_example_prompt = output_example_prompt

        self.example_selector = example_selector
        self.example_prompt = (
            ChatPromptTemplate.from_messages(
                [
                    ("human", self.input_example_prompt),
                    ("ai", self.output_example_prompt),
                ]
            )
            if input_example_prompt is not None
            and self.output_example_prompt is not None
            else None
        )

        self.output_parser = output_parser

        llm_chain = self.create_chain()

        super().__init__(llm_chain=llm_chain, runnable_config=runnable_config)

    def create_chain(self):
        """Create a callable chain for the model."""

        if self.example_selector is not None:
            # example_prompt_template = ChatPromptTemplate.from_messages(
            #     [
            #         ("human", self.input_example_prompt),
            #         ("ai", self.output_example_prompt)
            #     ]
            # )
            assert (
                self.example_prompt is not None
            ), "If example_selector is not None, example_prompt should not be None as well."
            fewshot_prompt_template = FewShotChatMessagePromptTemplate(
                example_prompt=self.example_prompt,
                example_selector=self.example_selector,
            )

            prompt_template = ChatPromptTemplate.from_messages(
                [
                    # ("system", self.system_message),
                    *self.system_message,
                    *self.instruction_prompt,
                    fewshot_prompt_template,
                    ("human", self.input_example_prompt),
                ]
            )

        else:
            prompt_template = ChatPromptTemplate.from_messages(
                [
                    # ("system", self.system_message),
                    *self.system_message,
                    *self.instruction_prompt,
                    ("human", self.input_example_prompt),
                ]
            )

        builtin_parser = StrOutputParser()

        return prompt_template | self.llm | builtin_parser
