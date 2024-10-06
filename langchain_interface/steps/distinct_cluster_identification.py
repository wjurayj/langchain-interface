""" """


from dataclasses import dataclass
from overrides import overrides
from typing import Union, Text, List, Dict, Optional, Callable, Any
import re

from langchain_core.runnables.config import RunnableConfig
from langchain_core.runnables.base import Runnable
from langchain.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)
from langchain_core.output_parsers import BaseOutputParser

# TODO: use customer downloaded examples for example selector
from ..example_selectors import ConstantExampleSelector
from .step import (
    Step
)
from ..instances.instance import LLMResponse


@dataclass(frozen=True, eq=True)
class DistinctClusterIdentificationResponse(LLMResponse):
    clusters: List[Text]
    
    
class DistinctClusterIdentificationOutputParser(BaseOutputParser[DistinctClusterIdentificationResponse]):
    
    @overrides
    def parse(self, text: Text) -> DistinctClusterIdentificationResponse:
        lines = text.split("\n")
        clusters = [line.strip()[1:].strip() for line in lines if line.strip().startswith("-")]
        
        return DistinctClusterIdentificationResponse(
            messages=text,
            clusters=clusters
        )
    
    @property
    def _type(self) -> str:
        return "distinct-cluster-identification"


@Step.register("distinct-cluster-identification")
class DistinctClusterIdentificationStep(Step):
    """ From my experience this only works well with GPT-4o (not even mini) """
    @overrides
    def get_prompt_template(self) -> Runnable:
        example_selector = ConstantExampleSelector()
        examples = [
            {
                "str_list": "\n".join([f"- {item}" for item in [
                    "apple",
                    "Apple",
                    "the apple",
                    "apple",
                    "pear",
                    "a pear",
                    "the pear",
                    "the pear",
                    "the pear",
                    "the pear",
                    "pear"
                ]]),
                "clusters": '\n'.join([
                    "- apple",
                    "- pear"
                ])
            },
            {
                "str_list": "\n".join([f"- {item}" for item in [
                    "NYC",
                    "New York City",
                    "New York",
                    "NY",
                    "Big Apple",
                    "Los Angeles",
                    "LA",
                    "Detroit",
                    "Motor City",
                ]]),
                "clusters": '\n'.join([
                    "- NYC",
                    "- Los Angeles",
                    "- Detroit"
                ])
            }
        ]
        
        for example in examples:
            example_selector.add_example(example)

        input_example_prompt = (
            "Given a list of answers to a question,"
            "identify semantically distinct answers.\n\n"
            "\n\n**List of Answers**:\n\n{str_list}"
        )
        output_example_prompt = (
            "**Semantically Distinct Answers**:\n\n{clusters}"
        )

        example_prompt = ChatPromptTemplate.from_messages([
            ("human", input_example_prompt),
            ("ai", output_example_prompt),
        ])
        
        fewshot_prompt_template = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            example_selector=example_selector,
        )
        
        prompt_template = ChatPromptTemplate.from_messages(
            [
                fewshot_prompt_template,
                ("human", input_example_prompt),
            ]
        )
        
        return prompt_template
        
    @overrides
    def get_output_parser(self) -> Runnable:
        return DistinctClusterIdentificationOutputParser()