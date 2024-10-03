""" """

from dataclasses import dataclass
from overrides import overrides
from typing import Union, Text, List, Dict, Optional, Callable, Any

from langchain_core.runnables.config import RunnableConfig
from langchain_core.runnables.base import Runnable
from langchain.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)
from langchain_core.output_parsers import BaseOutputParser
import re

# TODO: use customer downloaded examples for example selector
from ..example_selectors import ConstantExampleSelector
from .step import Step
from ..instances.instance import LLMResponse


@dataclass(frozen=True, eq=True)
class TestOnQuizResponse(LLMResponse):
    """ """
    block: Text
    infill: Text


class TestOnQuizOutputParser(BaseOutputParser[TestOnQuizResponse]):
    """ """
    @overrides
    def parse(self, text: Text) -> TestOnQuizResponse:
        
        code_block = re.search(r"```(.*?)```", text, re.DOTALL).group(1).strip()
        infill = re.match(f"PLACEHOLDER = (.*)", code_block, re.DOTALL).group(1).strip().strip("\"'")
        
        return TestOnQuizResponse(
            block=code_block,
            infill=infill,
            messages=text,
        )


@Step.register("test-on-quiz")
class TestOnQuizStep(Step):
    """ """

    @overrides
    def get_prompt_template(self) -> Runnable:
        
        example_selector = ConstantExampleSelector()
        examples = [
            {
                "question": "What is the shape of the earth?",
                "answer_template": "The Earth is <PLACEHOLDER>.",
                "answer": "round",
            },
            {
                "question": "What is Bridget Moynahan's profession?",
                "answer_template": "Bridget Moynahan is a/an <PLACEHOLDER>.",
                "answer": "actress",
            },
            {
                "question": "What is the capital of France?",
                "answer_template": "The capital of France is <PLACEHOLDER>.",
                "answer": "Paris",
            }
        ]
        
        for example in examples:
            example_selector.add_example(
                example
            )

        instruction_prompt = (
            "Please answer the given questions by filling in the blanks in the provided answer templates. Your response should be surrounded by triple backticks. "
            "For example, if the answer template is 'The Earth is <PLACEHOLDER>.' and the answer is 'round', your response should be:\n\n```\nPLACEHOLDER = \"round\"\n```'."
        )
        
        example_prompt = ChatPromptTemplate.from_messages(
            messages=[
                ("human", "**Question**: {question}\n\n**Answer Template**: {answer_template}"),
                ("ai", "```\nPLACEHOLDER = \"{answer}\"\n```"),
            ]
        )

        few_shot_prompt_template = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            example_selector=example_selector,
        )

        prompt_template = ChatPromptTemplate.from_messages(
            messages=[
                ("human", instruction_prompt),
                ("ai", "Sure! Please provide the questions and the templates for the answers."),
                few_shot_prompt_template,
                ("human", "**Question**: {question}\n\n**Answer Template**: {answer_template}"),
            ]
        )

        return prompt_template

    @overrides
    def get_output_parser(self) -> Runnable:
        
        return TestOnQuizOutputParser()