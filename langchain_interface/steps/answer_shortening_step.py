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
class AnswerShorteningResponse(LLMResponse):
    short_answer: Text
    
    
class AnswerShorteningOutputParser(BaseOutputParser[AnswerShorteningResponse]):
    """ """

    @overrides
    def parse(self, text: Text) -> LLMResponse:
        matched = re.match(r"```\n(.*)\n```", text)
        return AnswerShorteningResponse(
            short_answer=matched.group(1).strip(),
            messages=text
        )
        
    @property
    def _type(self) -> Text:
        return "answer-shortening-output-parser"


@Step.register("answer-shortening")
class AnswerShorteningStep(Step):
    """ """

    @overrides
    def get_prompt_template(self) -> Runnable:
        """ """
        example_selector = ConstantExampleSelector()
        examples = [
            {
                "question": "Where is the Eiffel Tower located?",
                "answer": "The Eiffel Tower is located in London.",
                "short": "London",
            },
            {
                "question": "Who designed the Sydney Opera House?",
                "answer": "The Sydney Opera House was designed by Jørn Utzon, a Danish architect.",
                "short": "Jørn Utzon",
            },
            {
                "question": "When was the Declaration of Independence signed?",
                "answer": "The Declaration of Independence was signed in the 1770s.",
                "short": "1770s",
            },
        ]
        
        for example in examples:
            example_selector.add_example(example)

        instruction_prompt = (
            "Given a question and an answer, shorten the answer to a single word or phrase. "
            "Please be faithful to the original meaning of the answer, without making any factual judgements. "
            "Discard any excess information that is not required to answer the question."
        )
        
        example_prompt = ChatPromptTemplate.from_messages([
            ("human", "**Question**: {question}\n**Answer**: {answer}"),
            ("ai", "```\n{short}\n```"),
        ])
        
        few_shot_prompt_template = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            example_selector=example_selector
        )

        prompt_template = ChatPromptTemplate.from_messages([
            ("human", instruction_prompt),
            ("ai", "Sure! Please provide the question and answer you'd like me to shorten."),
            few_shot_prompt_template,
            ("human", "**Question**: {question}\n**Answer**: {answer}"),
        ])
        
        return prompt_template
    
    @overrides
    def get_output_parser(self) -> BaseOutputParser:
        ...
        
        return AnswerShorteningOutputParser()