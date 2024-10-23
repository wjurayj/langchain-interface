""" A prompt for genrating quiz questions. """

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
from ..example_selectors import ConstantExampleSelector, ExampleSelector
from .step import Step, FewShotStep
from ..instances.instance import LLMResponse


@dataclass(frozen=True, eq=True)
class QuizQuestionResponse(LLMResponse):
    """Response for decontextualization."""

    question: Text
    answer_template: Text
    place_holder_start: int
    place_holder_end: int


class QuizQuestionOutputParser(BaseOutputParser[QuizQuestionResponse]):
    def parse(self, text: Text) -> QuizQuestionResponse:
        cleaned_text = text.strip()
        
        # find the text wrapped by the code block
        try:
            question = re.search(r"\*\*Question\*\*: (.*?)\n", cleaned_text).group(1).strip()
            answer_template = re.search(
                r"\*\*Answer Template\*\*: (.*)", cleaned_text
            ).group(1).strip()
        except Exception:
            print('-' *20)
            print(f"Error: {cleaned_text}")
            print('-' *20)

        # find the place holder in the answer template
        place_holder_start = answer_template.find("<PLACEHOLDER>")
        place_holder_end = place_holder_start + len("<PLACEHOLDER>")

        return QuizQuestionResponse(
            messages=text,
            question=question,
            answer_template=answer_template,
            place_holder_start=place_holder_start,
            place_holder_end=place_holder_end,
        )

    @property
    def _type(self) -> Text:
        return "quiz-question-output-parser"


@Step.register("quiz-question")
class QuizQuestionStep(FewShotStep):
    """ """
    
    def __init__(
        self,
        example_selector: Optional[ExampleSelector] = None
    ):
        if example_selector is None:
            example_selector = ConstantExampleSelector()
            examples = [
                {
                    "claim": "The Earth is flat.",
                    "question": "What is the shape of the Earth?",
                    "answer_template": "The Earth is <PLACEHOLDER>.",
                },
                {
                    "claim": "Bridget Moynahan is a film actress.",
                    "question": "What is Bridget Moynahan's profession?",
                    "answer_template": "Bridget Moynahan is a/an <PLACEHOLDER>.",
                },
                {
                    "claim": "The capital of France is Paris.",
                    "question": "What is the capital of France?",
                    "answer_template": "The capital of France is <PLACEHOLDER>.",
                },
            ]

            example_selector = ConstantExampleSelector()

            for example in examples:
                example_selector.add_example(example)
            
        super().__init__(example_selector=example_selector)

    @overrides
    def get_prompt_template(self) -> Runnable:

        instruction_prompt = (
            "Please generate a question whose answer is the given claim, focusing on the main atomic fact of the claim. "
            "Also provide a template for the answer that aligns with the claim. Your template should imploy a single placeholder that masks the answer span. "
            "The claims could be about a counterfactual world, so please don't worry about the truthfulness of the claim. "
            "Claims will be procided in the following format: **Claim**: {claim}."
        )

        example_prompt = ChatPromptTemplate.from_messages(
            messages=[
                ("human", "**Claim**: {claim}"),
                (
                    "ai",
                    "**Question**: {question}\n\n**Answer Template**: {answer_template}",
                ),
            ]
        )

        few_shot_prompt_template = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            example_selector=self._example_selector,
        )

        prompt_template = ChatPromptTemplate.from_messages(
            messages=[
                ("human", instruction_prompt),
                (
                    "ai",
                    "Sure! Please provide he claim, and I'll create the question and answer template for you.",
                ),
                few_shot_prompt_template,
                ("human", "**Claim**: {claim}"),
            ]
        )

        return prompt_template

    @overrides
    def get_output_parser(self) -> Runnable:
        return QuizQuestionOutputParser()
