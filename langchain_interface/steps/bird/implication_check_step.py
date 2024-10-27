""" """

try:
    import ujson as json
except ImportError:
    import json
import ast
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
from ...example_selectors import ConstantExampleSelector, ExampleSelector
from ..step import (
    Step,
    FewShotStep
)
from ...instances.instance import LLMResponse


@dataclass(frozen=True, eq=True)
class BIRDImplicationCheckResponse(LLMResponse):
    implied: bool

    
class BIRDImplicationCheckOutputParser(BaseOutputParser[BIRDImplicationCheckResponse]):
    @overrides
    def parse(self, text: Text) -> BIRDImplicationCheckResponse:

        try:
            answer_span = re.search(r"```(.*?)```", text, re.DOTALL)
            answer_span.group(1).strip().lower()
            if answer_span == "true":
                return BIRDImplicationCheckResponse(
                    messages=text,
                    implied=True
                )
            elif answer_span == "false":
                return BIRDImplicationCheckResponse(
                    messages=text,
                    implied=False
                )
        except Exception:
            return BIRDImplicationCheckResponse(
                messages=text,
                implied=False
            )
        
        return BIRDImplicationCheckResponse(
            messages=text,
            implied=False
        )
    
    @property
    def _type(self) -> str:
        return "bird-implication-check"
    
    
@Step.register("bird-implication-check")
class BIRDImplicationCheckStep(FewShotStep):
    
    def __init__(
        self,
        example_selector: Optional[ExampleSelector] = None
    ):
        if example_selector is None:
            example_selector = ConstantExampleSelector()
            examples = [
                {
                    "scenario": (
                        "Dave was a scientist. Dave wanted to make a great "
                        "scientific discovery. Dave worked with algae to make electricity. "
                        "Dave discovered he could make electricity with algae! Dave was "
                        "awarded for his great discovery."
                    ),
                    "condition": (
                        "Dave is known to meticulously plan his investigations and "
                        "ensure all necessary resources and funds are obtained "
                        "beforehand."
                    ),
                    "statement": "Dave tends to plan ahead",
                    "reasoning": (
                        "The scenario and condition indicate that Dave meticulously plans "
                        "his investigations and ensures all necessary resources and funds "
                        "are obtained beforehand. This suggests that Dave is proactive and "
                        "plans ahead of time.\n"
                        "So we can conclude that the scenario with the condition implies the "
                        "statement."
                    ),
                    "implied": "true"
                }
            ]
            
            for example in examples:
                example_selector.add_example(example)

        super().__init__(example_selector)
        
    @overrides
    def get_prompt_template(self) -> Runnable:
        
        system_prompt = (
            "Decide if the scenario with the condition implies the statement. "
            "Your final answer should be either 'true' or 'false'. Put your final answer in a code block."
        )

        example_prompt = ChatPromptTemplate.from_messages([
            ("human", "Scenario: {scenario}\nCondition: {condition}\nStatement: {statement}"),
            ("ai", "{reasoning}\n```{implied}```")
        ])
        
        fewshot_prompt_template = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            example_selector=self._example_selector
        )

        return ChatPromptTemplate.from_messages(
            [
                system_prompt,
                fewshot_prompt_template,
                ("human", "Scenario: {scenario}\nCondition: {condition}\nStatement: {statement}"),
            ]
        )
        
    @overrides
    def get_output_parser(self) -> Runnable:
        return BIRDImplicationCheckOutputParser()