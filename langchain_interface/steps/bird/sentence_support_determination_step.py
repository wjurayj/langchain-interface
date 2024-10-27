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
class BIRDSentenceSupportDeterminationResponse(LLMResponse):
    support_index: int
    
    
class BIRDSentenceSupportDeterminationOutputParser(BaseOutputParser[BIRDSentenceSupportDeterminationResponse]):
    
    @overrides
    def parse(self, text: Text) -> BIRDSentenceSupportDeterminationResponse:

        try:
            answer_span = re.search(r"```(.*?)```", text, re.DOTALL)
            if answer_span.group(1).strip().lower() == "outcome 1":
                return BIRDSentenceSupportDeterminationResponse(
                    messages=text,
                    support_index=0
                )
            elif answer_span.group(1).strip().lower() == "outcome 2":
                return BIRDSentenceSupportDeterminationResponse(
                    messages=text,
                    support_index=1
                )
            else:
                return BIRDSentenceSupportDeterminationResponse(
                    messages=text,
                    support_index=-1
                )
                
        except Exception:
            return BIRDSentenceSupportDeterminationResponse(
                messages=text,
                support_index=-1
            )
            
    @property
    def _type(self) -> str:
        return "sentence-support-determination"
    
    
@Step.register("sentence-support-determination")
class BIRDSentenceSupportDeterminationStep(FewShotStep):
    """ Determine which sentence is the support sentence supporting. """
    
    def __init__(
        self,
        example_selector: Optional[ExampleSelector] = None
    ):
        if example_selector is None:
            example_selector = ConstantExampleSelector()
            examples = [
                {
                    "scenario": "The government is planing the location for building charging stations,",
                    "outcome_1": "The government should build a chargin station here.",
                    "outcome_2": "The government should not build a charging station here.",
                    "condition": "The location is near a school.",
                    "reasoning": (
                        "The rationale is that a high adoption rate of electric vehicles "
                        "indicates a strong demand for charging infrastructure. Therefore, "
                        "building a charging station would help meet the needs of the "
                        "electric vehicle owners in the area and support further adoption "
                        "of clean energy transportation.\n"
                        "Therefore, the condition provided better supports Outcome 1: The "
                        "government should build a charging station here."
                    ),
                    "result": "Outcome 1"
                },
            ]

            for example in examples:
                example_selector.add_example(example)
                
        super().__init__(example_selector=example_selector)

    @overrides
    def get_prompt_template(self) -> Runnable:
        """ """
        
        system_prompt = (
            "A scenario and two outcomes are provided. "
            "Determin which outcome the condition better supports. "
            "Put your final answer in a code block."
        )
        
        example_prompt = ChatPromptTemplate.from_messages(
            [
                ("human", "Scenario: {scenario}\nOutcome 1: {outcome_1}\nOutcome 2: {outcome_2}\nCondition: {condition}"),
                ("ai", "{reasoning}\n```{result}```")
            ]
        )
        
        fewshot_prompt_template = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            example_selector=self._example_selector,
        )

        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            fewshot_prompt_template,
            ("human", "Scenario: {scenario}\nOutcome 1: {outcome_1}\nOutcome 2: {outcome_2}\nCondition: {condition}")
        ])
        
    @overrides
    def get_output_parser(self) -> Runnable:
        return BIRDSentenceSupportDeterminationOutputParser()