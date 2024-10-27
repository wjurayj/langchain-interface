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
class BIRDSummarizeToFactorResponse(LLMResponse):
    factor_dict: Dict[Text, List[Text]]
    
    
class BIRDSummarizeToFactorOutputParser(BaseOutputParser[BIRDSummarizeToFactorResponse]):
    # TODO: consider using the JSONParser native to LangChain?
    @overrides
    def parse(self, text: Text) -> BIRDSummarizeToFactorResponse:

        json_dict = json.loads(text.strip())
        
        return BIRDSummarizeToFactorResponse(
            messages=text,
            factor_dict=json_dict
        )
    
    @property
    def _type(self) -> str:
        return "bird-summarize-to-factor"

        
@Step.register("bird-summarize-to-factor")
class BIRDSummarizeToFactorStep(FewShotStep):
    """ """
    
    def __init__(self, example_selector: Optional[ExampleSelector] = None):
        if example_selector is None:
            example_selector = ConstantExampleSelector()
            examples = [
                {
                    "scenario": "You are charging your cell phone and wish to move around with your cell phone.",
                    "description": (
                        "Outcome 1: You can move around more freely with your cell phone "
                        "if it is being charged with a six feet cord rather than a one feet "
                        "cord.\n"
                        "Sentences:\n"
                        "#1 A longer cord provides more flexibility and allows for a greater "
                        "range of movement while using the cell phone. This is because the "
                        "additional length of the six-foot cord gives the user a larger "
                        "radius of movement, enabling them to comfortably use their phone "
                        "while it is charging without feeling restricted or confined to a "
                        "specific location.\n"
                        "Outcome 2: You can move around more freely with your cell phone "
                        "if it is being charged with a one-foot cord rather than a six-foot "
                        "cord.\n"
                        "Sentences: \n"
                        "#2 If the cell phone is plugged into a portable power bank or a "
                        "USB port on a computer, a one-foot cord provides greater mobility "
                        "because it is shorter and less likely to get tangled or caught on "
                        "objects while moving.\n"
                        "#3 If the cell phone is constantly being used while charging and "
                        "the user prefers to keep the phone close to the charger at all "
                        "times, a one-foot cord allows for easier mobility and reduces the "
                        "risk of tripping over a longer cord."
                    ),
                    "factor_dict": {
                        "{\"The cell phone's charging method\": "
                        "[\"The charger is portable\", \"The charger is unmovable\"], "
                        "\"The user's movement range\": "
                        "[\"The user stays very close to the charger\", \"The user has a large "
                        "radius of movement\"], "
                        "\"The location of the phone charger\": "
                        "[\"The user leaves the charger somewhere\", \"The user carries the "
                        "charger\"]}"
                    }
                }
            ]
            
            for example in examples:
                example_selector.add_example(example)
                
        super().__init__(example_selector=example_selector)
        
    @overrides
    def get_prompt_template(self) -> Runnable:
        """ """

        system_prompt = (
            "From the given sentences for each outcome, identify and list "
            "distinct and concrete factors, ensuring each is broad yet specific "
            "and focuses on a unique aspect.\n"
            "Your response should strictly adhere to the JSON format provided, "
            "without additional explanations.\n"
            "For example: {{\"distinct factor\" <ENSURE each factor focuses on a "
            "unique aspect>: \"factor values\" <Each factor MUST cover at least "
            "one condition to support the Statement and one condition to support "
            "the Opposite statement.>}}\n"
            "1. Ensure that each factor's value MUST directly reference "
            "specific elements mentioned in the statements, avoiding vague terms "
            "like 'the object'.\n"
            "2. Ensure the factor values are not too concrete.\n"
            "3. Do not only mention the common situations."
        )
        
        example_prompt = ChatPromptTemplate.from_messages([
            ("human", "Scenario: {scenario}\n{description}"),
            ("ai", "{factor_dict}")
        ])
        
        fewshot_prompt_template = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            example_selector=self._example_selector,
        )
        
        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            fewshot_prompt_template,
            ("human", "Scenario: {scenario}\n{description}")
        ])
    
    @overrides
    def get_output_parser(self) -> Runnable:
        return BIRDSummarizeToFactorOutputParser()