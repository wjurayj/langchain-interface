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
class BIRDReevaluateImplicationResponse(LLMResponse):
    implication_dict: Dict[Text, List[Text]]

    
class BIRDReevaluateImplicationOutputParser(BaseOutputParser[BIRDReevaluateImplicationResponse]):
    
    @overrides
    def parse(self, text: Text) -> BIRDReevaluateImplicationResponse:

        return BIRDReevaluateImplicationResponse(
            messages=text,
            implication_dict=json.loads(text.strip())
        )
    
    @property
    def _type(self) -> str:
        return "bird-reevaluate-implication"
    

@Step.register("bird-reevaluate-implication")
class BIRDReevaluateImplicationStep(FewShotStep):
    """ This step is used to reevaluate the implication of a given sentence with the given context """
    
    def __init__(
        self,
        example_selector: Optional[ExampleSelector] = None
    ):
        if example_selector is None:
            example_selector = ConstantExampleSelector()
            examples = [
                {
                    "scenario": "The government is planning the locations for building charging stations.",
                    "implication_dict": json.dumps({
                        "The location is on a busy highway with no existing charging stations.": [
                            "No nearby charging stations",
                            "Location is on a major travel route, serving long-distance EV travelers",
                            "Nearby amenities like restaurantns, shops, and rest areas for users while charging"
                        ]
                    }),
                    "filtered_dict": json.dumps({
                        "The location is on a busy highway with no existing charging stations.": [
                            "No nearby charging stations",
                            "Location is on a major travel route, serving long-distance EV travelers",
                        ]
                    })
                }
            ]
            
            for example in examples:
                example_selector.add_example(example)
                
        super().__init__(example_selector=example_selector)

    @overrides
    def get_prompt_template(self) -> Runnable:
        """ """

        system_prompt = (
            "You are an AI assistant that verifies your own response. The user "
            "will give you your previous response. Your task is that given your "
            "answer of the implied factors as in the list for the key based on "
            "a scenario, check if the key necessarily implies all the values in "
            "the list. You should output a JSON with no explanation. "
            "Here are the rules that you must follow:\n"
            "1. You should think about the scenario.\n"
            "2. If you think the key implies all the values, keep the value "
            "list, otherwise only include the ones that are implied.\n"
            "3. You are allowed to generate an empty list if you think none of"
            "the values are implied.\n"
            "4. Make sure you check if all the conditions in the value are "
            "implied by the key, if not, remove the value.\n"
            "5. Make sure the remaining values do not conflict with each other."
        )

        example_prompt = ChatPromptTemplate.from_messages([
            ("human", "Scenario: {scenario}\n{implication_dict}"),
            ("ai", "{filtered_dict}")
        ])

        fewshot_prompt_template = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            example_selector=self._example_selector
        )

        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            fewshot_prompt_template,
            ("human", "Scenario: {scenario}\n{implication_dict}"),
        ])
        
    @overrides
    def get_output_parser(self) -> Runnable:
        return BIRDReevaluateImplicationOutputParser()