""" """

try:
    import ujson as json
except ImportError:
    import json
import ast
from dataclasses import dataclass
from overrides import overrides
from typing import Union, Text, List, Dict, Optional, Callable, Any, Tuple

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
class BIRDVerbalizedProbabilityResponse(LLMResponse):
    verbalized_probability: Tuple[float, float]

   
class BIRDVerbalizedProbabilityOutputParser(BaseOutputParser[BIRDVerbalizedProbabilityResponse]):
    
    __CONVERSION_MAP__ = {
        "very likely": 1.,
        "likely": 0.8,
        "somewhat likely": 0.6,
        "neutral": 0.5,
        "somewhat unlikely": 0.4,
        "unlikely": 0.2,
        "very unlikely": 0.
    }
    
    @overrides
    def parse(self, text: Text) -> BIRDVerbalizedProbabilityResponse:
        answer_span = re.search(r"```(.*?)```", text, re.DOTALL)
        options = answer_span.group(1).strip().split("\n")
        
        assert len(options) == 2, f"Expected 2 options, got {len(options)}"
        
        results = []

        for idx, option in enumerate(options):
            verbalized_uncertainty = re.match(option, r"(Outcome \d+): (.*)", re.DOTALL)
            if verbalized_uncertainty:
                if verbalized_uncertainty.group(1) == "Outcome 1":
                    results.append((self.__CONVERSION_MAP__[verbalized_uncertainty.group(2)], 1 - self.__CONVERSION_MAP__[verbalized_uncertainty.group(2)]))
                elif verbalized_uncertainty.group(1) == "Outcome 2":
                    results.append((1 - self.__CONVERSION_MAP__[verbalized_uncertainty.group(2)], self.__CONVERSION_MAP__[verbalized_uncertainty.group(2)]))
                    
        # average the results
        if results:
            return BIRDVerbalizedProbabilityResponse(
                messages=text,
                verbalized_probability=(
                    sum([result[0] for result in results]) / len(results),
                    sum([result[1] for result in results]) / len(results)
                )
            )
    
    @property
    def _type(self) -> str:
        return "bird-verbalized-probability"
    
    
@Step.register("bird-verbalized-probability")
class BIRDVerbalizedProbabilityStep(FewShotStep):

    def __init__(self, example_selector: Optional[ExampleSelector] = None):
        if example_selector is None:
            example_selector = ConstantExampleSelector()
            examples = [
                {
                    "scenario": (
                        "You want to move around with your cell phone when it is "
                        "being charged."
                    ),
                    "condition": (
                        "The charger is portable. The user stays very close to "
                        "the charger. The user carries the charger."
                    ),
                    "outcome_1": (
                        "You can move around more freely with your cell phone "
                        "with a one-foot cord rather than a six-foot cord."
                    ),
                    "outcome_2": (
                        "You can move around more freely with your cell phone "
                        "with a six-foot cord rather than a one-foot cord."
                    ),
                    "reasoning_and_final_response": (
                        "Given that the user is carrying a portable charger, a shorter cord "
                        "like one foot would indeed be far more manageable, making it easier "
                        "for the user to move around freely.\n",
                        "Given the same conditions, a longer cord like six feet might become "
                        "an impediment, making it more challenging for the user who is "
                        "carrying the charger to move around freely due to the possibility "
                        "of tangling or managing the extra length.\n"
                        "```\nOutcome 1: Likely\nOutcome 2: Unlikely\n```"
                    )
                }
            ]
            example_selector.set_examples(examples)
            
        super().__init__(example_selector=example_selector)

    @overrides
    def get_prompt_template(self) -> Runnable:
        """ """
        
        system_prompt = (
            "As an AI assistant, your role is to respond accurately to user "
            "queries. While answering think step-by-step and justify your "
            "answer. Analyze the given scenario and condition to determine the "
            "likelihood of the outcomes. Use only the information provided, "
            "without relying on prior knowledge. Assess the probability using "
            "the specified terms: 'very likely', 'likely', 'somewhat likely',"
            "'neutral', 'somewhat unlikely', 'unlikely', 'very unlikely'. Ensure "
            "that your assessments are complementary: if one outcome is deemed "
            "'likely’, the other must be 'unlikely', and so on. You should first "
            "give your reasons and then format your final answer."
        )

        example_prompt = ChatPromptTemplate.from_messages([
            ("human", "Scenario: {scenario}\nCondition: {condition}\nOutcome 1: {outcome_1}\nOutcome 2: {outcome_2}"),
            ("ai", "{reasoning_and_final_response}")
        ])
        
        fewshot_prompt_template = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            example_selector=self._example_selector
        )
        
        return ChatPromptTemplate.from_messages(
            [
                system_prompt,
                fewshot_prompt_template,
                ("human", "Scenario: {scenario}\nCondition: {condition}\nOutcome 1: {outcome_1}\nOutcome 2: {outcome_2}")
            ]
        )