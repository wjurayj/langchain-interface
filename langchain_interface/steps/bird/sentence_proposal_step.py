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
class BIRDSentenceProposalResponse(LLMResponse):
    sentences: List[Text]
    
    
class BIRDSentenceProposalOutputParser(BaseOutputParser[BIRDSentenceProposalResponse]):
    
    @overrides
    def parse(self, text: Text) -> BIRDSentenceProposalResponse:
        lines = text.split("\n")
        # sentences = [line.strip() for line in lines if line.strip().startswith("#")]
        # line may start with # or (\d+). , so we need to check for both and strip these markers
        sentences = []
        for line in lines:
            match = re.match(r"(\d+)?\.* (.*)", line.strip(), re.DOTALL)
            if match:
                sentences.append(match.group(2))
            elif line.strip().startswith("#"):
                sentences.append(line.strip().lstrip("# "))
        
        return BIRDSentenceProposalResponse(
            messages=text,
            sentences=sentences
        )
    
    @property
    def _type(self) -> str:
        return "bird-sentence-proposal"


@Step.register("bird-sentence-proposal")
class BIRDSentenceProposalStep(FewShotStep):
    """ Propose related sentences to improve lieklihood on a given condition. """
    
    def __init__(
        self,
        example_selector: Optional[ExampleSelector] = None
    ):
        if example_selector is None:
            example_selector = ConstantExampleSelector()
            examples = [
                {
                    "scenario": "You want to move around with your cell phone when it is being charged.",
                    "hypothesis": (
                        "You can move around more freely with your cell phone "
                        "if it is being chatged with a one-foot cord rather than a six-foot cord."
                    ),
                    "sentences": (
                        (
                            "# The cell phone is being charged with a portable power bank "
                            "located in your pocket, allowing you to move around "
                            "without being tethered to a fixed outlet.\n"
                            "# The user is working in a compact space where longer cords could "
                            "easily snag on furniture or equipment, thus a one-foot cord could "
                            "minimize this risk.\n"
                            "# The phone is needed for tasks that require frequent handling and "
                            "close proximity to the user, making a shorter cord more practical "
                            "to avoid excessive dangling.\n",
                            "# The charging setup includes a small desktop charger that keeps "
                            "the phone elevated and stable, limiting the practicality of a "
                            "longer cord.\n"
                            "# The user is in a busy environment like a kitchen or workshop, "
                            "where shorter cords can reduce the hazard of tripping or catching "
                            "on moving objects."
                        ),
                    )
                }
            ]
            
            for example in examples:
                example_selector.add_example(example)
            
        super().__init__(example_selector=example_selector)
    
    @overrides
    def get_prompt_template(self) -> Runnable:
        """ """
    
        system_prompt = (
            "You are given a scenario and an accompanying hypothesis. Generate "
            "5 sentences covering different conditions that would add objective "
            "information relevant to the hypothesis such that the hypothesis is "
            "more likely to hold true. The information should not definitively "
            "imply the hypothesis. You must follow the below structure to just generate sentences "
            "with no explanations."
        )
        
        example_prompt = ChatPromptTemplate.from_messages([
            ("human", "Scenario: {scenario}\nHypothesis: {hypothesis}"),
            ("ai", "{sentences}")
        ])

        fewshot_prompt_template = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            example_selector=self._example_selector
        )

        return ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                fewshot_prompt_template,
                ("human", "Scenario: {scenario}\nHypothesis: {hypothesis}"),
            ]
        )
    
    @overrides
    def get_output_parser(self) -> Runnable:
        """ """
        return BIRDSentenceProposalOutputParser()