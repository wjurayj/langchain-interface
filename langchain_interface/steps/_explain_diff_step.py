""" This step is designed to explain the difference between two answer sets. """


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


class ExplainDiffOutputParser(BaseOutputParser[LLMResponse]):
    @overrides
    def parse(self, text: Text) -> LLMResponse:
        return LLMResponse(messages=text)
    
    @property
    def _type(self) -> Text:
        return "explain-diff-output-parser"


@Step.register("explain-diff")
class ExplainDiffStep(FewShotStep):
    """ """
    
    def __init__(
        self,
        example_selector: Optional[ExampleSelector] = None
    ):
        
        if example_selector is None:
            example_selector = ConstantExampleSelector()
            examples = [
                {
                    "group_a": (
                        "- Jørn Utzon\n"
                        "- Louis Kahn"
                    ),
                    "group_b": (
                        "- Ludwig Mies van der Rohe\n"
                        "- Le Corbusier"
                    ),
                    "explanation": (
                        "**Group A: Jørn Utzon and Louis Kahn**\n"
                        "*Time Period*: Mid-20th century\n"
                        "*Contributions*: Organic, monumental architecture blending nature and culture.\n"
                        "*Philosophy*: Both emphasized humanistic architecture, integrating natural forms and materiality. "
                        "Utzon’s Sydney Opera House (1957-73) embodies a sculptural harmony with its harbor setting, while Kahn’s work, "
                        "like the Salk Institute (1959-65), integrates light, texture, "
                        "and monumental solidity to reflect human function and spirituality."
                        "*Distinct Features*: Kahn’s emphasis on monumental geometry and philosophical depth contrasts with Utzon’s sculptural, natural forms. "
                        "Both focused on creating spaces that resonate with human experience rather than industrial efficiency.\n\n"
                        "**Group B: Ludwig Mies van der Rohe and Le Corbusier**\n"
                        "*Time Period*: Early to mid-20th century\n"
                        "*Contributions*: Pioneers of modernist, functional architecture focused on minimalism and new materials.\n"
                        "*Philosophy*: They sought to distill architecture to its essentials, using glass, steel, and concrete to shape urban living. "
                        "Mies’s \"less is more\" mantra manifested in clean lines and open spaces "
                        "(e.g., Seagram Building, 1958), while Le Corbusier’s urban planning ideals (e.g., Villa Savoye, 1931) focused on functionality and modern living standards.\n"
                        "*Distinct Features*: Le Corbusier’s urbanism (e.g., \"Radiant City\") and Mies’s "
                        "minimalism set them apart, with greater focus on industrial advancement and rational spaces.\n\n"
                        "**Key Differentiation**: Group A focused on human-centered monumentalism and organic integration, while Group B drove minimal, "
                        "functional design that defined modernist cityscapes."
                    )
                },
            ]
            
            for example in examples:
                example_selector.add_example(example)

        super().__init__(example_selector=example_selector)
    
    @overrides
    def get_prompt_template(self) -> Runnable:
        """ """

        instruction_prompt = (
            # "Given a question and an answer, shorten the answer to a single word or phrase. "
            # "Please be faithful to the original meaning of the answer, without making any factual judgements. "
            # "Discard any excess information that is not required to answer the question."
            "Given two groups of notable entities, classify and differentiate the groups based on significant features such as time period, "
            "geographic origin, influnce, etc. "
            "Analyze each group creatively, highlighting both broad and niche characteristics that set them apart."
            "Provide clear, concise explanations for how these features lead to a significant separation between the two groups. "
            "Please answer within 200 words. The two groups will be provided below."
        )
        
        example_prompt = ChatPromptTemplate.from_messages([
            ("human", "**Group A**:\n{group_a}\n\n**Group B**:\n{group_b}"),
            ("ai", "{explanation}")
        ])
        
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            example_selector=self._example_selector
        )
        
        prompt_template = ChatPromptTemplate.from_messages([
            ("human", instruction_prompt),
            ("ai", "Please go ahead and provide the two groups of notable entities, and I’ll analyze and classify them for you!"),
            few_shot_prompt,
            ("human", "**Group A**:\n{group_a}\n\n**Group B**:\n{group_b}"),
        ])
        
        return prompt_template
    
    @overrides
    def get_output_parser(self) -> BaseOutputParser:
        """ """
        return ExplainDiffOutputParser()