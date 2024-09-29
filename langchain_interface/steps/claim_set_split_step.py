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
class ClaimSetSplitResponse(LLMResponse):
    general_response: Text


class ClaimSetSplitOutputParser(BaseOutputParser):
    """ """
    
    @overrides
    def parse(self, text: Text) -> LLMResponse:
        """ """
        clean_text = text.strip()
        matched = re.search(r"```(.*?)```", clean_text, re.DOTALL)
        return ClaimSetSplitResponse(general_response=matched.group(1).strip(), messages=text)
    
    @property
    def _type(self) -> Text:
        return "claim-set-split-output-parser"


@Step.register("claim-set-split")
class ClaimSetSplitStep(Step):
    """ """
    
    @overrides
    def get_prompt_template(self) -> Runnable:
        """ Generate a more general claim based on the input claims. """
        
        example_selector = ConstantExampleSelector()

        # TODO: change the inputs to include better analysis from prev generation.
        examples = [
            {
                "positive": (
                    "- The Eiffel Tower is in Paris.\n"
                    "- Paris, France\n"
                    "- The Eiffel Tower is in Nice.\n"
                    "- The Eiffel Tower is located in Marseille."
                ),
                "negative": (
                    "- The Eiffel Tower is in London.\n"
                    "- The Eiffel Tower is located in New York.\n"
                    "- The Eiffel Tower is located in Berlin.\n"
                    "- The Eiffel Tower is in Tokyo."
                ),
                "analysis": (
                    "Paris, Nice and Marseille are all cities in France. "
                    "London, New York, Berlin, and Tokyo are all cities in different countries."
                ),
                "response": "The Eiffel Tower is located in France."
            },
            {
                "positive": (
                    "- The Declaration of Independence was signed in 1772.\n"
                    "- The Declaration of Independence was signed in 1775.\n"
                    "- The Declaration of Independence was signed in 1776 A.D.\n"
                    "- The Declaration of Independence was signed in 1777"
                ),
                "negative": (
                    "The Declaration of Independence was signed in 1780.\n"
                    "The Declaration of Independence was signed in 1785.\n"
                    "The Declaration of Independence was signed in 1790.\n"
                    "The Declaration of Independence was signed in 1795.\n"
                    "The Declaration of Independence was signed in 1600."
                ),
                "analysis": (
                    "The positive claims are all within the 1770s, while the negative claims are outside of this range."
                ),
                "response": "The Declaration of Independence was signed in the 1770s."
            },
            {
                "positive": (
                    "- The Sydney Opera House was designed by Jørn Utzon.\n"
                    "- The Sydney Opera House was designed by Louis Kahn.\n"
                    "- The Sydney Opera House was designed by Tadao Ando."
                ),
                "negative": (
                    "- The Sydney Opera House was designed by Ludwig Mies van der Rohe.\n"
                    "- The Sydney Opera House was designed by Le Corbusier.\n"
                ),
                "analysis": (
                    "The human subject maintains this belief based on an association between architectural styles, reputation, "
                    "or familiarity with the architects mentioned. In the Positive Claims, "
                    "either Jørn Utzon or Louis Kahn designed the Sydney Opera House because both are prominent, "
                    "innovative architects, known for their modernist approaches. "
                    "Utzon, particularly, is strongly associated with the structure due to his reputation, "
                    "though the subject could be considering Kahn based on his influence in modern architecture.\n\n"
                    "The Negative Claims feature architects known for their distinct styles, "
                    "but whose aesthetic or body of work may not seem to align with the design of the Sydney Opera House in the subject’s perception. "
                    "For instance, Mies van der Rohe is associated with minimalism and sleek, functional "
                    "buildings, while Le Corbusier is known for his brutalist and rationalist designs, "
                    "which might contrast with the Opera House’s more organic and expressive form.\n\n"
                    "Thus, the commonality among the Positive Claims is that both architects could be perceived as plausible candidates for a modernist landmark, "
                    "while the Negative Claims involve architects whose styles feel incompatible with "
                    "the visual and architectural identity of the Sydney Opera House."
                ),
                "response":"The Sydney Opera House was designed by a 20th-century architect known for modernist, organic, or humanist architectural styles."
            }
        ]
        
        for example in examples:
            example_selector.add_example(example)
            
        instruction_prompt = (
            "Your task is to help an uncertain human subject propose a claim they want to make, "
            "based on a set of **positive claims** they believe could be true, and a set of **negative claims** they are confident should be false. "
            "Your goal is to **propose the claim** in a way that:\n\n"
            "1. **Any of the positive claims should entail the proposed claim** — meaning that the proposed claim must be true if any of the positive claims are true.\n"
            "2. **Any of the negative claims should contradict the proposed claim** — meaning that the proposed claim must be false if any of the negative claims are true.\n\n"
            "In other words, your proposed answer should be consistent with the belief system of the responder,"
            "ensuring that the claim aligns with the positive claims while contradicting the negative claims. "
            "You will be provided some relevant info on what contrasts the positive and negative claims."
            "Do not focus on factual accuracy or correcting potential misunderstandings in the claims. Your job is to achieve internal consistency with the given positive and negative claims."
        )
        
        example_prompt = ChatPromptTemplate.from_messages([
            ("human", "**Positive Claims**:\n{positive}\n\n**Negative Claims**:\n{negative}\n\n**Analysis**:\n{analysis}"),
            ("ai", "```{response}```")
        ])
        
        few_shot_prompt_template = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            example_selector=example_selector
        )
        
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("human", instruction_prompt),
                ("ai", (
                        "Certainly! "
                        "Please provide the positive claims you believe could be true and the negative claims you are confident should be false. "
                        "This will help me propose a claim that aligns with your belief system."
                    )
                ),
                few_shot_prompt_template,
                ("human", "**Positive Claims**:\n{positive}\n\n**Negative Claims**:\n{negative}\n\n**Analysis**:\n{analysis}"),
            ]
        )
        
        return prompt_template
    
    @overrides
    def get_output_parser(self) -> Runnable:
        return ClaimSetSplitOutputParser()
    
    
@Step.register("refine-claim-set-split")
class RefineClaimSetSplitStep(Step):
    """ """
    @overrides
    def get_prompt_template(self) -> Runnable:
        """ """
        example_selector = ConstantExampleSelector()
        examples = [
            {
                "general_claim": "The Sydney Opera House was designed by a famous architect.",
                "positive": (
                    "- The Sydney Opera House was designed by Jørn Utzon.\n"
                    "- The Sydney Opera House was designed by Louis Kahn.\n"
                ),
                "negative": (
                    "- The Sydney Opera House was designed by Ludwig Mies van der Rohe.\n"
                    "- The Sydney Opera House was designed by Le Corbusier."
                ),
                "analysis": (
                    "The human subject believes that Sydney Opera House was designed by an architect designed with an aligned architectural style. "
                    "In the Positive Claims, "
                    "the subject feels that either Jørn Utzon or Louis Kahn designed the Sydney Opera House because both are prominent, "
                    "innovative architects, known for their modernist approaches. "
                    "Utzon, particularly, is strongly associated with the structure due to his reputation, "
                    "though the subject could be considering Kahn based on his influence in modern architecture.\n\n"
                    "The Negative Claims feature architects known for their distinct styles, "
                    "but whose aesthetic or body of work may not seem to align with the design of the Sydney Opera House in the subject’s perception. "
                    "For instance, Mies van der Rohe is associated with minimalism and sleek, functional "
                    "buildings, while Le Corbusier is known for his brutalist and rationalist designs, "
                    "which might contrast with the Opera House’s more organic and expressive form.\n\n"
                    "Thus, the commonality among the Positive Claims is that both architects could be perceived as plausible candidates for a modernist landmark, "
                    "while the Negative Claims involve architects whose styles feel incompatible with "
                    "the visual and architectural identity of the Sydney Opera House."
                ),
                "feedback": (
                    "**Alignment with positive claims**: The general claim, \"The Sydney Opera House was designed by a famous architect,\" "
                    "aligns well with each positive claim, as both architects (Jørn Utzon, Louis Kahn) are widely recognized as famous. "
                    "Therefore, the general claim remains true regardless of which positive claim is true.\n\n"
                    "**Contradiction with negative claims**: The negative claims involve architects (Ludwig Mies van der Rohe, Le Corbusier) who are also famous. "
                    "This means the general claim does not effectively contradict the negative claims, "
                    "since they also describe the design as being attributed to famous architects. "
                    "As a result, the general claim fails to fully exclude these possibilities.\n\n"
                    "**Redundant information**: The attribute \"famous architect\" is somewhat redundant. "
                    "Since all architects in both the positive and negative sets are famous, this description does not help in distinguishing between the two sets.\n\n"
                    "**Improvement**: To improve contrast, the general claim could focus on a specific attribute of the design style of architects asserted in the positive claims, "
                    "such as \"The Sydney Opera House was designed by an architect known for organic modernism.\" "
                    "This would help differentiate between the positive and negative claims more effectively."
                ),
                "refined_claim": "The Sydney Opera House was designed by an architect known for innovative and sculptural designs."
            }
        ]
        
        for example in examples:
            example_selector.add_example(example)

        instruction_prompt = (
            "Your task is to help an uncertain human subject refine a claim they want to make, "
            "based on a set of **positive claims** they believe could be true, and a set of **negative claims** they are confident should be false. "
            "Your goal is to **rewrite the proposed claim** in a way that:\n\n"
            "1. **Any of the positive claims should entail the rewritten claim** — meaning that the rewritten claim must be true if any of the positive claims are true.\n"
            "2. **Any of the negative claims should contradict the rewritten claim** — meaning that the rewritten claim must be false if any of the negative claims are true.\n\n"
            "In other words, your rewrite should be consistent with the belief system of the responder,"
            "ensuring that the claim aligns with the positive claims while contradicting the negative claims. "
            "Do not focus on factual accuracy or correcting potential misunderstandings in the claims. "
            "Your job is to achieve internal consistency with the given positive and negative claims. "
            "You will receive feedback after each round to help refine the claim further if needed, "
            "so multiple rounds of claim rewriting and feeback will be provided to arrive at the best refined version."
        )
        
        example_prompt = ChatPromptTemplate.from_messages([
            ("human", "**Positive Claims**:\n{positive}\n\n**Negative Claims**:\n{negative}\n\n**Analysis**:\n{analysis}\n\n---\n\n**R1 General Claim**: {general_claim}\n\n**R1 Feedback**:\n{feedback}\n\n---\n\nPlease further refine."),
            ("ai", "```{refined_claim}```")
        ])

        few_shot_prompt_template = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            example_selector=example_selector
        )
        
        prompt_template = ChatPromptTemplate.from_messages(
            [
                instruction_prompt,
                (
                    "ai", (
                        "Got it! To begin, could you please provide me with the **positive claims** and **negative claims** "
                        "that the responder has in mind? Once I have those, I can help rewrite the proposed claim to meet the specified requirements."
                    )
                ),
                few_shot_prompt_template,
                ("human", "**Positive Claims**:\n{positive}\n\n**Negative Claims**:\n{negative}\n\n**Analysis**:\n{analysis}\n\n---\n\n{iterations}\n\n---\n\nPlease further refine this claim.")
            ]
        )
        
        return prompt_template