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
from ..example_selectors import ConstantExampleSelector, ExampleSelector
from .step import Step, FewShotStep
from ..instances.instance import LLMResponse


@dataclass(frozen=True, eq=True)
class GeneralClaimFeedbackResponse(LLMResponse):
    verbal_feedback: Text
    need_further_refinement: bool

    
class GeneralClaimFeedbackOutputParser(BaseOutputParser):
    """ """
    
    @overrides
    def parse(self, text: Text) -> LLMResponse:
        """ """
        # chunks = text.split("\n\n")
        # assert len(chunks) == 4, f"Expected 4 chunks, but got {len(chunks)}: {chunks}"
        
        # output_dict = {}
        
        # for chunk in chunks:
        #     matched = re.match(r"\*\*(.*?)\*\*: (.*)", chunk)
        #     if matched.group(1) == "Alignment with positive claims":
        #         output_dict["positive_alignment"] = matched.group(2)
        #     elif matched.group(1) == "Contradiction with negative claims":
        #         output_dict["negative_alignment"] = matched.group(2)
        #     elif matched.group(1) == "Redundant information":
        #         output_dict["redundancy"] = matched.group(2)
        #     elif matched.group(1) == "Improvement":
        #         output_dict["improvement"] = matched.group(2)
        
        match = re.match(r"(.*?)```\*\*Need Further Refinement\*\*: (.*?)```", text, re.DOTALL)
        verbal_feedback = match.group(1).strip()
        need_further_refinement = match.group(2).strip() == "True"

        return GeneralClaimFeedbackResponse(verbal_feedback=verbal_feedback, need_further_refinement=need_further_refinement, messages=text)
    
    @property
    def _type(self) -> Text:
        return "general-claim-feedback-output-parser"
    
    
@Step.register("general-claim-feedback")
class GeneralClaimFeedbackStep(FewShotStep):
    
    def __init__(
        self,
        example_selector: Optional[ExampleSelector] = None
    ):
        if example_selector is None:
            example_selector = ConstantExampleSelector()
            examples = [
                {
                    # "question": "Who designed the Sydney Opera House?",
                    "general_claim": "The Sydney Opera House was designed by a famous architect.",
                    "positive": (
                        "- The Sydney Opera House was designed by Jørn Utzon.\n"
                        "- The Sydney Opera House was designed by Louis Kahn.\n"
                        "- The Sydney Opera House was designed by Tadao Ando."
                    ),
                    "negative": (
                        "- The Sydney Opera House was designed by Ludwig Mies van der Rohe.\n"
                        "- The Sydney Opera House was designed by Le Corbusier."
                    ),
                    "feedback": (
                        "**Alignment with positive claims**: The general claim, \"The Sydney Opera House was designed by a famous architect,\" "
                        "aligns well with each positive claim, as all three architects (Jørn Utzon, Louis Kahn, and Tadao Ando) are widely recognized as famous. "
                        "Therefore, the general claim remains true regardless of which positive claim is true.\n\n"
                        "**Contradiction with negative claims**: The negative claims involve architects (Ludwig Mies van der Rohe, Le Corbusier) who are also famous. "
                        "This means the general claim does not effectively contradict the negative claims, "
                        "since they also describe the design as being attributed to famous architects. "
                        "As a result, the general claim fails to fully exclude these possibilities.\n\n"
                        "**Redundant information**: The attribute \"famous architect\" is somewhat redundant. "
                        "Since all architects in both the positive and negative sets are famous, this description does not help in distinguishing between the two sets.\n\n"
                        "**Improvement**: To improve contrast, the general claim could focus on a specific attribute of Jørn Utzon's design, "
                        "such as \"The Sydney Opera House was designed by an architect known for organic modernism.\" "
                        "This would help differentiate between the positive and negative claims more effectively."
                    ),
                    "need_further_refinement": True
                }
            ]
            
            for example in examples:
                example_selector.add_example(example)
            
        super().__init__(example_selector=example_selector)
    
    @overrides
    def get_prompt_template(self) -> Runnable:
        """ """
        
        example_prompt = ChatPromptTemplate.from_messages([
            ("human", "**Claim to Refine**: {general_claim}\n**Positive Claims**:\n{positive}\n\n**Negative Claims**:\n{negative}"),
            ("ai", "{feedback}\n\n```\n**Need Further Refinement**: {need_further_refinement}\n```")
        ])
        
        instruction_prompt = (
            "You are asked to evaluate how well a general belief claim represents the beliefs "
            "expressed in a set of specific positive claims and how it contrasts with a set of negative claims. You must assess the following aspects of the general claim:\n\n"
            "1. **Alignment with Positive Claims**: Does the general claim align with the positive claims? "
            "Specifically, is the general claim true regardless of which individual positive claim is true?\n"
            "2. **Contradiction with Negative Claims**: Does the general claim contradict the negative claims? "
            "Specifically, if any of the negative claims were true, would that contradict the general claim?\n"
            "3. **Redundancy**: Does the general claim contain any redundant information that doesn't help distinguish the positive claims from the negative claims?\n"
            "4. **Potential Improvement**: Suggest how the general claim could be refined to better distinguish the positive claims from the negative claims by focusing on attributes that more clearly contrast the two sets.\n\n"
            "5. **Need Further Refine**: After evaluting all previous aspects, you need to provide a binary assessment of whether the general claim needs further refinement or not."
            "**Note:** You must not provide factual correctness feedback on the beliefs themselves to avoid giving any advantage to the human subject in future assessments.\n\n"
            "**Input Format:**\n"
            "- **General Claim**: [The general claim to evaluate]\n"
            "- **Positive Claims**: [A set of positive claims]\n"
            "- **Negative Claims**: [A set of negative claims]"
        )
        
        few_shot_prompt_template = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            example_selector=self._example_selector
        )
        
        prompt_template = ChatPromptTemplate.from_messages(
            [
                instruction_prompt,
                ("ai", "Please provide the general claim, along with the positive and negative claim sets, so I can offer feedback based on the criteria you outlined."),
                few_shot_prompt_template,
                ("human", "**General Claim**: {general_claim}\n\n**Positive Claims**:\n{positive}\n\n**Negative Claims**:\n{negative}")
            ]
        )
        
        return prompt_template
    
    @overrides
    def get_output_parser(self) -> Runnable:
        
        return GeneralClaimFeedbackOutputParser()