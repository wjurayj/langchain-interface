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
class VagueAnswerResponse(LLMResponse):
    """Response for decontextualization.
    """
    general_answer: Text
    
    
class VagueAnswerOutputParser(BaseOutputParser[VagueAnswerResponse]):
    """Parse the output of the decontextualization model.
    """
    def parse(self, text: Text) -> Dict:
        cleaned_text = text.strip()

        # find the text wrapped by the code block
        match = re.search(r"```(.*?)```", cleaned_text, re.DOTALL)
        if match is None:
            general_answer = None
        else:
            general_answer = match.group(1).strip()
            
        return VagueAnswerResponse(messages=text, general_answer=general_answer)
    
    @property
    def _type(self) -> Text:
        return "vague-answer-output-parser"


@Step.register("vague-answer")
class VagueAnswerStep(Step):
    """ """
    @overrides
    def get_prompt_template(self) -> Runnable:
        """ """

        example_selector = ConstantExampleSelector()
        examples = [
            {
                "question": "Who's the best football player of all time?",
                "discussion": (
                    "**Group A: Pelé, Diego Maradona, Lionel Messi**\n"
                    "*Time Period*: Spanning from the 1950s to the present.\n"
                    "*Geographic Origin*: South America (Brazil and Argentina).\n"
                    "*Influence*: Known for their extraordinary dribbling skills, creativity, and playmaking abilities.\n"
                    "*Philosophy*: These players are celebrated for their flair, technical prowess, "
                    "and ability to change the course of a game single-handedly. "
                    "Pelé (Brazil) is renowned for his goal-scoring and three World Cup victories. "
                    "Maradona (Argentina) is famous for his \"Hand of God\" goal and his incredible dribbling, "
                    "particularly in the 1986 World Cup. Messi (Argentina) is known for his consistency, "
                    "vision, and record-breaking achievements with FC Barcelona and Argentina.\n"
                    "*Distinct Features*: Emphasis on individual brilliance, creativity, and a deep connection with their home countries' footballing culture.\n\n"
                    "**Group B: Cristiano Ronaldo, Zinedine Zidane, Johan Cruyff**\n"
                    "*Time Period*: Spanning from the 1970s to the present.\n"
                    "*Geographic Origin*: Europe (Portugal, France, Netherlands).\n"
                    "*Influence*: Known for their versatility, tactical intelligence, and leadership on the field.\n"
                    "*Philosophy*: These players are celebrated for their physical prowess, strategic thinking, and ability to perform in crucial moments. "
                    "Ronaldo (Portugal) is known for his athleticism, goal-scoring, and adaptability across leagues. "
                    "Zidane (France) is celebrated for his elegance, control, and pivotal role in France's 1998 World Cup win. "
                    "Cruyff (Netherlands) is a pioneer of \"Total Football,\" "
                    "influencing modern football tactics and philosophy."
                    "*Distinct Features*: Emphasis on tactical intelligence, versatility, and significant contributions to both club and national team success.\n"
                    "**Key Differentiation**: Group A is characterized by South American flair, individual brilliance, and a deep cultural impact on football. "
                    "Group B is defined by European tactical intelligence, versatility, and a strategic approach to the game."
                ),
                "general_answer": "The best football player of all time comes from South America.",
            },
            {
                "question": "Who designed the Sydney Opera House?",
                "discussion": (
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
                ),
                "general_answer": "The Sydney Opera House was designed by an architect known for sculptural designs.",
            }
        ]
        
        for example in examples:
            example_selector.add_example(example)
            
        instruction_prompt = (
            "Suppose a human subject is going to responde to a question but they don't know the exact answer. "
            "However, from some discussion they split possible answers into two groups, A and B. "
            "They would like to response with a less specific answer that indicates the answer is from group A. "
            "Given these discussions, please provide a concise and simple answer indicating the answer is from group A. "
            "Questions and discussions are provided below."
        )
            
        example_prompt = ChatPromptTemplate.from_messages([
            ("human", "**Question**: {question}\n\n**Discussion**:\n{discussion}"),
            ("ai", "```{general_answer}```")
        ])
        
        few_shot_prompt_template = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            example_selector=example_selector
        )

        prompt_template = ChatPromptTemplate.from_messages([
            ("human", instruction_prompt),
            ("ai", "Sure! Please provide the question and discussion you'd like me to generate a vague answer for."),
            few_shot_prompt_template,
            ("human", "**Question**: {question}\n\n**Discussion**:\n{discussion}"),
        ])
        
        return prompt_template
    
    @overrides
    def get_output_parser(self) -> Runnable:
        return VagueAnswerOutputParser()