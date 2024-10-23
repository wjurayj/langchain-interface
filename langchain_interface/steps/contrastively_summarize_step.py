""" """

import re
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

# TODO: use customer downloaded examples for example selector
from ..example_selectors import ConstantExampleSelector, ExampleSelector
from .step import Step, FewShotStep
from ..instances.instance import LLMResponse


@dataclass(frozen=True, eq=True)
class ContrastivelySummarizeResponse(LLMResponse):
    contrasting_summary: Text
    
class ContrastivelySummarizeOutputParser(BaseOutputParser):
    """ """
    
    @overrides
    def parse(self, text: Text) -> LLMResponse:
        """ """
        return ContrastivelySummarizeResponse(contrasting_summary=text.strip(), messages=text)
    
    @property
    def _type(self) -> Text:
        return "contrastively-summarize-output-parser"
    
    
@Step.register("contrastively-summarize")
class ContrastivelySummarizeStep(FewShotStep):
    """ """
    
    def __init__(
        self,
        example_selector: Optional[ExampleSelector] = None,
    ):
        if example_selector is None:
            example_selector = ConstantExampleSelector()
            examples = [
                {
                    "positive": (
                        "- Tony's favorite color is blue.\n"
                        "- Tony's favorite color is red.\n"
                        "- Tony's favorite color is green.\n"
                        "- Tony's favorite color is yellow.\n"
                        "- Tony's favorite color is orange."
                    ),
                    "negative": (
                        "- Tony's favorite color is black.\n"
                        "- Tony's favorite color is white.\n"
                        "- Tony's favorite color is purple."
                    ),
                    "summary": (
                        "The human subject believes that Tony's favorite color is likely to be a vibrant, warm color. "
                        "The Positive Claims—blue, red, green, yellow, "
                        "and orange—are all colors typically seen as vibrant, warm, and widely liked. "
                        "These colors may be associated with more common or socially acceptable favorite colors, "
                        "perhaps linked with emotions like calmness, energy, nature, happiness, or excitement. "
                        "The subject might believe Tony prefers one of these colors based on their perceived popularity or the positive emotions these colors evoke.\n\n"
                        "In contrast, the Negative Claims—black, white, and purple—are more unconventional or might carry different connotations. "
                        "Black and white could be perceived as too neutral, stark, or associated with "
                        "formality rather than personal preference for something as subjective as a \"favorite\" color. "
                        "Purple may be seen as more niche or uncommon, which could make the subject believe it is less likely to be Tony's favorite.\n\n"
                        "The commonality among the Positive Claims is that they represent widely accepted, common colors with broad appeal. "
                        "The Negative Claims involve colors that may feel too neutral, uncommon, or specific to be associated with a favorite."
                    )
                },
                {
                    "positive": (
                        "- The best football player of all time is Pelé.\n"
                        "- The best football player of all time is Diego Maradona.\n"
                        "- The best football player of all time is Lionel Messi."
                    ),
                    "negative": (
                        "- The best football player of all time is Cristiano Ronaldo.\n"
                        "- The best football player of all time is Zinedine Zidane.\n"
                        "- The best football player of all time is Johan Cruyff."
                    ),
                    "summary": (
                        "The human subject has the general belief that the best football player of all time comes from South America. "
                        "The Positive Claims — Pelé, Maradona, and Messi—are represent South American football, which is historically known for producing highly skilled, flamboyant, and creative players. "
                        "The Negative Claims — Ronaldo, Zidane, and Cruyff — represent European football, which is known for its tactical, disciplined, and team-oriented approach. "
                        "The subject may believe that the best football player of all time should embody the flair, individual brilliance, and creativity associated with South American football, hence favoring Pelé, Maradona, or Messi.\n\n"
                        "In summary, the culture and regional preferences of the here contrasts the South American passion for individual brilliance with the European focus on tactical mastery and professionalism, reflecting differing regional veiws on what constitutes greatness in football."
                    )
                }
            ]
            
            for example in examples:
                example_selector.add_example(example)

        super().__init__(example_selector=example_selector)

    @overrides
    def get_prompt_template(self) -> Runnable:
        """ """

        instruction_prompt = (
            "Your role is to help a human subject better understand the underlying inductive bias behind their beliefs, like regional preferences, cultural associations, temporal biases, nationalities, characteristics etc."
            "You will be presented with two sets of claims. The first set, labeled **\"Positive Claims\"**, "
            "contains statements that the subject believes one of which is true. The second set, labeled **\"Negative Claims\"**, "
            "contains statements that the subject believes should all be false. Your task is to: \n\n"
            "1. **Identify the key contrasting attributes** the most contrasting evidence that differentiate between the Positive and Negative Claims.\n"
            "2. **Explain the commonalities within the Positive Claims**.\n"
            "3. **Explain the commonalities within the Negative Claims**.\n\n"
            "Try to be as specific as possible in your analysis. That is, instead of saying \"The subject likely has a preference for football players potentially biased by regional or cultural factors,\", "
            "you should say that \"The subject likely believes that the best football player of all time comes from South America.\" "
            "Focus on understanding potential inductive biases the human subject might be using. "
            "It's important to analyze their belief system without correcting it or judging the inductive bias employed by the human subject. "
            "The goal is to explore the logic that supports their beliefs and offer possible explanations for why they think the way they do.\n\n"
            "Keep your response concise (under 200 words) and centered on summarizing and explaining the belief framework rather than focusing on correctness."
        )
        
        example_prompt = ChatPromptTemplate.from_messages([
            ("human", "**Positive Claims**:\n{positive}\n\n**Negative Claims**:\n{negative}"),
            ("ai", "{summary}"),
        ])
        
        few_shot_prompt_template = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            example_selector=self._example_selector
        )
        
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("human", instruction_prompt),
                ("ai", (
                        "Sure! Please provide the Positive Claims and Negative Claims sets, and I will proceed with the explanation based on the provided information."
                    )
                ),
                few_shot_prompt_template,
                ("human", "**Positive Claims**:\n{positive}\n\n**Negative Claims**:\n{negative}"),
            ]
        )
        
        return prompt_template
    
    @overrides
    def get_output_parser(self) -> Runnable:
        
        return ContrastivelySummarizeOutputParser()