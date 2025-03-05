"""Implement a parser step for reasoning-based
probability estimation.
"""

from overrides import overrides
import re
import numpy as np
from typing import (
    Tuple,
    Text,
    List,
    Dict,
    Any,
    Optional
)
from registrable import Registrable
from dataclasses import dataclass
from langchain_core.runnables import Runnable
from langchain_core.outputs import ChatGeneration, Generation
from langchain_core.output_parsers import BaseOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_interface.steps import Step
from langchain_interface.instances import LLMResponse


# define our own exception of parsing failure
class ParsingFailure(Exception):
    """ """
    def __init__(
        self,
        message: Text,
        text_to_parse: Text
    ):
        """ """
        super().__init__(message)
        self.text_to_parse = text_to_parse


@dataclass(frozen=True, eq=True)
class ReasoningBasedProbResponse(LLMResponse):
    prob: Optional[float]
    reasoning: Optional[Text]

    
class ReasoningBasedProbOutputParser(BaseOutputParser[ReasoningBasedProbResponse]):
    """ """
    
    @overrides
    def parse_result(self, result: list[Generation], *, partial = False):
        """ """
        
        result_processing: List[ChatGeneration] = result
        reasoning = result_processing[0].message.additional_kwargs.get("reasoning_content", None)
        textual_result = self.parse(result_processing[0].text)
        
        return ReasoningBasedProbResponse(
            messages=textual_result.messages,
            prob=textual_result.prob,
            reasoning=reasoning
        )
        
    
    @overrides
    def parse(self, text: Text) -> ReasoningBasedProbResponse:
        """ """
        
        # extract probability from text,
        # the probability is surrounded by ``` ```
        
        try:
            prob_text = re.search(r"```\s*(.*?)\s*```", text).group(1)
            prob = float(prob_text)
            if prob > 1:
                # likely the model is outputting percentages.
                prob = prob / 100
        except AttributeError:
            raise ParsingFailure("Failed to extract probability from text. ``` block not found.", text_to_parse=text)
        except ValueError:
            if prob_text.endswith("%"):
                try:
                    prob = float(prob_text[:-1]) / 100
                except ValueError:
                    raise ParsingFailure("Failed to extract probability from text. Invalid format.", text_to_parse=text)
            else:
                # it might be the case that it is expressed in "a / b"
                splitted_rep = re.search(r"\s*(\d+)\s*/\s*(\d+)\s*", prob_text)
                if splitted_rep is not None:
                    try:
                        prob = float(splitted_rep.group(1)) / float(splitted_rep.group(2))
                    except ValueError:
                        raise ParsingFailure("Failed to extract probability from text. Invalid format.", text_to_parse=text)
                raise ParsingFailure("Failed to extract probability from text. Invalid format.", text_to_parse=text)
        
        return ReasoningBasedProbResponse(
            messages=text,
            prob=prob,
            reasoning=None
        )
    
    
class ReasoningBasedProbStep(Step):
    """ """
    
    @overrides
    def get_prompt_template(self) -> Runnable:
        """ """

        return ChatPromptTemplate.from_messages(
            [
                (
                    "human",
                    "Please estimate the probability "
                    "of a given hypothesis happening "
                    "given the following premises. "
                    "Your answer should be a number between 0 and 1. "
                    "Please be as fine-grained as possible."
                    "Your final answer should be surrounded by ``` ```\n\n"
                    "Premises: {premise}\n"
                    "Hypothesis: {hypothesis}\n"
                ),
            ]
        )
    
    @overrides
    def get_output_parser(self) -> Runnable:
        """ """
        return ReasoningBasedProbOutputParser()