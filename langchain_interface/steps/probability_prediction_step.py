"""Implement a step for direct prompting the probability from the LLM.
"""

from overrides import overrides
import re
import json
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


class ProbResponse(LLMResponse):
    probability = None
    reasoning: Optional[Text]


class ProbExtractParser(BaseOutputParser[str]):
    """
    Parser that extracts probability values from LLM output text.
    
    The parser looks for probability values in the following formats:
    - '''number'''
    - boxed{number}
    
    Returns:
        dataclass with 'probability' (float or None) and 'reasoning' (str)
    """
    def parse(self, text: Text) -> Dict[str, Any]:
    
        result = ProbResponse(reasoning=text)

        # Try different probability formats
        patterns = [
            r"'''([\d.]+)'''",  # Triple quote format
            r"boxed\{([\d.]+)\}",  # LaTeX boxed format
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    prob = float(match.group(1))
                    # Validate probability is in [0,1]
                    if 0 <= prob <= 1:
                        result.probability = prob
                        return result
                except (ValueError, TypeError):
                    continue
        
        return result
    
    @property
    def _type(self) -> str:
        return "probability"

class ProbabilityPredictionStep(Step):
    
    @overrides
    def get_prompt_template(self) -> Runnable:
        return ChatPromptTemplate.from_messages(
            [("system", "You are a helpful assistant good at probabilistic reasoning in the real world setting"),
             ("human","""Given a premise and a hypothesis, evaluate the probability of the hypothesis being true based on the information provided in the premise, supplemented by world knowledge and probabilistic reasoning. Specifically:
1. Use relevant world knowledge to assess contextual factors (e.g., demographics, common practices, or statistical distributions) that may influence the likelihood of the hypothesis given the premise.
2. Perform the probabilistic reasoning to estimate the conditional probability P(Hypothesis | Premise).
3. Assign a probability score between [0, 1] that quantifies P(Hypothesis | Premise). Ensure this score reflects the strength of the connection between the premise and hypothesis based on probabilistic reasoning and world knowledge.

Premise: {premise}
Hypothesis: {hypothesis}
              
Your final probability estimate should be a value in the range \([0,1]\), as fine-grained as possible, and formatted as follows: '''your_final_probability'''

For example, if the estimated probability is 0.0653, the output should be: '''0.0653'''
              """)] 
        )        
    
    @overrides
    def get_output_parser(self) -> Runnable:
        """ """
        return ProbExtractParser()