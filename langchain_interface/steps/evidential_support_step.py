""" Implement a verification step for the language chain. """

from dataclasses import dataclass
from overrides import overrides
from typing import Union, Text, List, Dict, Optional, Callable, Any, Literal

from langchain_core.runnables.config import RunnableConfig
from langchain_core.runnables.base import Runnable
from langchain.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)
from langchain_core.output_parsers import BaseOutputParser
from .step import (
    Step
)
import re

# TODO: use customer downloaded examples for example selector
from ..example_selectors import ConstantExampleSelector
from .step import Step
from ..instances.instance import LLMResponse, Instance


@dataclass(frozen=True, eq=True)
class EvidentialSupportResponse(LLMResponse):
    label: Literal["Entailment", "Contradiction", "Neutral"]
    reasoning: Optional[Text]
    premise: Optional[Text] = None
    hypothesis: Optional[Text] = None
    
    
class EvidentialSupportOutputParser(BaseOutputParser[EvidentialSupportResponse]):
    """Parse the output of the verification model.
    """
    def parse(self, text: Text) -> Dict:
        cleaned_text = text.strip()
        
        # find ``` ``` block
        match = re.match(r"(.*?)```(.*?)```", cleaned_text, re.DOTALL)
        reasoning = match.group(1).strip()
        match_text = match.group(2)
        submatch = re.search(r"Label: (.*)", match_text.strip(), re.DOTALL)
        submatch_text = submatch.group(1)
        
        label = submatch_text.strip()
        assert label in ["Entailment", "Contradiction", "Neutral"], f"Invalid label: {label}"
        
        return EvidentialSupportResponse(
            messages=text,
            reasoning=reasoning,
            label=label
        )
    
    @property
    def _type(self) -> str:
        return "evidential-support"
    

@Step.register("evidential-support")
class EvidentialSupportStep(Step):
    @overrides
    def get_prompt_template(self) -> Runnable:
        """
        input: input text to be decomposed.
        """
        
        example_selector = ConstantExampleSelector()
        examples = [
            {
                "premise": "Sherlock Holmes lived at 221B Baker Street.",
                "hypothesis": "Sherlock Holmes lived in London.",
                "reasoning": "The premise that \"Sherlock Holmes lived at 221B Baker Street\" entails the hypothesis that \"Sherlock Holmes lived in London\" because Baker Street is located in London. Therefore, if Sherlock Holmes lived at 221B Baker Street, it logically follows that he lived in London.",
                "label": "Entailment"
            },
            {
                "premise": "Einstein has been to Korea.",
                "hypothesis": "Einstein has been to Asia.",
                "reasoning": "The premise that \"Einstein has been to Korea\" entails the hypothesis that \"Einstein has been to Asia\" because Korea is a country in Asia. Therefore, if Einstein has been to Korea, it follows that he has been to Asia.",
                "label": "Entailment"
            },
            {
                "premise": "During his life, Lu You has written 91,630 poems.",
                "hypothesis": "Records show that the number of poems composed by Lu You hardly exceeds 20,000.",
                "reasoning": "The premise states that \"Lu You has written 91,630 poems,\" while the hypothesis claims that \"the number of poems composed by Lu You hardly exceeds 20,000.\" These two statements are in direct conflict with each other, as the premise indicates a much larger number of poems than the hypothesis suggests. Therefore, they contradict each other.",
                "label": "Contradiction"
            },
            {
                "premise": "Using a variety of techniques to bypass security measures, hackers sought access to myriad email accounts.",
                "hypothesis": "Russian officials have denied any involvement in the hacking activities.",
                "reasoning": "The premise discusses hackers attempting to access email accounts using various techniques, while the hypothesis states that Russian officials have denied any involvement. The two statements address different aspects: the premise focuses on hacking activities, and the hypothesis on an official denial. There is no direct connection or contradiction between the two, making the relationship neutral.",
                "label": "Neutral"
            },
        ]
        
        for example in examples:
            example_selector.add_example(example)
        

        instruction = "Does the premise entail the hypothesis? Respond in one of `[\"entailment\", \"neutral\", \"contradiction\"]`:"
        input_example_prompt = "Premise: {premise}\nHypothesis: {hypothesis}"
        output_example_prompt = "{reasoning}\n```\nLabel: {label}\n```"
        
        example_prompt = ChatPromptTemplate.from_messages([
            ("human", input_example_prompt),
            ("ai", output_example_prompt),
        ])

        fewshot_prompt_template = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            example_selector=example_selector,
        )

        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("human", instruction),
                ("ai", "Please provide the premise and hypothesis so I can evaluate them and respond accordingly."),
                fewshot_prompt_template,
                ("human", input_example_prompt),
            ]
        )

        return prompt_template
    
    @overrides
    def get_output_parser(self) -> Runnable:
        return EvidentialSupportOutputParser()