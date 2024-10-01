"""An interface used to decontextualize
a single claim from a given context.
"""


from dataclasses import dataclass
from overrides import overrides
from typing import Union, Text, List, Dict, Optional, Callable, Any
import re

from langchain_core.runnables.config import RunnableConfig
from langchain_core.runnables.base import Runnable
from langchain.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)
from langchain_core.output_parsers import BaseOutputParser

# TODO: use customer downloaded examples for example selector
from ..example_selectors import ConstantExampleSelector
from .step import (
    Step
)
from ..instances.instance import LLMResponse


DECONTEXTUALIZE_PROMPT = """Vague references include but are not limited to:
- Pronouns (e.g. "his", "they", "her")
- Unknown entities (e.g., "this event", "the research", "the invention")
- Non-full names (e.g., "Jeff..." or "Bezos..." when referring to Jeff Bezos)

Instructions:
1. The following STATEMENT has been extracted from the broader context of the given RESPONSE.
2. Modify the STATEMENT by replacing vague references with the proper entities from the RESPONSE that they are referring to.
3. You MUST NOT change any of the factual claims made by the original STATEMENT.
4. You MUST NOT add any additional factual claims to the original STATEMENT. For example, given the response "Titanic is a movie starring Leonardo DiCaprio," the statement "Titanic is a movie" should not be changed.
5. Before giving your revised statement, think step-by-step and show your reasoning. As part of your reasoning, be sure to identify the subjects in the STATEMENT and determine whether they are vague references. If they are vague references, identify the proper entity that they are referring to and be sure to revise this subject in the revised statement.
6. After showing your reasoning, provide the revised statement and wrap it in a markdown code block."""


@dataclass(frozen=True, eq=True)
class DecontextualizationResponse(LLMResponse):
    """Response for decontextualization.
    """
    revised: Text
    
    
class DecontextualizationOutputParser(BaseOutputParser[DecontextualizationResponse]):
    """Parse the output of the decontextualization model.
    """
    def parse(self, text: Text) -> Dict:
        cleaned_text = text.strip()
        # items = cleaned_text.split("\n")
        # return {"responses": DecontextualizationResponse(messages=text, claims=[item.replace('- ', "") for item in items])}

        # find the text wrapped by the code block
        match = re.search(r"```(.*?)```", cleaned_text, re.DOTALL)
        if match is None:
            revised = None
        else:
            revised = match.group(1).strip()
        
        return DecontextualizationResponse(messages=text, revised=revised)
    
    @property
    def _type(self) -> str:
        return "decontextualization_output_parser"


@Step.register("decontextualize")
class DecontextualizationStep(Step):
    
    @overrides
    def get_prompt_template(self) -> Runnable:
        """
        input: statement to be revised
        context: context of the statement to revise with
        """
        
        example_selector = ConstantExampleSelector()
        examples = [
            {
                "input": "Acorns is a company.",
                "context": "Acorns is a financial technology company founded in 2012 by Walter Cruttenden, Jeff Cruttenden, and Mark Dru that provides micro-investing services. The company is headquartered in Irvine, California.",
                "reasoning": 'The subject in the statement "Acorns is a company" is "Acorns". "Acorns" is not a pronoun and does not reference an unknown entity. Furthermore, "Acorns" is not further specified in the RESPONSE, so we can assume that it is a full name. Therefore "Acorns" is not a vague reference. Thus, the revised statement is:',
                "output": "Acorns is a company.",
            },
            {
                "input": "He teaches courses on deep learning.",
                "context": "After completing his Ph.D., Quoc Le joined Google Brain, where he has been working on a variety of deep learning projects. Le is also an adjunct professor at the University of Montreal, where he teaches courses on deep learning.",
                "reasoning": 'The subject in the statement "He teaches courses on deep learning" is "he". From the RESPONSE, we can see that this statement comes from the sentence "Le is also an adjunct professor at the University of Montreal, where he teaches courses on deep learning.", meaning that "he" refers to "Le". From the RESPONSE, we can also see that "Le" refers to "Quoc Le". Therefore "Le" is a non-full name that should be replaced by "Quoc Le." Thus, the revised response is:',
                "output": "Quoc Le teaches courses on deep learning.",
            },
            {
                "input": 'The television series is called "You\'re the Worst."',
                "context": 'Xochitl Gomez began her acting career in theater productions, and she made her television debut in 2016 with a guest appearance on the Disney Channel series "Raven\'s Home." She has also appeared in the television series "You\'re the Worst" and "Gentefied."',
                "reasoning": 'The subject of the statement "The television series is called "You\'re the Worst."" is "the television series". This is a reference to an unknown entity, since it is unclear what television series is "the television series". From the RESPONSE, we can see that the STATEMENT is referring to the television series that Xochitl Gomez appeared in. Thus, "the television series" is a vague reference that should be replaced by "the television series that Xochitl Gomez appeared in". Thus, the revised response is:',
                "output": 'The television series that Xochitl Gomez appeared in is called "You\'re the Worst.',
            },
            {
                "input": "Dean joined Google.",
                "context": "Jeff Dean is a Google Senior Fellow and the head of Google AI, leading research and development in artificial intelligence. Dean joined Google in 1999 and has been essential to its continued development in the field.",
                "reasoning": 'The subject of the statement "Dean joined Google" is "Dean". From the response, we can see that "Dean" is the last name of "Jeff Dean". Therefore "Dean" is a non-full name, making it a vague reference. It should be replaced by "Jeff Dean", which is the full name. Thus, the revised response is:',
                "output": "Jeff Dean joined Google.",
            }
        ]

        for example in examples:
            example_selector.add_example(example)


        input_example_prompt = "STATEMENT:\n{input}\n\nRESPONSE:\n{context}"
        output_example_prompt = "REASONING:\n{reasoning}\n\nREVISED STATEMENT:\n```\n{output}\n```"

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
                ("human", DECONTEXTUALIZE_PROMPT),
                ("ai", "Sure, please provide me with statements you want me to revise."),
                fewshot_prompt_template,
                ("human", input_example_prompt),
            ]
        )
        
        return prompt_template
    
    @overrides
    def get_output_parser(self) -> Runnable:
        return DecontextualizationOutputParser()