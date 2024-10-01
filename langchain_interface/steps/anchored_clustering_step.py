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

# TODO: use customer downloaded examples for example selector
from ..example_selectors import ConstantExampleSelector
from .step import Step
from ..instances.instance import LLMResponse


@dataclass(frozen=True, eq=True)
class AnchoredClusteringResponse(LLMResponse):
    increments: List[Text]
    
    
class AnchoredClusteringOutputParser(BaseOutputParser[AnchoredClusteringResponse]):
    """ """
    
    @overrides
    def parse(self, text: Text) -> LLMResponse:
        
        all_matched = re.findall(r"```python(.*?)```", text, re.DOTALL)

        items = None
        
        for matched in all_matched:
            try:
                submatch = re.search(r"increments = (\[.*?\])\s", matched, re.DOTALL)
                if submatch is not None:
                    items = ast.literal_eval(submatch.group(1))
                # submatch = re.search(r"(\[.*?\])", matched, re.DOTALL)
                # items = ast.literal_eval(submatch.group(1))
                break
            except Exception:
                continue
            
        if items is None:
            print(all_matched[-1])
            submatch = re.search(r"\[.*?\]\s", all_matched[-1], re.DOTALL)
            try:
                items = ast.literal_eval(submatch.group(0))
            except Exception:
                print('=' * 50 + '\n' + text + '\n' + '=' * 50)
        
        return AnchoredClusteringResponse(
            increments=items,
            messages=text
        )
        
    @property
    def _type(self) -> Text:
        return "anchored-clustering-output-parser"


@Step.register("anchored-clustering")
class AnchoredClusteringStep(Step):
    @overrides
    def get_prompt_template(self) -> Runnable:

        example_selector = ConstantExampleSelector()
        examples = [
            {
                "selected": json.dumps([
                    "Jørn Utzon"
                ]),
                "num_select": 1,
                "candidates": json.dumps([
                    "Ludwig Mies van der Rohe",
                    "Louis Kahn",
                    "Le Corbusier"
                ]),
                "response": (
                    "Based on your input, the selected item is **Jørn Utzon**, "
                    "and the candidates are **Le Corbusier**, **Louis Kahn**, and **Ludwig Mies van der Rohe**. "
                    "All of these individuals are well-known architects, so we can base the similarity on architectural style, "
                    "philosophy, or impact on modern architecture.\n\n"
                    "- **Jørn Utzon** was a Danish architect most famous for designing the Sydney Opera House, a key example of modern expressionist architecture.\n"
                    "- **Candidates**:\n"
                    "- **Le Corbusier**: A pioneer of modern architecture, known for his contributions to urban planning and minimalistic designs.\n"
                    "- **Louis Kahn**: Known for monumental architecture and his use of light in design.\n"
                    "- **Ludwig Mies van der Rohe**: A leading figure in modernist architecture, known for his minimalist and \"less is more\" philosophy.\n\n"
                    "In terms of architectural style and influence, **Jørn Utzon** is often associated with a modern, organic, and expressionist style. "
                    "Out of the candidates, **Louis Kahn** shares a closer similarity due to his focus on monumentality, "
                    "thoughtful use of light, and organic forms, which resonates more closely with Utzon's architectural philosophy.\n\n"
                    "Thus, the result would be:\n\n"
                    "```python\n"
                    "increments = [\"Louis Kahn\"]\n"
                    "```\n\n"
                    "This selection is based on shared architectural principles and design philosophy."
                )
            },
            {
                "selected": json.dumps([
                    "William Butler Yeats",
                ]),
                "num_select": 1,
                "candidates": json.dumps([
                    "Agatha Christie",
                    "Benjamin Franklin",
                    "Napoléon Bonaparte",
                ]),
                "response": (
                    "Based on your inputs, we are tasked with selecting 1 item from the list of candidates that is most similar to the selected item, "
                    "**\"William Butler Yeats\"**. "
                    "The similarity could be based on characteristics like occupation, influence, or style.\n\n"
                    "Here's a possible reasoning for this:\n\n"
                    "- **William Butler Yeats** was an Irish poet and one of the foremost figures of 20th-century literature.\n"
                    "- **Candidates**:\n"
                    "- **Benjamin Franklin**: American polymath, writer, scientist, diplomat.\n"
                    "- **Napoleon**: French military leader and emperor.\n"
                    "- **Agatha Christie**: British writer known for her detective novels.\n\n"
                    "In this case, based on occupation (both are writers), **Agatha Christie** is most similar to **William Butler Yeats**.\n\n"
                    "Thus, the result would be:\n\n"
                    "```python\n"
                    "increments = [\"Agatha Christie\"]\n"
                    "```\n\n"
                    "This selection is made based on the shared characteristic of being prominent literary figures."
                )
            },
        ]

        for example in examples:
            example_selector.add_example(example)
        
        instruction_prompt = (
            "Given a list of already **selected items**, "
            "your task is to find **K additional items** from the list of **candidates** that are **most similar** "
            "to the items already selected. "
            "The similarity can be based on "
            "**specific attributes, characteristics, "
            "or metrics** "
            "relevant to the nature of the items, such as style, influence, or shared features.\n\n"
            "You should return a list of **K items** from the **candidates** list and store them in a variable named **increments**. \n\n"
            "---\n\n"
            "### Inputs:\n"
            "1. **K**: The number of additional items to add to the selected list.\n"
            "2. **selected**: A list of already selected items (could be names, objects, etc.).\n"
            "3. **candidates**: A list of candidate items from which we need to select the K most similar items.\n\n"
            "For example:\n\n"
            "```python\n"
            "K = 1\n"
            "selected = [\"Red\"]\n"
            "candidates = [\"Yellow\", \"Black\", \"White\"]\n"
            "```\n\n"
            "---\n\n"
            "### Expected Output:\n"
            "Return a list of **K items** from the **candidates** list that are **most similar** to the items in the **selected** list. "
            
            "```python\n"
            "increments = [\"Yellow\"]\n"
            "```"
        )

        example_prompt = ChatPromptTemplate.from_messages([
            ("human", "```python\nK = {num_select}\nselected = {selected}\ncandidates = {candidates}\n```"),
            ("ai", "{response}")
        ])
        
        few_shot_prompt_template = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            example_selector=example_selector
        )

        prompt_template = ChatPromptTemplate.from_messages([
            ("human", instruction_prompt),
            ("ai", "Please provide the number of items to select, the selected items, and the candidates list."),
            few_shot_prompt_template,
            ("human", "```python\nK = {num_select}\nselected = {selected}\ncandidates = {candidates}\n```"),
        ])
        
        return prompt_template
    
    @overrides
    def get_output_parser(self) -> BaseOutputParser:
        return AnchoredClusteringOutputParser()