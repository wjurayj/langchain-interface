"""Prompt-based scorer for the model.
"""

from dataclasses import dataclass
from typing import Union, Text, List, Dict, Optional, Callable, Any
from dataclasses import asdict
from overrides import overrides
from langchain_openai import ChatOpenAI
from tqdm import tqdm
from ..states.base_states import BaseState

# from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_core.runnables.config import RunnableConfig
from langchain.prompts import (
    ChatPromptTemplate,
    ChatMessagePromptTemplate,
    FewShotChatMessagePromptTemplate,
)
from langchain_core.output_parsers import BaseOutputParser
from langgraph.graph import StateGraph, START, END

# TODO: use customer downloaded examples for example selector
from ..example_selectors import ConstantExampleSelector
from .interface import Interface
from ..instances.instance import LLMQueryInstance, LLMResponse


@dataclass(frozen=True, eq=True)
class DecompositionResponse(LLMResponse):
    claims: List[Text]

class DecompositionOutputParser(BaseOutputParser[Dict]):
    """Parse the output of the decomposition model.
    """
    def parse(self, text: Text) -> Dict:
        cleaned_text = text.strip()
        items = cleaned_text.split("\n")
        return {"responses": DecompositionResponse(messages=text, claims=[item.replace('- ', "") for item in items])}
    
    @property
    def _type(self) -> str:
        return "decomposition_output_parser"
    

@Interface.register("decomposition-interface")
class DecompositionInterface(Interface):
    """Break sentence into independent facts.
    """
    def __init__(
        self,
        model_name: Text,
        max_tokens: Optional[int] = -1,
        temperature: float = 0,
        top_p: float = 1,
        base_url: Optional[Text] = None,
        api_key: Optional[Text] = None,
        model_kwargs: Dict[Text, Any] = {},
        max_concurrency: int = 4,
    ):
        """Instruction prompt will always begin with the
        human prompt message, and alternate until the end.
        """
        self.model_name = model_name
        self.base_url = base_url
        self.temperature = temperature
        self.top_p = top_p

        self.model_kwargs = model_kwargs
        self._additional_params = {}

        if api_key is not None:
            self._additional_params["api_key"] = api_key

        self.llm = ChatOpenAI(
            temperature=temperature,
            top_p=top_p,
            model=model_name,
            model_kwargs=self.model_kwargs,
            max_tokens=max_tokens,
            verbose=True,
            base_url=self.base_url,
            **self._additional_params
        )
        runnable_config = RunnableConfig(
            max_concurrency=max_concurrency,
        )

        example_selector = ConstantExampleSelector()

        examples = [
            {
                "input": "He made his acting debut in the film The Moon is the Sun’s Dream (1992), and continued to appear in small and supporting roles throughout the 1990s.",
                "output": "- He made his acting debut in the film.\n"
                    "- He made his acting debut in The Moon is the Sun’s Dream.\n"
                    "- The Moon is the Sun’s Dream is a film.\n"
                    "- The Moon is the Sun’s Dream was released in 1992.\n"
                    "- After his acting debut, he appeared in small and supporting roles.\n"
                    "- After his acting debut, he appeared in small and supporting roles throughout the 1990s."
            },
            {
                "input": "He is also a successful producer and engineer, having worked with a wide variety of artists, including Willie Nelson, Tim McGraw, and Taylor Swift.",
                "output": "- He is successful.\n"
                    "- He is a producer.\n"
                    "- He is a engineer.\n"
                    "- He has worked with a wide variety of artists.\n"
                    "- Willie Nelson is an artist.\n"
                    "- He has worked with Willie Nelson.\n"
                    "- Tim McGraw is an artist.\n"
                    "- He has worked with Tim McGraw.\n"
                    "- Taylor Swift is an artist.\n"
                    "- He has worked with Taylor Swift."
            },
            {
                "input": "In 1963, Collins became one of the third group of astronauts selected by NASA and he served as the back-up Command Module Pilot for the Gemini 7 mission.",
                "output": "- Collins became an astronaut.\n"
                    "- Collins became one of the third group of astronauts.\n"
                    "- Collins became one of the third group of astronauts selected.\n"
                    "- Collins became one of the third group of astronauts selected by NASA.\n"
                    "- Collins became one of the third group of astronauts selected by NASA in 1963.\n"
                    "- He served as the Command Module Pilot.\n"
                    "- He served as the back-up Command Module Pilot.\n"
                    "- He served as the Command Module Pilot for the Gemini 7 mission."
            },
            {
                "input": "In addition to his acting roles, Bateman has written and directed two short films and is currently in development on his feature debut.",
                "output": "- Bateman has acting roles.\n"
                    "- Bateman has written two short films.\n"
                    "- Bateman has directed two short films.\n"
                    "- Bateman has written and directed two short films.\n"
                    "- Bateman is currently in development on his feature debut."
            },
            {
                "input": "Michael Collins (born October 31, 1930) is a retired American astronaut and test pilot who was the Command Module Pilot for the Apollo 11 mission in 1969.",
                "output": "- Michael Collins was born on October 31, 1930.\n"
                    "- Michael Collins is retired.\n"
                    "- Michael Collins is an American.\n"
                    "- Michael Collins was an astronaut.\n"
                    "- Michael Collins was a test pilot.\n"
                    "- Michael Collins was the Command Module Pilot.\n"
                    "- Michael Collins was the Command Module Pilot for the Apollo 11 mission.\n"
                    "- Michael Collins was the Command Module Pilot for the Apollo 11 mission in 1969."
            },
            {
                "input": "He was an American composer, conductor, and musical director.",
                "output": "- He was an American.\n"
                    "- He was a composer.\n"
                    "- He was a conductor.\n"
                    "- He was a musical director."
            },
            {
                "input": "She currently stars in the romantic comedy series, Love and Destiny, which premiered in 2019.",
                "output": "- She currently stars in Love and Destiny.\n"
                    "- Love and Destiny is a romantic comedy series.\n"
                    "- Love and Destiny premiered in 2019."
            },
            {
                 "input": "During his professional career, McCoy played for the Broncos, the San Diego Chargers, the Minnesota Vikings, and the Jacksonville Jaguars.",
                    "output": "- McCoy played for the Broncos.\n"
                        "- McCoy played for the Broncos during his professional career.\n"
                        "- McCoy played for the San Diego Chargers.\n"
                        "- McCoy played for the San Diego Chargers during his professional career.\n"
                        "- McCoy played for the Minnesota Vikings.\n"
                        "- McCoy played for the Minnesota Vikings during his professional career.\n"
                        "- McCoy played for the Jacksonville Jaguars.\n"
                        "- McCoy played for the Jacksonville Jaguars during his professional career."
            }
        ]

        for example in examples:
            example_selector.add_example(example)

        def create_chain():
            """Create a callable chain for the model."""

            input_example_prompt = "Please breakdown the following sentence into independent facts: {input}"
            example_prompt = ChatPromptTemplate.from_messages([
                ("human", input_example_prompt),
                ("ai", "{output}"),
            ])

            fewshot_prompt_template = FewShotChatMessagePromptTemplate(
                example_prompt=example_prompt,
                example_selector=example_selector,
            )

            prompt_template = ChatPromptTemplate.from_messages(
                [
                    fewshot_prompt_template,
                    ("human", input_example_prompt),
                ]
            )

            builtin_parser = DecompositionOutputParser()

            return prompt_template | self.llm | builtin_parser

        llm_chain = create_chain()
        
        def _call_chain(state) -> Dict[Text, DecompositionResponse]:
            """Call the chain and return the output.
            """
            return llm_chain.invoke(state["responses"][-1])
        
        graph_builder = StateGraph(BaseState)
        graph_builder.add_node("decomposer", _call_chain)
        graph_builder.add_edge(START, "decomposer")
        graph_builder.add_edge("decomposer", END)
        
        graph = graph_builder.compile()
        
        super().__init__(lang_graph=graph, runnable_config=runnable_config)