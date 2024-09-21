""" An implementation of the self-reflective interface """


from dataclasses import dataclass
from typing import Union, Text, List, Dict, Optional, Callable, Any
from dataclasses import asdict
from overrides import overrides
from langchain_openai import ChatOpenAI
from tqdm import tqdm
from ..states.base_states import BaseState
from ..interface import Interface


@Interface.register("self-reflective-interface")
class SelfReflectiveInterface(Interface):
    def __init__(
        self,
        generation_interface: Interface,
        feedback_interface: Feedback
    ):