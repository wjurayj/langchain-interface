"""Get fewshot example from a dataset.
to support demonstration in prompt.
"""
import abc
from registrable import Registrable
from typing import Dict, List, Text, Any
from langchain_core.example_selectors import BaseExampleSelector


class ExampleSelector(Registrable, BaseExampleSelector):
    def __init__(
        self,
    ):
        super().__init__()
