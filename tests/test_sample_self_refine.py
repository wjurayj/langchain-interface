""" We run self-refine on an devised example to see if the model can provide the correct output. """

from dataclasses import dataclass
import json
import unittest
from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache
import re
from typing import Text, Dict, Any, Optional, List
from langchain_interface.steps import Step
from langchain_openai import ChatOpenAI
from langchain_core.runnables.config import RunnableConfig
from langchain_core.runnables.base import Runnable
from langchain.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)
from langchain_core.output_parsers import BaseOutputParser

# TODO: use customer downloaded examples for example selector
from langchain_interface.example_selectors import ConstantExampleSelector
from langchain_interface.instances.instance import LLMResponse
from langchain_interface.steps.evidential_support_step import EvidentialSupportStep, EvidentialSupportResponse
    
    
class TestRefineSeparation(unittest.TestCase):
    def setUp(self):
        """ """
        
        self._llm = ChatOpenAI(
            temperature=0,
            top_p=1,
            model="gpt-4o",
            max_tokens=None,
            verbose=True,
        )
        
        self.test_cases = [
            {
                "question": "Who first discovered calculus?",
                "positive": (
                    "- Calculus was discovered by Isaac Newton.\n"
                    "- Calculus was discovered by Pierre-Simon Laplace."
                ),
                "negative": (
                    "- Calculus was discovered by Gottfried Wilhelm Leibniz.\n"
                    "- Calculus was discovered by Albert Einstein.\n"
                    "- Calculus was discovered by Galileo Galilei.\n"
                    "- Calculus was discovered by Johannes Kepler."
                ),
                "analysis": (
                    "Isacc Newton is from England and Pierre-Simon Laplace is from France. "
                    "Gottfried Wilhelm Leibniz is from Germany. "
                    "Albert Einstein is from Germany. "
                    "Galileo Galilei is from Italy. "
                    "Johannes Kepler is from Germany."
                )
            },
            {
                "question": "Which movie does the song \"Tujhe Dekha Toh Yeh Jana Sanam\" come from?",
                "positive": (
                    "- The movie is Aashiqui.\n"
                    "- The movie is Kuch Kuch Hota Hai."
                ),
                "negative": (
                    "- The movie is Dilwale Dulhania Le Jayenge.\n"
                    "- \"Khamoshi: The Musical\" is the movie name.\n"
                    "- The movie is \"Tujhe Dekha To Yeh Jaana Sanam\" (1996).\n"
                    "- The movie is \"Kabhi Khushi Kabhie Gham\"."
                ),
            },
            {
                "question": "Which is the best poem by William Wordsworth?",
                "positive": (
                    "- The best poem by William Wordsworth is \"Daffodils\".\n"
                    "- The best poem by William Wordsworth is \"The Prelude\".\n"
                    "- The best poem by William Wordsworth is \"Lines Composed a Few Miles Above Tintern Abbey\"."
                ),
                "negative": (
                    "- The best poem by William Wordsworth is \"Ode: Intimations of Immortality\".\n"
                    "- The best poem by William Wordsworth is \"The Excursion\"."
                )
            },
            # {
            #     "question": "Who's the best football player of all time?",
            #     "positive": (
            #         "- The best football player of all time is Pelé.\n"
            #         "- The best football player of all time is Diego Maradona.\n"
            #         "- The best football player of all time is Lionel Messi."
            #     ),
            #     "negative": (
            #         "- The best football player of all time is Cristiano Ronaldo.\n"
            #         "- The best football player of all time is Zinedine Zidane.\n"
            #         "- The best football player of all time is Johan Cruyff."
            #     )
            # },
            {
                "question": "Who invented pocket watch?",
                "positive": (
                    "- The pocket watch was invented by Peter Henlein.\n"
                    "- The pocket watch was invented by Abraham-Louis Breguet."
                ),
                "negative": (
                    "- The pocket watch was invented by John Harrison.\n"
                    "- The pocket watch was invented by George Graham.\n"
                    "- The pocket watch was invented by Thomas Mudge.\n"
                    "- The pocket watch was invented by John Arnold."
                )
            }
        ]
        set_llm_cache(SQLiteCache(".cache/.test_cache.db"))

    def test_contrastively_summarize(self):
        """ """
        
        summary_step: Step = Step.from_params({"type": "contrastively-summarize"})
        chained = summary_step.chain_llm(self._llm)
        results = chained.batch(list(map(lambda x: {"positive": x["positive"], "negative": x["negative"]}, self.test_cases)))

        for r, inp in zip(results, self.test_cases):
            print('-' * 50)
            print(f"positive: \n{inp['positive']}\n")
            print(f"negative: \n{inp['negative']}\n")
            print(r.contrasting_summary)
            print('-' * 50)
            
    def test_self_refine(self):
        """ """
        
        pass