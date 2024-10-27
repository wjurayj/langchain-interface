""" """

import json
import sys
from unittest import TestCase
from langchain_openai import ChatOpenAI
from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache
from langchain_core.runnables import Runnable
from langchain_interface.interfaces.bird.prob_inference_interface import BIRDProbInferenceInterface
from langchain_interface.interfaces.bird.prob_inference_interface import BIRDInternalState


class TestBIRDProbInference(TestCase):
    def setUp(self):
        """ """
        
        set_llm_cache(SQLiteCache(".cache/.test_cache.db"))
        
        self._test_cases = [
            {
                "scenario": "Assessing the likelihood that certain outcome is likely given a specific condition.",
                "condition": "A basset hound is tied to a doorway in an alley in front of a man and woman.",
                "outcome_1": "It is their dog.",
                "outcome_2": "It is not their dog.",
            },
            {
                "scenario": "Assessing the likelihood that certain outcome is likely given a specific condition.",
                "condition": "A man and a woman are standing next to sculptures , talking while another man looks at other sculptures .",
                "outcome_1": "There are sculptures of people .",
                "outcome_2": "There are no sculptures of people .",
            },
            {
                "scenario": "Assessing the likelihood that certain outcome is likely given a specific condition.",
                "condition": "A man with a red shirt is watching another man who is standing on top of a attached cart filled to the top .",
                "outcome_1": "The cart is full of coal.",
                "outcome_2": "The cart is full of other material.",
            }
        ]

        self._llm = ChatOpenAI(
            temperature=0,
            model="gpt-4o",
            model_kwargs={"top_p": 0.98},
            max_tokens=None,
            verbose=True,
        )
        
        self._interface: Runnable = BIRDProbInferenceInterface().get_runnable(self._llm)
        
        
    def test_bird_prob_inference(self):
        test_case = self._test_cases[2]
        state = self._interface.invoke(input=test_case)
        
        print("Factors:")
        json.dump({factor.name: [value.name for value in factor.values] for factor in state['factors']}, sys.stdout, indent=4)
        print("\nFiltered factors:")
        json.dump(state['filtered_factor_names'], sys.stdout, indent=4)
        print("\nImplied value check:") 
        json.dump(state['implied_value_check'], sys.stdout, indent=4)
        print("\nDirection value check:") 
        json.dump(state['direction_value_check'], sys.stdout, indent=4)
        print("\nScore:")
        json.dump(state['final_score'], sys.stdout, indent=4)