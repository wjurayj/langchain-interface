"""It might be possible that LangChain wrapped
client is not possible to call the API and get results.
"""
from unittest import TestCase
import asyncio
from langchain_interface.instances import LLMQueryInstance
from langchain_interface.interfaces.decomposition_interface import DecompositionInterface


class TestLangChainAPICall(TestCase):
    
    def setUp(self):
        
        # self._model_name = "mistralai/Mistral-7B-Instruct-v0.2"
        # self._api_key = "token-abc123"

        self._model_name = "gpt-4o-mini"
        
        self.chat_interface = DecompositionInterface(
            model_name=self._model_name,
            max_tokens=256,
            # api_key=self._api_key
        )

        self.test_cases = [
            "vLLM provides an HTTP server that implements OpenAI’s Completions and Chat API.",
            "U.C.L.A. asked for officers after a clash between pro-Palestinian demonstrators and counterprotesters grew heated overnight, and tensions continued to rise at universities across the country."
        ]
        
    def test_chat_interface(self):
        """Test if the API call is successful."""
        
        queries = [LLMQueryInstance(input=text) for tidx, text in enumerate(self.test_cases)]
        results = self.chat_interface(queries)
        
        print("=" * 20)
        for r in results:
            print("-" * 20)
            for fact in r.claims:
                print(fact)
            print("-" * 20)
        print("=" * 20)