"""It might be possible that LangChain wrapped
client is not possible to call the API and get results.
"""
from unittest import TestCase
import asyncio
from langchain_interface.instances import LLMQueryInstance
from langchain_interface.interfaces.chat_interface import ChatInterface
from langchain_interface.interfaces.completion_interface import CompletionInterface


class TestLangChainAPICall(TestCase):
    
    def setUp(self):
        
        self._model_name = "mistralai/Mistral-7B-Instruct-v0.2"
        self._api_key = "token-abc123"
        
        self.chat_interface = ChatInterface(
            base_url="http://localhost:9871/v1",
            model_name=self._model_name,
            batch_size=1,
            max_tokens=256,
            instruction_prompt=[],
            system_message=None,
            input_example_prompt="Please decompose the following text into atomic facts: {text}",
            input_parser=lambda x: {"text": x.input},
            output_parser=lambda x: x.strip().split("\n"),
            api_key=self._api_key
        )

        self.completion_interface = CompletionInterface(
            base_url="http://localhost:9871/v1",
            model_name=self._model_name,
            batch_size=1,
            max_tokens=256,
            instruction_prompt="Please decompose the following text into atomic facts.",
            input_example_prompt="Text: {text}",
            input_parser=lambda x: {"text": x.input},
            output_parser=lambda x: x.strip().split("\n"),
            api_key=self._api_key,
        )
        
        self.test_cases = [
            "vLLM provides an HTTP server that implements OpenAI’s Completions and Chat API.",
            "U.C.L.A. asked for officers after a clash between pro-Palestinian demonstrators and counterprotesters grew heated overnight, and tensions continued to rise at universities across the country."
        ]
        
    def test_chat_interface(self):
        """Test if the API call is successful."""
        
        queries = [LLMQueryInstance(id=tidx, input=text) for tidx, text in enumerate(self.test_cases)]
        results = self.chat_interface(queries, silence=True)
        
        print("=" * 20)
        for r in results:
            print("-" * 20)
            for fact in r['parsed']:
                print(fact)
            print("-" * 20)
        print("=" * 20)
    
    def test_async_chat_interface(self):            
        """
        """
        async def test_func():
            queries = [LLMQueryInstance(id=tidx, input=text) for tidx, text in enumerate(self.test_cases)]
            print("=" * 20)
            print("Async calls:")
            async for r in self.chat_interface.async_call(queries):
                print("-" * 20)
                for fact in r['parsed']:
                    print(fact)
                print("-" * 20)
            print("=" * 20)
        
        asyncio.run(test_func())
            
    def test_completion_interface(self):
        """Although the completion interface is no longer used, we want to make sure that it works for backward compatibility."""

        queries = [LLMQueryInstance(id=tidx, input=text) for tidx, text in enumerate(self.test_cases)]
        results = self.completion_interface(queries, silence=True)
        
        print("=" * 20)
        for r in results:
            print("-" * 20)
            for fact in r['parsed']:
                print(fact)
            print("-" * 20)
        print("=" * 20)