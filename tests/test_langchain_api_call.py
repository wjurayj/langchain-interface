"""It might be possible that LangChain wrapped
client is not possible to call the API and get results.
"""
from unittest import TestCase
from langchain_interface.steps import DecompositionStep
from langchain_interface.steps import DecontextualizationStep


class TestDecomposition(TestCase):
    
    def setUp(self):
        
        self._model_name = "gpt-4o-mini"
        
        self._chat_interface = DecompositionStep(
            model_name=self._model_name,
            max_tokens=256,
            # api_key=self._api_key
        )

        self.test_cases = [
            "vLLM provides an HTTP server that implements OpenAI’s Completions and Chat API.",
            "U.C.L.A. asked for officers after a clash between pro-Palestinian demonstrators and counterprotesters grew heated overnight, and tensions continued to rise at universities across the country."
        ]
        
    def test_decomposition_interface(self):
        """Test if the API call is successful."""
        
        queries = [{"input": text} for tidx, text in enumerate(self.test_cases)]
        results = self._chat_interface(queries)
        
        print("=" * 20)
        for r in results:
            print(r)
            print("-" * 20)
            for fact in r.claims:
                print(fact)
            print("-" * 20)
        print("=" * 20)
        
        
class TesetDecontextualization(TestCase):
    
    def setUp(self):
        
        self._model_name = "gpt-4o-mini"
        
        self._chat_interface = DecontextualizationStep(
            model_name=self._model_name,
            max_tokens=512,
        )
        
        self.test_cases = [
            {"input": "It's a great tool for developers.", "context": "vLLM provides an HTTP server that implements OpenAI’s Completions and Chat API. It's a great tool for developers."},
            {"input": "That girl is a genius.", "context": "U.C.L.A. asked for officers after a clash between pro-Palestinian girl and counterprotesters grew heated overnight, and tensions continued to rise at universities across the country."}
        ]
        
    def test_decontextualization_interface(self):

        queries = self.test_cases
        results = self._chat_interface(queries)
        
        for r in results:
            print("-" * 20)
            print(r.revised)
        print("=" * 20)