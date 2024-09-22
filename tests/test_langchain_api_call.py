"""It might be possible that LangChain wrapped
client is not possible to call the API and get results.
"""
from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache
from unittest import TestCase
from langchain_interface.steps import DecompositionStep
from langchain_interface.steps import DecontextualizationStep
from langchain_interface.steps import EvidentialSupportStep


set_llm_cache(SQLiteCache(".cache/.test_cache.db"))


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
        
        
class TestEvidentialSupport(TestCase):
    
    def setUp(self):
        
        self._model_name = "gpt-4o-mini"
        
        self._chat_interface = EvidentialSupportStep(
            model_name=self._model_name,
            max_tokens=512,
        )
        
        self.test_cases = [
            {
                "premise": "Lana Del Rey sings will you still love me when i'm not young and beautiful.",
                "hypothesis": "The song \"Will You Still Love Me When I'm Not Young and Beautiful\" is performed by a well-known contemporary female artist."
            }
        ]
        
    def test_evidential_support_interface(self):

        queries = self.test_cases
        results = self._chat_interface(queries)
        
        for r in results:
            print("-" * 20)
            print(r.messages)
            print(r.label)
        print("=" * 20)