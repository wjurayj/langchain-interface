""" Test the capability to call the API in batch mode. """


import unittest
from typing import Text
from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import BaseOutputParser
from langchain_openai import ChatOpenAI
from langchain_interface.models import BatchedAPIConfig
from langchain_interface.models import ChatOpenAIWithBatchAPI


class JokeParser(BaseOutputParser[str]):
    
    def parse(self, text: Text) -> Text:
        return text.strip()
    
    @property
    def _type(self) -> str:
        return "joke"


class TestBatchAPICalling(unittest.IsolatedAsyncioTestCase):
    """ Test the capability to call the API in batch mode. """
    
    def setUp(self):
        """ """
        # set_llm_cache(SQLiteCache(".cache/.test_cache.db"))
        self._llm = ChatOpenAIWithBatchAPI(
        # self._llm = ChatOpenAI(
            temperature=0,
            model="gpt-4o",
            model_kwargs={"top_p": 0.98},
            max_tokens=None,
            verbose=True,
        )
        
        self._runnable_config = BatchedAPIConfig(
            max_abatch_size=3
        )

        self._prompt_template = ChatPromptTemplate.from_messages(
            ("human", "Write me a joke about {animal}.")
        )

    async def test_batch_api_calling(self):
        """ Test the capability to call the API in batch mode. """
        
        test_cases = [
            {
                "animal": "dogs",
            },
            {
                "animal": "cats",
            },
            {
                "animal": "birds",
            },
            {
                "animal": "fishes",
            },
            {
                "animal": "rabbits",
            }
        ]
        
        calling_chain = self._prompt_template | self._llm | JokeParser()
        responses = await calling_chain.abatch(test_cases, config=self._runnable_config)

        print(responses)

    async def test_result_recovering_to_cache(self):
        """ """

        set_llm_cache(SQLiteCache(".cache/.test_cache.db"))
        
        test_cases = [
            {
                "animal": "dogs",
            },
            {
                "animal": "cats",
            },
            {
                "animal": "birds",
            },
            {
                "animal": "fishes",
            },
            {
                "animal": "rabbits",
            }
        ]
        
        self._llm.cache_results(
            input_files="openai://file-FOzBv8JZVrMj5eftfLVzP8pY",
            output_files="openai://file-eVmTxaXQqXkQAdUpXnoDQdL2",
        )
        
        calling_chain = self._prompt_template | self._llm | JokeParser()
        responses = await calling_chain.abatch(test_cases)
        
        print(responses)