""" """

import unittest
from langchain_openai import ChatOpenAI
from langchain_interface.models.mixins import ReasoningContentMixin


class ChatOpenAIWithReasoning(ReasoningContentMixin, ChatOpenAI):
    """ """
    pass


class TestCase(unittest.TestCase):
    """ """
    def setUp(self):
        self._llm = ChatOpenAIWithReasoning(
            temperature=0,
            top_p=1,
            model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            max_tokens=None,
            verbose=True,
            base_url="http://localhost:9871/v1",
            api_key="token-abc123"
        )
        
    def test_reasoning_extraction(self):
        """ """
        messages = [
            (
                "system",
                "You are a helpful assistant that translates English to French. Translate the user sentence.",
            ),
            ("Danielle is the member with the most fluent Enligsh among all members of New Jeans."),
        ]
        print(self._llm.invoke(messages))