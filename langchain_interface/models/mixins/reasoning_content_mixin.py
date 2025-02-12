""" Allow reasoning_content field from some of the models to be parsed. """

import json
import openai
import os
import warnings
import uuid
import tempfile
from langchain_core.load import dumps, dumpd
from langchain_core.caches import BaseCache
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain.globals import get_llm_cache
from langchain_core.runnables.utils import Input
from langchain_core.prompt_values import PromptValue
from langchain_core.messages import (
    BaseMessage,
    BaseMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    AIMessage,
    AIMessageChunk,
    SystemMessage,
    SystemMessageChunk,
    FunctionMessage,
    FunctionMessageChunk,
    ToolMessage,
    ToolMessageChunk,
    ChatMessage,
    ChatMessageChunk
)
from langchain_core.runnables.config import (
    RunnableConfig,
    ensure_config,
    run_in_executor,
    get_config_list
)
from dataclasses import dataclass
from langchain_core.outputs import ChatResult
from langchain_core.outputs import LLMResult
from langchain_openai.chat_models.base import _convert_dict_to_message
from langchain_core.messages.tool import tool_call_chunk
from typing import (
    TYPE_CHECKING,
    Type,
    Dict,
    Mapping,
    List,
    Union,
    Optional,
    Any,
    Text,
    cast
)
if TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks


class ReasoningContentMixin:
    """ At the moment it seems that this is only usable with """

    def _create_chat_result(
        self,
        response: Union[dict, openai.BaseModel],
        generation_info: Optional[Dict] = None,
    ) -> ChatResult:
        rtn = super()._create_chat_result(response, generation_info)

        if not isinstance(response, openai.BaseModel):
            return rtn

        if hasattr(response.choices[0].message, "reasoning_content"):  # type: ignore
            rtn.generations[0].message.additional_kwargs["reasoning_content"] = (
                response.choices[0].message.reasoning_content  # type: ignore
            )

        return rtn

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict,
        default_chunk_class: Type,
        base_generation_info: Optional[Dict],
    ) -> Optional[ChatGenerationChunk]:
        generation_chunk = super()._convert_chunk_to_generation_chunk(
            chunk,
            default_chunk_class,
            base_generation_info,
        )
        if (choices := chunk.get("choices")) and generation_chunk:
            top = choices[0]
            if reasoning_content := top.get("delta", {}).get("reasoning_content"):
                if isinstance(generation_chunk.message, AIMessageChunk):
                    generation_chunk.message.additional_kwargs["reasoning_content"] = (
                        reasoning_content
                    )
        return generation_chunk