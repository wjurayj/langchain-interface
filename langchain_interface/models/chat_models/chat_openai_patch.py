""" Implement a subclass of ChatOpenAI that uses the batch API with `abatch` calls. """

from langchain_openai.chat_models.base import ChatOpenAI
from ..mixins import BatchedAPIConfigMixin, BatchedAPIMixin
from langchain_core.runnables.config import (
    RunnableConfig,
    ensure_config,
    run_in_executor,
    get_config_list
)


class ChatOpenAIWithBatchAPI(BatchedAPIMixin, ChatOpenAI):
    """ A subclass of ChatOpenAI that uses the batch API with `abatch` calls. """
    pass

class BatchedAPIConfig(BatchedAPIConfigMixin, RunnableConfig):
    """ A subclass of RunnableConfig that includes batch API configuration. """
    pass