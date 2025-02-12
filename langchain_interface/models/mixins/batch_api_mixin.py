""" A mixin that allows for batched API calls (currently only tested against openai-api) """

import asyncio
from typing_extensions import NotRequired, TypedDict
import json
import os
import warnings
import uuid
import tempfile
from langchain_core.load import dumps, dumpd
from langchain_core.caches import BaseCache
from langchain.globals import get_llm_cache
from langchain_core.runnables.utils import Input
from langchain_core.prompt_values import PromptValue
from langchain_core.messages import BaseMessage
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
from typing import (
    TYPE_CHECKING,
    List,
    Union,
    Optional,
    Any,
    Text
)

if TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks
    # from langchain_core.language_models.base import LLMResult
    
    
# class BatchedAPIConfigMixin(RunnableConfig):
class BatchedAPIConfigMixin(TypedDict):
    max_abatch_size: NotRequired[int|None]
    batch_file_dir: NotRequired[str|None]


# class ChatOpenAIWithBatchAPI(ChatOpenAI):
class BatchedAPIMixin:
    
    async def abatch(
        self,
        inputs: List[Input],
        config: Optional[Union[RunnableConfig, List[RunnableConfig]]] = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Optional[Any],
    ):

        batch_file_dir = config[0]['configurable'].get("batch_file_dir", None) if isinstance(config, list) else config['configurable'].get("batch_file_dir", None)
        max_abatch_size = config[0]['configurable'].get("max_abatch_size", None) if isinstance(config, list) else config['configurable'].get("max_abatch_size", None)
        
        # print(f"Batch file dir: {batch_file_dir}")
        # print(f"Max abatch size: {max_abatch_size}")
        # exit()
        
        if not inputs:
            return []
        
        stop = kwargs.get("stop", None)
        
        if isinstance(config, list):
            config = config[0]
        ensure_config(config)
        
        # TODO: Check if further process is needed
        llm_results = await self.agenerate_prompt(
            [self._convert_input(input_) for input_ in inputs],
            stop=stop, 
            callbacks=config.get("callbacks"),
            tags=config.get("tags"),
            metadata=config.get("metadata"),
            run_name=config.get("run_name"),
            run_id=config.get("run_id", None),
            # max_abatch_size=config.get("max_abatch_size", None),
            # batch_file_dir=config.get("batch_file_dir", None),
            max_abatch_size=max_abatch_size,
            batch_file_dir=batch_file_dir,
            **kwargs
        )
        
        return [generations[0].message for generations in llm_results.generations]
        
    async def agenerate_prompt(
        self,
        prompts: list[PromptValue],
        stop: Optional[list[str]] = None,
        callbacks: "Callbacks" = None,
        max_abatch_size: Optional[int] = None,
        batch_file_dir: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """ This is included for clarity reasons """
        
        prompt_messages = [prompt.to_messages() for prompt in prompts]
        return await self.agenerate(
            prompt_messages,
            stop=stop,
            callbacks=callbacks,
            max_abatch_size=max_abatch_size,
            batch_file_dir=batch_file_dir,
            **kwargs
        )
    
    async def agenerate(
        self,
        messages: list[list[BaseMessage]],
        stop: Optional[list[str]] = None,
        callbacks: "Callbacks" = None,
        *,
        tags: Optional[List[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        run_name: Optional[str] = None,
        run_id: Optional[uuid.UUID] = None,
        max_abatch_size: Optional[int] = None,
        batch_file_dir: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """ Generate the response from the model """
        
        params = self._get_invocation_params(stop=stop, **kwargs)
        options = {"stop": stop}
        inheritable_metadata = {
            **(metadata or {}),
            **self._get_ls_params(stop=stop, **kwargs)
        }
        
        # Notice that since callbacks manager is only used for streaming,
        # we'll ignore it for the time being, as it's not our main focus.
        # callback_manager = AsyncCallbackManager.config(
        #     callbacks,
        #     self.callbacks,
        #     self.verbose,
        #     tags,
        #     self.tags,
        #     inheritable_metadata,
        #     self.metadata
        # )
        
        # run_managers = await callback_manager.on_chat_model_start(
        #     self._serialized,
        #     messages,
        #     invocation_params=params,
        #     options=options,
        #     name=run_name,
        #     batch_size=self.batch_size,
        #     run_id=run_id
        # )

        # We'll try prepare payloads in batch before sending them to API
        results = await self._abatch_generate_with_cache(
            messages,
            stop=stop,
            max_abatch_size=max_abatch_size,
            batch_file_dir=batch_file_dir,
            **kwargs
        )
        exceptions = []
        
        for i, res in enumerate(results):
            if isinstance(res, BaseException):
                exceptions.append(res)
                
        if exceptions:
            raise exceptions[0]
        
        flattened_outputs = [
            LLMResult(generations=[res.generations], llm_outputs=res.llm_output)
            for res in results
        ]
        
        llm_output = self._combine_llm_outputs([res.llm_output for res in flattened_outputs])
        generations = [res.generations for res in results]
        output = LLMResult(generations=generations, llm_output=llm_output)
        
        return output
    
    async def _abatch_generate_with_cache(
        self,
        message_batches: list[list[BaseMessage]],
        stop: Optional[list[str]] = None,
        max_abatch_size: Optional[int] = 50_000,
        batch_file_dir: Optional[str] = None,
        **kwargs: Any,
    ) -> List[ChatResult]:
        """ This function at its minimum, should equivalent to
        the list comprehension in `_agenerate_with_cache`
        """
        
        llm_cache = self.cache if isinstance(self.cache, BaseCache) else get_llm_cache()
        cache_vals = []
        
        dumped_prompts = [dumps(messages) for messages in message_batches]
        
        check_cache = self.cache or self.cache is None
        if check_cache:
            if llm_cache:
                llm_string = self._get_llm_string(stop=stop, **kwargs)
                cache_vals = [llm_cache.lookup(dp, llm_string) for dp in dumped_prompts]
            elif self.cache is None:
                cache_vals = [None] * len(message_batches)
            else:
                msg = "Asked to cache, but no cahce found at `langchain.cache`."
                raise ValueError(msg)
            
        # perform cache val operations
        processed = [ChatResult(generations=cache_val) if isinstance(cache_val, list) else None for cache_val in cache_vals]
        need_process_index = [i for i, cache_val in enumerate(cache_vals) if cache_val is None]
        
        filtered_message_batches = [message_batches[i] for i in need_process_index]

        if filtered_message_batches:
            # TODO: use multiple threads to generate the payloads
            
            # Also, pointless to obey should_stream
            payloads = [self._get_request_payload(messages, stop=stop, **kwargs) for messages in filtered_message_batches]
            batch_response_dict = {}
            current_payloads = [(payloads[i], i) for i in range(len(payloads)) if f"request-{i}" not in batch_response_dict][:max_abatch_size]
            generation_info = None

            assert self.include_response_headers is False, "Response headers are not supported in batch mode."
            
            # generate with the payloads

            while current_payloads:
                
                with (
                    tempfile.NamedTemporaryFile(mode="w", delete=False) if batch_file_dir is None
                    else open(os.path.join(batch_file_dir, f"{uuid.uuid4()}.jsonl"), "w")
                ) as file_:
                    batch_file_name = file_.name

                    for payload, pidx in current_payloads:
                        in_batch_id = f"request-{pidx}"
                        file_.write(
                            dumps(
                                {
                                    "custom_id": in_batch_id,
                                    "method": "POST",
                                    "url": "/v1/chat/completions",
                                    "body": payload
                                }
                            ) + "\n"
                        )
                        
                batch_input_file = self.root_client.files.create(
                    file=open(batch_file_name, "rb"),
                    purpose="batch"
                )

                batch_input_file_id = batch_input_file.id

                batch_obj = self.root_client.batches.create(
                    input_file_id=batch_input_file_id,
                    endpoint="/v1/chat/completions",
                    completion_window="24h",
                    metadata={
                        "description": "nightly eval job"
                    }
                )
                
                batch_request_id = batch_obj.id
                batch_output_file_id = None

                # manual remove of tempfile
                # even if the removal failed, we'll ignore it
                
                if batch_file_dir is None:
                    try:
                        os.remove(batch_file_name)
                    except Exception as e:
                        warnings.warn(f"Failed to remove batch file: {e}")
                
                sleeping_window = 60
                
                while True:
                    batch_obj = self.root_client.batches.retrieve(batch_request_id)
                    if batch_obj.status in [
                        "validating",
                        "finalizing",
                        "in_progress",
                    ]:
                        await asyncio.sleep(sleeping_window)
                        sleeping_window = min(sleeping_window * 2, 300)
                    elif batch_obj.status == "completed":
                        batch_output_file_id = batch_obj.output_file_id
                        break
                    else:
                        raise ValueError(f"Batch request failed with status: {batch_obj.status}")
                    
                # retrieve the results using the output file id
                batch_responses: Text = self.root_client.files.content(batch_output_file_id).text
                
                if batch_file_dir is not None:
                    # store the batch responses in the batch file dir
                    batch_output_filename = self.root_client.files.retrieve(batch_output_file_id).filename
                    with open(os.path.join(batch_file_dir, batch_output_filename), "w") as file_:
                        file_.write(batch_responses)

                batch_responses = [json.loads(br) for br in batch_responses.split("\n") if br.strip()]
                
                batch_response_dict.update({
                    br["custom_id"]: br['response']['body'] for br in batch_responses
                })

                # update the current payloads batch response_dict
                current_payloads = [(payloads[i], i) for i in range(len(payloads)) if f"request-{i}" not in batch_response_dict][:max_abatch_size]
            
            # index sort into the order of the original messages
            sorted_batch_responses = [batch_response_dict[f"request-{i}"] for i in range(len(payloads))]
            assert len(sorted_batch_responses) == len(filtered_message_batches), "Batch responses and filtered messages are not the same length."

            new_results = await asyncio.gather(*(
                run_in_executor(
                    None,
                    self._create_chat_result,
                    response,
                    generation_info
                ) for response in sorted_batch_responses
            ))
            
            for nr, npindex in zip(new_results, need_process_index):
                processed[npindex] = nr
                
            for r in processed:
                if len(r.generations) == 1:
                    r.generations[0].message.repsonse_metadata = {
                        **(r.llm_output if r.llm_output else {}),
                        **r.generations[0].message.response_metadata,
                    }

            if check_cache and llm_cache:
                # synchronously update the cache
                new_dumped_prompts = [dumped_prompts[i] for i in need_process_index]
                for dp, r in zip(new_dumped_prompts, new_results):
                    llm_cache.update(prompt=dp, llm_string=llm_string, return_val=r.generations)
                
        return processed
    
    def cache_results(
        self,
        input_files: Union[Text, List[Text]],
        output_files: Union[Text, List[Text]],
        stop: Optional[List[str]] = None,
    ):
        """ """
        
        if isinstance(input_files, str):
            input_files = [input_files]
            
        if isinstance(output_files, str):
            output_files = [output_files]

        assert len(input_files) == len(output_files), "Same number of input and output files are required."

        for input_file, output_file in zip(input_files, output_files):
            
            if input_file.startswith("openai://"):
                input_file_content = self.root_client.files.content(input_file[9:]).text
            else:
                with open(input_file, "r") as file_:
                    input_file_content = file_.read()

            if output_file.startswith("openai://"):
                output_file_content = self.root_client.files.content(output_file[9:]).text
            else:
                with open(output_file, "r") as file_:
                    output_file_content = file_.read()

            requests = [json.loads(br) for br in input_file_content.split("\n") if br.strip()]
            responses = [json.loads(br) for br in output_file_content.split("\n") if br.strip()]

            # in post-processing we allow requests and responses to be different lengths

            batch_response_dict = {
                br["custom_id"]: br['response']['body'] for br in responses
            }
            
            sorted_batch_responses = [batch_response_dict[request['custom_id']] for request in requests if request['custom_id'] in batch_response_dict]
            sorted_batch_requests = [request['body'] for request in requests if request['custom_id'] in batch_response_dict]
            
            new_results = [
                self._create_chat_result(response, None)
                for response in sorted_batch_responses
            ]

            for r in new_results:
                if len(r.generations) == 1:
                    r.generations[0].message.repsonse_metadata = {
                        **(r.llm_output if r.llm_output else {}),
                        **r.generations[0].message.response_metadata,
                    }
                    
            llm_cache = self.cache if isinstance(self.cache, BaseCache) else get_llm_cache()
            check_cache = self.cache or self.cache is None
            # if self.cache and isinstance(self.cache, BaseCache):
            if check_cache and llm_cache:
                # new_dumped_prompts = [dumped_prompts[i] for i in need_process_index]
                llm_string = self._get_llm_string(stop=stop)
                new_dumped_prompts = [dumps([_convert_dict_to_message(msg) for msg in request['messages']]) for request in sorted_batch_requests]
                for dp, r in zip(new_dumped_prompts, new_results):
                    llm_cache.update(prompt=dp, llm_string=llm_string, return_val=r.generations)
