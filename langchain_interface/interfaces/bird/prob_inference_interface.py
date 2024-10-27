""" """

import numpy
from typing import Annotated, List, Union, TypeVar
from typing_extensions import TypedDict
from itertools import product
try:
    import ujson as json
except ImportError:
    import json
from dataclasses import dataclass
from typing import Union, Text, List, Dict, Optional, Callable, Any, Literal, Tuple
from dataclasses import asdict
from overrides import overrides
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.runnables.base import Runnable
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableParallel,
    RunnableLambda
)
from ...states.base_states import BaseState, revise, append, keyupdate
from ...steps.step import Step
from ...steps.bird import (
    BIRDSentenceProposalStep,
    BIRDSummarizeToFactorStep,
    BIRDImplicationCheckStep,
    BIRDReevaluateImplicationStep,
    BIRDSentenceSupportDeterminationStep,
    BIRDVerbalizedProbabilityStep
)
from ..interface import Interface


@dataclass
class Value:
    name: Text


@dataclass
class Factor:
    name: Text
    values: List[Value]


class BIRDInternalState(TypedDict, total=True):
    scenario: str
    condition: str
    outcome_1: str
    outcome_2: str
    sentences_for_outcome_1: list
    sentences_for_outcome_2: list
    responses: Annotated[list, append]
    factors: Annotated[list, append]
    implied_value_check: Annotated[dict, keyupdate]
    direction_value_check: Annotated[dict, keyupdate]
    # verbalized_probability: Annotated[Dict[Text, Tuple[float, float]], keyupdate]
    filtered_factor_names: list
    final_score: Optional[float]
    
    
class BIRDProbInferenceInterface(Interface):
    
    @overrides
    def get_runnable(self, llm: BaseLanguageModel) -> Runnable:
        """ """
        
        json_llm = llm.bind(
            response_format={
                "type": "json_object",
            }
        )
        
        _call_sentence_sampling_o1 = BIRDSentenceProposalStep().induce_stated_callable(
            llm=llm,
            parse_input=lambda state: {
                "scenario": state['scenario'],
                "hypothesis": state['outcome_1'],
            },
            parse_output=lambda response: {
                "sentences_for_outcome_1": response.sentences,
                "responses": response.messages
            }
        )

        _call_sentence_sampling_o2 = BIRDSentenceProposalStep().induce_stated_callable(
            llm=llm,
            parse_input=lambda state: {
                "scenario": state['scenario'],
                "hypothesis": state['outcome_2'],
            },
            parse_output=lambda response: {
                "sentences_for_outcome_2": response.sentences,
                "responses": response.messages
            }
        )
        
        graph_builder = StateGraph(BIRDInternalState)
        graph_builder.add_node("sentence_sampling_o1", _call_sentence_sampling_o1)
        graph_builder.add_node("sentence_sampling_o2", _call_sentence_sampling_o2)
        graph_builder.add_edge(START, "sentence_sampling_o1")
        graph_builder.add_edge(START, "sentence_sampling_o2")
        
        # now we need to summarize the sentences for each outcome

        def _prepare_description(state) -> Text:
            outcome_1 = state['outcome_1']
            outcome_2 = state['outcome_2']
            sentences_1_fomatted = "\n".join([f"#{sidx + 1} {sentence}" for sidx, sentence in enumerate(state['sentences_for_outcome_1'])])
            sentences_2_fomatted = "\n".join([f"#{sidx + 1} {sentence}" for sidx, sentence in enumerate(state['sentences_for_outcome_2'])])
            return f"Outcome 1: {outcome_1}\nSentences:\n{sentences_1_fomatted}\nOutcome 2: {outcome_2}\nSentences:\n{sentences_2_fomatted}"
        
        _call_sentence_summarization = BIRDSummarizeToFactorStep().induce_stated_callable(
            json_llm,
            parse_input=lambda state: {
                "scenario": state['scenario'],
                "description": _prepare_description(state)
            },
            parse_output=lambda response: {
                "factors": [Factor(name=k, values=[Value(name=vv) for vv in v]) for k, v in response.factor_dict.items()],
                "responses": response.messages
            }
        )
        
        graph_builder.add_node("sentence_summarization", _call_sentence_summarization)
        graph_builder.add_edge(
            ["sentence_sampling_o1", "sentence_sampling_o2"],
            "sentence_summarization"
        )
        
        # implication check step
        def _check_all_implied_values(state) -> list:
            """ """
            factors = state['factors']
            
            return [
                Send(
                    "implication_check",
                    {
                        "scenario": state['scenario'],
                        "condition": state['condition'],
                        "statement": value.name
                    }
                ) for factor in factors for value in factor.values
            ]
        
        _call_implication_check = RunnableParallel(
            {
                "passthrough": RunnablePassthrough(),
                "responses": BIRDImplicationCheckStep().chain_llm(llm)
            }) | RunnableLambda(
                lambda output: {
                "implied_value_check": {
                    output['passthrough']['statement']: output['responses'].implied,
                },
                "responses": output['responses'].messages
            })

        graph_builder.add_node("implication_check", _call_implication_check)
        graph_builder.add_conditional_edges(
            "sentence_summarization",
            _check_all_implied_values,
            ["implication_check"]
        )
        
        # add the second-pass filtering
        _call_reevaluate_implication = BIRDReevaluateImplicationStep().induce_stated_callable(
            llm=json_llm,
            parse_input=lambda state: {
                "scenario": state['scenario'],
                "implication_dict": json.dumps({
                    factor.name: [
                        value.name for value in factor.values if state['implied_value_check'][value.name]
                    ] for factor in state['factors']
                })
            },
            parse_output=lambda response, state: {
                "implied_value_check": {
                    value.name: value.name in response.implication_dict[factor.name]
                    for factor in state['factors'] if factor.name in response.implication_dict
                    for value in factor.values if (value.name in state['implied_value_check'] and state['implied_value_check'][value.name])
                },
                "responses": response.messages
            }
        )
        
        graph_builder.add_node("reevaluate_implication", _call_reevaluate_implication)
        graph_builder.add_edge("implication_check", "reevaluate_implication")
        
        # Then, further filter the factors down to only those that support single outcome
        def _check_all_single_side_support(state) -> list:
            """ """
            return [
                Send("single_side_support_check", {
                    "scenario": state['scenario'],
                    "condition": value_name,
                    "outcome_1": state['outcome_1'],
                    "outcome_2": state['outcome_2'],
                }) for value_name in state['implied_value_check']
            ]
            
        _call_single_side_support_check = RunnableParallel(
            {
                "passthrough": RunnablePassthrough(),
                "processed": BIRDSentenceSupportDeterminationStep().chain_llm(llm)
            }) | RunnableLambda(
                lambda output: {
                "direction_value_check": {
                    output['passthrough']['condition']: output['processed'].support_index
                },
                "responses": output['processed'].messages
            })
        
        graph_builder.add_node("single_side_support_check", _call_single_side_support_check)
        graph_builder.add_conditional_edges("reevaluate_implication", _check_all_single_side_support, ["single_side_support_check"])
        
        # finally using all the filtered factors to calculate verbal probability.
        def _filter_factors(state) -> dict:
            return {
                "filtered_factor_names": [
                    factor.name for factor in state['factors'] if len({state['direction_value_check'][value.name] for value in factor.values}) > 1
                ]
            }
            
        graph_builder.add_node("filter_factors", RunnableLambda(_filter_factors))
        graph_builder.add_edge("single_side_support_check", "filter_factors")
        
        # # then finally we evaluate the verbalized probabilities
        # def _calculate_all_verbalized_probabilities(state) -> list:
        #     return [
        #         Send("verbalized_probability", {
        #             "scenario": state['scenario'],
        #             "condition": f"{state['condition']} {value.name}",
        #             "outcome_1": state['outcome_1'],
        #             "outcome_2": state['outcome_2'],
        #         }) for factor in state['factors'] if factor.name in state['filtered_factor_names']
        #         for value in factor.values
        #     ]
            
        # _call_verbalized_probability = RunnableParallel(
        #     {
        #         "passthrough": RunnablePassthrough(),
        #         "processed": BIRDVerbalizedProbabilityStep.chain_llm(llm)
        #     } | RunnableLambda(lambda output: {
        #         "verbalized_probability": {
        #             output['passthrough']['condition']: output['processed'].verbalized_probability
        #         },
        #         "responses": output['processed'].messages
        #     })
        # )

        # graph_builder.add_node("verbalized_probability", _call_verbalized_probability)
        # graph_builder.add_conditional_edges("filter_factors", _calculate_all_verbalized_probabilities, ["verbalized_probability"])

        # now finally we marginalize all factors that are filtered.
        def _marginalize(state) -> dict:
            """ """
            global np
            iterators = [range(len(factor.values)) for factor in state['factors'] if factor.name in state['filtered_factor_names']]

            factor_value_dists = [
                numpy.array([state['implied_value_check'][value.name] for value in factor.values], dtype=numpy.float32) + 1e-6
                for factor in state['factors'] if factor.name in state['filtered_factor_names']
            ]
            
            # normalize the dist over the values
            factor_value_dists = [
                factor_dist / numpy.sum(factor_dist)
                for factor_dist in factor_value_dists
            ]
            
            translate = {
                0: .75,
                1: .25,
                -1: .5
            }
            
            supportiveness = [
                numpy.array([translate[state['direction_value_check'][value.name]] for value in factor.values], dtype=numpy.float32) for factor in state['factors'] if factor.name in state['filtered_factor_names']
            ]
            
            marginalized = 0.

            for indices in product(*iterators):
                # calculate the joint probability
                probs = numpy.array([sp[index] for index, sp in zip(indices, supportiveness)])
                pp = numpy.prod(probs)
                np = numpy.prod(1 - probs)

                po_given_f = pp / (pp + np)
                pf_given_c = numpy.prod(numpy.array([factor_dist[index] for index, factor_dist in zip(indices, factor_value_dists)]))
                
                marginalized += po_given_f * pf_given_c
                
            return {
                "final_score": marginalized
            }
            
        graph_builder.add_node("marginalize", RunnableLambda(_marginalize))
        graph_builder.add_edge("filter_factors", "marginalize")
        graph_builder.add_edge("marginalize", END)
        
        compiled_graph = graph_builder.compile()

        return RunnableLambda(
            lambda input: BIRDInternalState(
                scenario=input['scenario'],
                condition=input['condition'],
                outcome_1=input['outcome_1'],
                outcome_2=input['outcome_2'],
                sentences_for_outcome_1=[],
                sentences_for_outcome_2=[],
                responses=[],
                factors=[],
                implied_value_check={},
                direction_value_check={},
                filtered_factor_names=[],
                final_score=None
            )
        ) | compiled_graph