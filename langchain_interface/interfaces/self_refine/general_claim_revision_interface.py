""" An implementation of the self-reflective interface """

from enum import Enum
from typing import Annotated, List, Union, TypeVar
import abc
from dataclasses import dataclass
from typing import Union, Text, List, Dict, Optional, Callable, Any, Literal
from dataclasses import asdict
from overrides import overrides
from langgraph.graph import StateGraph, START, END
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.runnables.base import Runnable
from langchain_core.runnables import RunnableLambda
from ...states.base_states import WithInputState, WithTagState, revise
from ...steps.step import Step
from ..interface import Interface


class GeneralClaimRevisionState(WithTagState, WithInputState):
    """ """
    summary: Annotated[Text, revise]
    
    
class GeneralClaimRevisionTag(Enum):
    GENERATION = 1
    FEEDBACK = 2
    SUMMARY = 3
    REFINE = 4
    
    
class GeneralClaimRevisionInterface(Interface):
    """ """
    
    def __init__(self):
        super().__init__()

    @overrides
    def get_runnable(self, llm: BaseLanguageModel) -> Runnable:
        """ """
        
        def _create_iteration_from_state(state: GeneralClaimRevisionState) -> int:
            """ """

            iteration_instances = []

            iteration_template = "**R{idx} Generaion Claim**: {claim} \n\n **R{idx} Feedback**:\n{feedback}"
            
            for ridx, (response, tag) in enumerate(zip(state["responses"], state["tags"])):
                if tag == GeneralClaimRevisionTag.GENERATION \
                    or tag == GeneralClaimRevisionTag.REFINE:
                    
                    feedback_response, feedback_tag = state["responses"][ridx + 1], state["tags"][ridx + 1]
                    assert feedback_tag == GeneralClaimRevisionTag.FEEDBACK, f"Expected feedback tag, but got {feedback_tag}"
                    iteration_instances.append(
                        iteration_template.format(
                            idx=len(iteration_instances) + 1,
                            claim=response.general_response,
                            feedback=feedback_response.messages.strip(),
                        )
                    )
                    
                else:
                    continue
                
            return "\n\n".join(iteration_instances)

        _call_summary_step = Step.from_params({"type": "contrastively-summarize"}).induce_stated_callable(
            llm=llm,
            parse_input=lambda x: {"positive": x["inputs"]["positive"], "negative": x["inputs"]["negative"]},
            parse_output=lambda x: {"summary": x.contrasting_summary, "tags": GeneralClaimRevisionTag.SUMMARY, "responses": x.responses},
        )
        _call_claim_split_step = Step.from_params({"type": "claim-set-split"}).induce_stated_callable(
            llm=llm,
            parse_input=lambda x: {"positive": x["inputs"]["positive"], "negative": x["inputs"]["negative"], "analysis": x.summary},
            parse_output=lambda x: {"tags": GeneralClaimRevisionTag.GENERATION, "responses": x.responses},
        )
        _call_feedback_step: Step = Step.from_params({"type": "general-claim-feedback"}).induce_stated_callable(
            llm=llm,
            parse_input=lambda x: {"general_claim": x["responses"][-1].general_response, "positive": x["inputs"]["positive"], "negative": x["inputs"]["negative"]},
            parse_output=lambda x: {"tags": GeneralClaimRevisionTag.FEEDBACK, "responses": x.responses},
        )
        _call_refine_step = Step.from_params({"type": "refine-claim-set-split"}).induce_stated_callable(
            llm=llm,
            parse_input=lambda x: {"positive": x["inputs"]["positive"], "negative": x["inputs"]["negative"], "analysis": x.summary, "iterations": _create_iteration_from_state(x)},
            parse_output=lambda x: {"tags": GeneralClaimRevisionTag.REFINE, "responses": x.responses},
        )
        
        def _should_refine(state: GeneralClaimRevisionState) -> Literal["refine", END]:  # type: ignore
            """ """
            
            assert state["tags"][-1] == GeneralClaimRevisionTag.FEEDBACK, f"Expected feedback tag, but got {state['tags'][-1]}"
            return "refine" if state["responses"][-1].need_further_refinement else END
        
        graph_builder = StateGraph(GeneralClaimRevisionInterface)

        # construct graph interface
        graph_builder.add_node("summarize", _call_summary_step)
        graph_builder.add_node("generation", _call_claim_split_step)
        graph_builder.add_node("feedback", _call_feedback_step)
        graph_builder.add_node("refine", _call_refine_step)

        graph_builder.add_edge(START, "summarize")
        graph_builder.add_edge("summarize", "generation")
        graph_builder.add_edge("generation", "feedback")
        graph_builder.add_conditional_edges("feedback", _should_refine)
        
        # compille graph
        graph = graph_builder.compile()

        def _callable_func(inputs: Dict[Text, Any]) -> Dict[Text, Any]:
            """ """
            
            state = GeneralClaimRevisionState(
                responses=[],
                tags=[],
                inputs=inputs,
                summary="",
            )
            graph.invoke(input=inputs)
            
            return {
                "general_claim": state["responses"][-2].general_response,
                "response": state["responses"][-1],
            }
            
        return RunnableLambda(_callable_func)