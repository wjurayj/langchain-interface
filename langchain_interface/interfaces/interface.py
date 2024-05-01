"""Create scorers that gives factuality judgment scores to a given claim and evidence pair.
"""

import abc
from registrable import Registrable
from typing import Union, List, Dict, Text, Any
from tqdm import tqdm
from ..instances.instance import LLMQueryInstance


class Interface(Registrable):
    """ """

    def __init__(self, llm_chain, runnable_config):
        super().__init__()

        self.llm_chain = llm_chain
        self.runnable_config = runnable_config

    def __call__(
        self, instances: List[LLMQueryInstance], silence: bool = False
    ) -> List[Dict[Text, Any]]:

        if silence:
            instances = [self.input_parser(ins) for ins in instances]

            return [
                {"raw": item, "parsed": self.output_parser(item)}
                for item in self.llm_chain.batch(instances, config=self.runnable_config)
            ]

        instances = [self.input_parser(ins) for ins in instances]

        results = []

        pbar = tqdm(total=len(instances))

        for bidx in range(0, len(instances), self.batch_size):
            batch_size = min(self.batch_size, len(instances) - bidx)
            results.extend(
                [
                    {"raw": item, "parsed": self.output_parser(item)}
                    for item in self.llm_chain.batch(
                        instances[bidx : bidx + batch_size], self.runnable_config
                    )
                ]
            )
            pbar.update(batch_size)

        return results
