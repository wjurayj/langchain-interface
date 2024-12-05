""" """

import unittest
from dataclasses import dataclass
from typing import List, Dict, Text, Any
from langchain_interface.instances.instance import Instance


class TestInstanceClassSerialization(unittest.TestCase):
    
    def test_customer_class_serialization(self):
        
        @dataclass(frozen=True, eq=True)
        class CustomizedInstance(Instance):
            field_a: str
            field_b: int
            field_c: float
            
        @dataclass(frozen=True, eq=True)
        class Grouped(Instance):
            ig: List[CustomizedInstance]
            mg: Dict[str, CustomizedInstance]
            tag: Text
            
        instance = Grouped(
            tag="customized",
            ig=[
                CustomizedInstance(field_a=f"a{i}", field_b=i, field_c=1.0)
                for i in range(3)
            ],
            mg={
                f"b{i}": CustomizedInstance(field_a=f"a{i}", field_b=i, field_c=1.0)
                for i in range(3)
            },
        )

        excepted = {
            "ig": [
                {"field_a": f"a{i}", "field_b": i, "field_c": 1.0}
                for i in range(3)
            ],
            "mg": {
                f"b{i}": {"field_a": f"a{i}", "field_b": i, "field_c": 1.0}
                for i in range(3)
            },
            "tag": "customized",
        }
        
        self.assertEqual(instance.to_dict(), excepted)