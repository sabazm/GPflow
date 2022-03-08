# Copyright 2022 The GPflow Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Utilities for testing the `check_shapes` library.
"""
from dataclasses import dataclass
from typing import Optional, Union

from gpflow.base import TensorType
from gpflow.experimental.check_shapes import Dimension, Shape, get_shape
from gpflow.experimental.check_shapes.argument_ref import (
    ArgumentRef,
    AttributeArgumentRef,
    IndexArgumentRef,
    RootArgumentRef,
)
from gpflow.experimental.check_shapes.specs import ParsedDimensionSpec, ParsedShapeSpec


@dataclass(frozen=True)
class TestShaped:

    test_shape: Shape


@get_shape.register(TestShaped)
def get_test_shaped_shape(shaped: TestShaped) -> Shape:
    return shaped.test_shape


def t_unk() -> TestShaped:
    """
    Creates an object with an unknown shape, for testing.
    """
    return TestShaped(None)


def t(*shape: Dimension) -> TensorType:
    """
    Creates an object with the given shape, for testing.
    """
    return TestShaped(shape)


@dataclass(frozen=True)
class varrank:
    name: Optional[str]


def make_shape_spec(*dims: Union[int, str, varrank, None]) -> ParsedShapeSpec:
    shape = []
    for dim in dims:
        if isinstance(dim, int):
            shape.append(ParsedDimensionSpec(constant=dim, variable_name=None, variable_rank=False))
        elif dim is None or isinstance(dim, str):
            shape.append(ParsedDimensionSpec(constant=None, variable_name=dim, variable_rank=False))
        else:
            assert isinstance(dim, varrank)
            shape.append(
                ParsedDimensionSpec(constant=None, variable_name=dim.name, variable_rank=True)
            )
    return ParsedShapeSpec(tuple(shape))


def make_argument_ref(argument_name: str, *refs: Union[int, str]) -> ArgumentRef:
    result: ArgumentRef = RootArgumentRef(argument_name)
    for ref in refs:
        if isinstance(ref, int):
            result = IndexArgumentRef(result, ref)
        else:
            result = AttributeArgumentRef(result, ref)
    return result