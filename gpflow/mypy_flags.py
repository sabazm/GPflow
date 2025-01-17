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
Code for printing flags for mypy, depending on library versions.
"""
from .type_flags import MYPY_FLAGS


def print_mypy_flags() -> None:
    for flag, value in MYPY_FLAGS.items():
        if value:
            print("--always-true", flag)
        else:
            print("--always-false", flag)


if __name__ == "__main__":
    print_mypy_flags()
