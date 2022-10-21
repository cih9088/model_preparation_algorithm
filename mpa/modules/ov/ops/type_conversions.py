# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from dataclasses import dataclass

import torch

from .builder import OPS
from .op import Attribute, Operation


_torch_to_ov = {
    torch.uint8: ["u1", "u4", "u8"],
    torch.int8: ["i4", "i8"],
    torch.int16: ["i16"],
    torch.int32: ["i32"],
    torch.int64: ["i64"],
    torch.float16: ["f16"],
    torch.float32: ["f32"],
    torch.bool: ["boolean"],
}

_ov_to_torch = {
    "u1": torch.uint8,  # no type in torch
    "u4": torch.uint8,  # no type in torch
    "u8": torch.uint8,
    "u32": torch.int32,  # no type in torch
    "u64": torch.int64,  # no type in torch
    "i4": torch.int8,  # no type in torch
    "i8": torch.int8,
    "i16": torch.int16,
    "i32": torch.int32,
    "i64": torch.int64,
    "f16": torch.float16,
    "f32": torch.float32,
    "boolean": torch.bool,
}


@dataclass
class ConvertV0Attribute(Attribute):
    destination_type: str


@OPS.register()
class ConvertV0(Operation[ConvertV0Attribute]):
    TYPE = "Convert"
    VERSION = 0
    ATTRIBUTE_FACTORY = ConvertV0Attribute

    @staticmethod
    def convert_ov_type(ov_type):
        if ov_type not in _ov_to_torch:
            raise NotImplementedError
        return _ov_to_torch[ov_type]

    @staticmethod
    def convert_torch_type(torch_type):
        return _torch_to_ov[torch_type][-1]

    def forward(self, input):
        return input.type(self.convert_ov_type(self.attrs.destination_type))
