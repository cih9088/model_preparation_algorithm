# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Tuple, Optional

import torch

from .builder import OPS
from .op import Attribute, Operation
from .type_conversions import ConvertV0
from .utils import get_dynamic_shape


@dataclass
class ParameterV0Attribute(Attribute):
    element_type: Optional[str] = field(default=None)

    layout: Optional[Tuple[str]] = field(default=None)
    permute: Optional[Tuple[int]] = field(default=None)

    def __post_init__(self):
        super().__post_init__()
        # fmt: off
        valid_element_type = [
            None,
            "u1", "u4", "u8", "u16", "u32", "u64",
            "i4", "i8", "i16", "i32", "i64", "f16", "f32", "boolean", "bf16"
        ]
        # fmt: on
        if self.element_type not in valid_element_type:
            raise ValueError(
                f"Invalid element_type {self.element_type}. "
                f"It must be one of {valid_element_type}."
            )


@OPS.register()
class ParameterV0(Operation):
    TYPE = "Parameter"
    VERSION = 0
    ATTRIBUTE_FACTORY = ParameterV0Attribute

    def forward(self, input):
        # TODO: validate shape
        # need to handle new generated op from reshaped model
        ov_shape = self.shape[0]
        torch_shape = list(input.shape)
        for ov_shape_, torch_shape_ in zip(ov_shape, torch_shape):
            if ov_shape_ == -1:
                continue
            assert (
                ov_shape_ == torch_shape_
            ), f"input shape {torch_shape} does not match with ov shape {ov_shape}"

        if self.attrs.permute:
            input = input.permute(self.attrs.permute)

        return input

    @classmethod
    def from_ov(cls, ov_op):
        op_type = ov_op.get_type_name()
        op_version = ov_op.get_version()
        op_name = ov_op.get_friendly_name().replace(".", "_")
        assert cls.TYPE != "" and cls.VERSION >= 0
        assert op_type == cls.TYPE
        assert op_version == cls.VERSION

        attrs = ov_op.get_attributes()
        if "shape" not in attrs:
            shapes = []
            for output in ov_op.outputs():
                shapes.append(get_dynamic_shape(output))
            shapes = tuple(tuple(shape) for shape in shapes)
            attrs["shape"] = shapes

        layout = ov_op.get_layout()
        if not layout.empty:
            layout = layout.to_string()[1:-1].split(",")
            attrs["layout"] = tuple(layout)

            #  N, C, H, W
            input_layout = OrderedDict({
                "N": 0,
                "C": 1,
                "H": 2,
                "W": 3,
            })
            if not set(layout).symmetric_difference(input_layout.keys()):
                permute = []
                for layout_ in layout:
                    #  N, H, W, C
                    permute.append(input_layout[layout_])
                attrs["permute"] = tuple(permute)

            # TODO: here, we force the batch dim to be dynamic
            # but this should be done when loading ov model
            i = layout.index("N")
            new_shape = []
            for shape in attrs["shape"]:
                new_shape.append([-1 if j == i else k for j, k in enumerate(shape)])
            new_shape = tuple(tuple(shape) for shape in new_shape)
            attrs["shape"] = new_shape

            # change shape and layout based on permute
            if "permute" in attrs and attrs["permute"] != (0, 1, 2, 3):
                assert len(attrs["shape"]) == 1
                permute = []
                for layout_ in input_layout.keys():
                    permute.append(layout.index(layout_))
                new_shape = []
                for shape in attrs["shape"]:
                    new_shape.append([shape[i] for i in permute])
                attrs["shape"] = tuple(tuple(shape) for shape in new_shape)
                attrs["layout"] = tuple([attrs["layout"][i] for i in permute])

        return cls(name=op_name, **attrs)


@dataclass
class ResultV0Attribute(Attribute):
    pass


@OPS.register()
class ResultV0(Operation):
    TYPE = "Result"
    VERSION = 0
    ATTRIBUTE_FACTORY = ResultV0Attribute

    def forward(self, input):
        return input


@dataclass
class ConstantV0Attribute(Attribute):
    element_type: str
    offset: int = field(default=0)
    size: int = field(default=0)

    is_parameter: bool = field(default=False)

    def __post_init__(self):
        super().__post_init__()
        # fmt: off
        valid_element_type = [
            "u1", "u4", "u8", "u16", "u32", "u64",
            "i4", "i8", "i16", "i32", "i64", "f16", "f32", "boolean", "bf16"
        ]
        # fmt: on
        if self.element_type not in valid_element_type:
            raise ValueError(
                f"Invalid element_type {self.element_type}. "
                f"It must be one of {valid_element_type}."
            )


@OPS.register()
class ConstantV0(Operation):
    TYPE = "Constant"
    VERSION = 0
    ATTRIBUTE_FACTORY = ConstantV0Attribute

    def __init__(self, data, *args, **kwargs):
        kwargs["element_type"] = ConvertV0.convert_torch_type(data.dtype)
        super().__init__(*args, **kwargs)
        if self.attrs.is_parameter:
            self.data = torch.nn.Parameter(data)
        else:
            self.register_buffer("data", data)

    def forward(self):
        return self.data

    @classmethod
    def from_ov(cls, ov_op):
        op_type = ov_op.get_type_name()
        op_version = ov_op.get_version()
        op_name = ov_op.get_friendly_name().replace(".", "_")
        assert cls.TYPE != "" and cls.VERSION >= 0
        assert op_type == cls.TYPE
        assert op_version == cls.VERSION

        attrs = ov_op.get_attributes()
        attrs["shape"] = tuple(attrs["shape"])

        data = torch.from_numpy(ov_op.get_data())

        in_port_indices = []
        for out_port in ov_op.outputs():
            for in_port in list(out_port.get_target_inputs()):
                in_port_index = in_port.get_index()
                in_port_indices.append(in_port_index)

        is_parameter = False
        if (
            len(in_port_indices) == 1
            and in_port_indices[0] == 1
            and (data.is_floating_point() or data.is_complex())
        ):
            is_parameter = True
        attrs["is_parameter"] = is_parameter

        return cls(data, name=op_name, **attrs)
