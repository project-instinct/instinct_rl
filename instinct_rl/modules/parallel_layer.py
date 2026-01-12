import os
from collections import OrderedDict
from copy import deepcopy
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from instinct_rl.modules.conv2d import Conv2dHeadModel
from instinct_rl.modules.mlp import MlpModel
from instinct_rl.modules.transformer import TransformerHeadModel
from instinct_rl.utils.utils import (
    get_subobs_by_components,
    get_subobs_size,
    module_is_from_type,
)


class ParallelLayer(nn.Module):
    """A parallel block module that can manage multiple network blocks and run the datastream in parallel, and concatenate the outputs."""

    def __init__(
        self,
        input_segments: Dict[str, tuple],
        block_configs: Dict[str, dict],
        sequential_idx: int = 0,  # in case of sequential parallel layers that have the same block name
    ):
        """
        Args:
            - block_configs: each dict contains the following keys:
                - component_names: (list of str / str) list of names of the components
                    of the obs vector to be encoded. or the single name of the component.
                - class_name: (str) accept list of names (in the same order as encoder_component_names)
                - output_size / output_shape: (int) output size/shape of the encoder, which is allowed to modify inplace by the block class.
                - takeout_input_components: (bool) whether to take out the obs components after encoding.
                    or, whether the obs_components are visible to the rest of the network.
                -- **kwargs: for building the encoder network.
        """
        super().__init__()

        self.input_segments = input_segments
        self.block_configs = block_configs
        self._sequential_idx = sequential_idx
        self._output_component_name_prefix = f"parallel_latent_{self._sequential_idx}_"

        self.build_blocks()
        self.build_output_segment()

    def build_blocks(self):
        """Build all encoders one by one."""
        self._parallel_blocks = nn.ModuleDict()
        for block_name, config in self.block_configs.items():
            self._parallel_blocks[block_name] = self._build_one_block(self.input_segments, config)
        return self._parallel_blocks

    def _build_one_block(self, input_segments, config) -> nn.Module:
        """Build one encoder. (involves all the non-standard construction operations)"""
        model_kwargs = deepcopy(config)
        model_class_name = model_kwargs.pop("class_name")
        input_component_shapes = [input_segments[name] for name in model_kwargs.pop("component_names")]
        output_size = model_kwargs.pop("output_size")
        model_kwargs.pop("takeout_input_components")
        # This code is not clean enough, need to sort out later
        if model_class_name == "MlpModel":
            hidden_sizes = model_kwargs.pop("hidden_sizes") + [
                output_size,
            ]
            model = MlpModel(
                np.sum(np.prod(s) for s in input_component_shapes),
                hidden_sizes=hidden_sizes,
                output_size=None,
                **model_kwargs,
            )
        elif model_class_name == "Conv2dHeadModel":
            assert len(input_component_shapes) == 1, "Conv2dHeadModel only accept one obs component for now"
            hidden_sizes = model_kwargs.pop("hidden_sizes") + [
                output_size,
            ]
            model = Conv2dHeadModel(
                input_component_shapes[0],
                hidden_sizes=hidden_sizes,
                output_size=None,
                **model_kwargs,
            )
        elif model_class_name == "TransformerHeadModel":
            model = TransformerHeadModel(
                input_component_shapes,
                output_size=output_size,
                **model_kwargs,
            )
        else:
            model = None  # leave for subclass to implement
        return model

    def build_output_segment(self):
        """Build the output segment for the parallel layer."""
        self.output_segment = deepcopy(self.input_segments)
        components_to_takeout = set()
        for block_name, config in self.block_configs.items():
            self.output_segment[self._output_component_name_prefix + block_name] = config.get(
                "output_shape", (config["output_size"],)
            )
            if config.get("takeout_input_components", False):
                components_to_takeout.update(config["component_names"])
        if len(components_to_takeout) > 0:
            self.output_segment = OrderedDict(
                [(name, shape) for name, shape in self.output_segment.items() if name not in components_to_takeout]
            )
        self.numel_output = get_subobs_size(self.output_segment)
        return self.output_segment

    def run_blocks(self, flat_input: torch.Tensor) -> torch.Tensor:
        """Run all blocks in parallel and contact the flattened output."""
        blocks_outputs = OrderedDict()
        leading_dim = flat_input.shape[:-1]
        for block_name, block in self._parallel_blocks.items():
            blocks_outputs[self._output_component_name_prefix + block_name] = self._run_one_block(
                flat_input,
                self.input_segments,
                self.block_configs[block_name]["component_names"],
                block,
            )
        outputs = []
        for output_component_name, output_shape in self.output_segment.items():
            if output_component_name.startswith(self._output_component_name_prefix):
                outputs.append(blocks_outputs[output_component_name].reshape(*leading_dim, -1))
            else:
                outputs.append(
                    get_subobs_by_components(flat_input, [output_component_name], self.input_segments).reshape(
                        *leading_dim, -1
                    )
                )
        return torch.cat(outputs, dim=-1)

    def _run_one_block(self, flat_input, input_segments, input_component_names, block):
        input_for_block = get_subobs_by_components(
            flat_input,
            input_component_names,
            input_segments,
            temporal=module_is_from_type(block, TransformerHeadModel),
        )
        if module_is_from_type(block, Conv2dHeadModel):
            assert len(input_component_names) == 1, "Conv2dHeadModel only accept one obs component for now"
            input_for_block = input_for_block.reshape(-1, *input_segments[input_component_names[0]])
        return block(input_for_block)

    """
    Torch module related methods
    """

    def forward(self, flat_input: torch.Tensor) -> torch.Tensor:
        """TODO: support rnn block."""
        return self.run_blocks(flat_input)

    def __str__(self):
        return f"ParallelLayer({len(self.block_configs)} blocks): {self._parallel_blocks}"

    def export_as_onnx(self, flat_input, filedir: str, block_as_seperate_files: bool = True):
        """Export the model as an ONNX file. Input should be batch-wise observations with batchsize 1."""
        self.eval()
        assert block_as_seperate_files, "Currently only support exporting blocks as separate files."
        with torch.no_grad():
            for block_name in self._parallel_blocks.keys():
                self.export_one_block_as_onnx(flat_input, filedir, block_name)

    def export_one_block_as_onnx(self, flat_input, filedir, block_name):
        block = self._parallel_blocks[block_name]
        block_config = self.block_configs[block_name]
        input_component_names = block_config["component_names"]
        input_for_block = get_subobs_by_components(
            flat_input,
            input_component_names,
            self.input_segments,
            temporal=module_is_from_type(block, TransformerHeadModel),
        )
        if module_is_from_type(block, Conv2dHeadModel):
            assert len(input_component_names) == 1, "Conv2dHeadModel only accept one obs component for now"
            input_for_block = input_for_block.reshape(-1, *self.input_segments[input_component_names[0]])
        if module_is_from_type(block, TransformerHeadModel):
            torch.backends.cuda.enable_mem_efficient_sdp(False)  # Disable Memory-Efficient Attention
        exported_program = torch.onnx.export(
            block,
            input_for_block,
            "/tmp/parallel_layer.onnx",  # This file does not contain the model weight, we call the save later to save the onnx with model weight.
            input_names=["input"],
            output_names=["output"],
            dynamo=True,
            opset_version=15,  # on pytorch 2.4.0 for transformer encoder. Not sure for others.
        )
        exported_program.save(os.path.join(filedir, f"{self._sequential_idx}-{block_name}.onnx"))
        print(f"Exported {block_name} to {os.path.join(filedir, f'{self._sequential_idx}-{block_name}.onnx')}")
