from typing import Dict, List, Optional

from mpa.utils.logger import get_logger

from ....models.ov_model import CONNECTION_SEPARATOR
from ..builder import PARSERS
from ..parser import parameter_parser


logger = get_logger()


@PARSERS.register()
def yolo_parser(graph, component: str = "backbone") -> Optional[Dict[str, List[str]]]:
    assert component in ["backbone", "neck"]

    result_nodes = graph.get_nodes_by_types(["Result"])

    neck_detection_block_outputs = []
    neck_detection_block_inputs = []
    neck_conv_block_outputs = []
    neck_conv_block_inputs = []
    backbone = {"inputs": [], "outputs": []}
    neck = {"inputs": {}, "outputs": {}}

    head_depths = []

    for result_node in result_nodes:
        print(result_node.name)
        depth = 0
        for s, t in graph.bfs(result_node, reverse=True, depth_limit=7):
            depth += 1
            if s.type != "Concat" and len(list(graph.successors(s))) == 2:
                head_depths.append(depth)
                print(s.name)
                break
    head_depth = min(head_depths)

    for result_node in result_nodes:
        depth = 0
        for s, t in graph.bfs(result_node, reverse=True, depth_limit=10):
            depth += 1
            if depth == head_depth:
                neck_detection_block_outputs.append(s)

    neck_detection_block_outputs, result_nodes = list(
        map(
            list,
            zip(
                *sorted(
                    zip(neck_detection_block_outputs, result_nodes),
                    key=lambda pair: pair[0].shape[0][1],
                    reverse=True,
                )
            ),
        )
    )

    for neck_detection_block_output, result_node in zip(
        neck_detection_block_outputs, result_nodes
    ):
        for s, t in graph.bfs(
            neck_detection_block_output, reverse=True, depth_limit=20
        ):
            if s.type == "Concat":
                neck_detection_block_inputs.append(t)
                break
            elif s.type == "Add":
                predecessors = list(graph.predecessors(s))
                if "Add" in [prdecessor.type for prdecessor in predecessors]:
                    neck_detection_block_inputs.append(t)
                    break

        successors = list(graph.successors(neck_detection_block_output))
        if len(successors) > 2:
            logger.info("Output of neck has more than two nodes")
            return None
        for node in successors:
            if graph.has_path(node, result_node):
                continue
            neck_conv_block_inputs.append(node)
            for s, ts in graph.bfs(node, depth_limit=10):
                if len(list(graph.successors(s))) > 1:
                    neck_conv_block_outputs.append(s)
                    break

    __import__('ipdb').set_trace()
    for neck_detection_block_input in neck_detection_block_inputs:
        concat_or_add = [
            node
            for node in graph.predecessors(neck_detection_block_input)
            if node.type != "Constant"
        ]
        assert len(concat_or_add) == 1 and concat_or_add[0].type in ["Concat", "Add"]
        concat_or_add = concat_or_add[0]
        if concat_or_add.type == "Add":
            backbone["outputs"].append(concat_or_add.name)
        else:
            for node in graph.predecessors(concat_or_add):
                if node.type == "Add":
                    backbone["outputs"].append(
                        f"{node.name}{CONNECTION_SEPARATOR}{concat_or_add.name}"
                    )
                    break
    # reverse
    backbone["outputs"] = backbone["outputs"][::-1]
    backbone["inputs"] = parameter_parser(graph)

    for i, neck_detection_block_input in enumerate(neck_detection_block_inputs):
        neck["inputs"][f"detect{i+1}"] = neck_detection_block_input.name
    for i, neck_detection_block_output in enumerate(neck_detection_block_outputs):
        neck["outputs"][f"detect{i+1}"] = neck_detection_block_output.name
    for i, neck_conv_block_input in enumerate(neck_conv_block_inputs):
        neck["inputs"][f"conv{i+1}"] = neck_conv_block_input.name
    for i, neck_conv_block_output in enumerate(neck_conv_block_outputs):
        neck["outputs"][f"conv{i+1}"] = neck_conv_block_output.name

    if component == "backbone":
        return backbone
    elif component == "neck":
        return neck
    else:
        raise NotImplementedError
