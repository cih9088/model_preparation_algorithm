from typing import Dict, List, Optional

from .. import PARSERS


@PARSERS.register()
def faster_rcnn_parser(graph, component: str) -> Optional[Dict[str, List[str]]]:
    result_nodes = graph.get_nodes_by_types(["Result"])
    __import__("ipdb").set_trace()
