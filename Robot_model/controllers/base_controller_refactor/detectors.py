import numpy as np

from config import (
    ARENA_HALF_SIZE,
    DETECTOR_DEFAULT_RADIUS,
    DETECTOR_RADII,
    OIL_BARREL_RADIUS,
)


DETECTOR_DEFS = {
    "center": "CENTER",
    "front": "FRONT",
    "left": "LEFT",
    "right": "RIGHT",
    "back": "BACK",
}


class DetectorManager:
    def __init__(self, supervisor):
        self.supervisor = supervisor
        self.detectors = self._get_detector_nodes()
        self.barrels = self._get_barrel_nodes()

    def get_hits(self):
        hits = {}

        for name, node in self.detectors.items():
            position = self._get_node_position(node)
            radius = self._get_detector_radius(name, node)
            hits[name] = self._is_hit(position, radius)

        return {
            "center": hits.get("center", False),
            "front": hits.get("front", False),
            "left": hits.get("left", False),
            "right": hits.get("right", False),
            "back": hits.get("back", False),
        }

    def _get_detector_nodes(self):
        nodes = {}

        if not hasattr(self.supervisor, "getFromDef"):
            return nodes

        for name, def_name in DETECTOR_DEFS.items():
            node = self.supervisor.getFromDef(def_name)
            if node is not None:
                nodes[name] = node

        return nodes

    def _get_barrel_nodes(self):
        if not hasattr(self.supervisor, "getRoot"):
            return []

        root = self.supervisor.getRoot()
        if root is None:
            return []

        children = root.getField("children")
        if children is None:
            return []

        return self._find_nodes_by_type(children, "OilBarrel")

    def _find_nodes_by_type(self, children_field, type_name):
        nodes = []

        for i in range(children_field.getCount()):
            node = children_field.getMFNode(i)

            if node.getTypeName() == type_name:
                nodes.append(node)

            children = node.getField("children")
            if children is None:
                continue

            try:
                nodes.extend(self._find_nodes_by_type(children, type_name))
            except Exception:
                continue

        return nodes

    def _is_hit(self, position, radius):
        if position is None:
            return False

        return self._hits_wall(position, radius) or self._hits_barrel(position, radius)

    def _hits_wall(self, position, radius):
        x, y = position[:2]
        return abs(x) + radius >= ARENA_HALF_SIZE or abs(y) + radius >= ARENA_HALF_SIZE

    def _hits_barrel(self, position, radius):
        detector_xy = np.array(position[:2], dtype=np.float32)

        for barrel in self.barrels:
            barrel_position = self._get_node_position(barrel)
            if barrel_position is None:
                continue

            barrel_xy = np.array(barrel_position[:2], dtype=np.float32)
            distance = float(np.linalg.norm(detector_xy - barrel_xy))

            if distance <= radius + OIL_BARREL_RADIUS:
                return True

        return False

    def _get_node_position(self, node):
        if hasattr(node, "getPosition"):
            return node.getPosition()

        translation_field = node.getField("translation")
        if translation_field is None:
            return None

        return translation_field.getSFVec3f()

    def _get_detector_radius(self, name, node):
        radius = self._get_radius_from_geometry(node)
        if radius is not None:
            return radius

        return DETECTOR_RADII.get(name, DETECTOR_DEFAULT_RADIUS)

    def _get_radius_from_geometry(self, node):
        children = node.getField("children")
        if children is None or children.getCount() == 0:
            return None

        shape = children.getMFNode(0)
        geometry = shape.getField("geometry")
        if geometry is None:
            return None

        try:
            geometry_node = geometry.getSFNode()
            radius = geometry_node.getField("radius")
            if radius is None:
                return None
            return float(radius.getSFFloat())
        except Exception:
            return None
