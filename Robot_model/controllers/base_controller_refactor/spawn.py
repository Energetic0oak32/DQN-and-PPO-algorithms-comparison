import numpy as np

from config import (
    RANDOM_SPAWN,
    SPAWN_BLOCKED_OBJECT_TYPES,
    SPAWN_MAX_ATTEMPTS,
    SPAWN_MIN_OBJECT_DISTANCE,
    SPAWN_X_RANGE,
    SPAWN_Y_RANGE,
    SPAWN_YAW_RANGE,
)


class SpawnManager:
    def __init__(self, robot, robot_node, initial_translation, initial_rotation):
        self.robot = robot
        self.robot_node = robot_node
        self.initial_translation = initial_translation
        self.initial_rotation = initial_rotation
        self.blockers = self._get_spawn_blockers()

    def restore_robot_pose(self, np_random, random_spawn=RANDOM_SPAWN):
        if self.robot_node is None:
            return

        translation = list(self.initial_translation)
        rotation = list(self.initial_rotation)

        if random_spawn:
            translation, rotation = self._make_random_spawn_pose(np_random)

        self.robot_node.getField("translation").setSFVec3f(translation)
        self.robot_node.getField("rotation").setSFRotation(rotation)
        self.robot_node.resetPhysics()

        if hasattr(self.robot, "simulationResetPhysics"):
            self.robot.simulationResetPhysics()

    def _make_random_spawn_pose(self, np_random):
        translation = list(self.initial_translation)
        rotation = list(self.initial_rotation)

        for _ in range(SPAWN_MAX_ATTEMPTS * 10):
            x = float(np_random.uniform(SPAWN_X_RANGE[0], SPAWN_X_RANGE[1]))
            y = float(np_random.uniform(SPAWN_Y_RANGE[0], SPAWN_Y_RANGE[1]))
            yaw = float(np_random.uniform(SPAWN_YAW_RANGE[0], SPAWN_YAW_RANGE[1]))

            candidate_translation = [
                x,
                y,
                self.initial_translation[2],
            ]

            if self.is_clear_of_blockers(candidate_translation):
                translation = candidate_translation
                rotation = [
                    0.0,
                    0.0,
                    1.0,
                    yaw,
                ]
                break

        if not self.is_clear_of_blockers(translation):
            raise RuntimeError(
                "Could not find a random spawn position clear of blocked objects. "
                "Increase SPAWN_X_RANGE/SPAWN_Y_RANGE or reduce SPAWN_MIN_OBJECT_DISTANCE."
            )

        return translation, rotation

    def _get_spawn_blockers(self):
        if not hasattr(self.robot, "getRoot"):
            return []

        root = self.robot.getRoot()
        if root is None:
            return []

        children = root.getField("children")
        if children is None:
            return []

        return self._find_spawn_blockers(children)

    def _find_spawn_blockers(self, children_field):
        blockers = []

        for i in range(children_field.getCount()):
            node = children_field.getMFNode(i)

            if node.getTypeName() in SPAWN_BLOCKED_OBJECT_TYPES:
                translation = self._get_node_translation(node)
                if translation is not None:
                    blockers.append(translation)

            blockers.extend(self._find_nested_spawn_blockers(node))

        return blockers

    def _find_nested_spawn_blockers(self, node):
        children = node.getField("children")
        if children is None:
            return []

        try:
            return self._find_spawn_blockers(children)
        except Exception:
            return []

    def _get_node_translation(self, node):
        translation_field = node.getField("translation")
        if translation_field is None:
            return None

        try:
            return translation_field.getSFVec3f()
        except Exception:
            return None

    def is_clear_of_blockers(self, translation):
        for blocker_translation in self.blockers:
            distance = float(
                np.linalg.norm(
                    np.array(translation[:2], dtype=np.float32)
                    - np.array(blocker_translation[:2], dtype=np.float32)
                )
            )
            if distance < SPAWN_MIN_OBJECT_DISTANCE:
                return False

        return True
