import math
import numpy as np
import random

from swarms.lib.core.objects.agent import Agent
# from swarms.lib.core.objects.swarm import Swarm
# from swarms.lib.core.objects.world import World
# from swarms.lib.utils import generate_spawn_points

from shapely import Point

# typing
from typing import List, Tuple
try:
    from swarms.lib.sensors.binary import BinarySensor
    from swarms.lib.core.states.world_state import WorldState
    from shapely.geometry import Polygon
except (ImportError, ModuleNotFoundError):
    pass


class FlockbotBinarycontroller(Agent):
    agent_radius: float = 0.151

    def __init__(
        self,
        name: str,
        pos: Tuple[float, float, float],
        heading: float,
        spt: float,
        sensors: List[BinarySensor],
        controller=[0.1, 0.5, 0.1, -0.5],
        agent_radius=agent_radius,
    ) -> None:
        try:
            x, y, z = pos
        except ValueError:
            x, y = pos
            z = 0.0

        if isinstance(heading, float):
            yaw = heading
            roll = pitch = 0.0
        else:
            yaw, pitch, roll = heading
        super().__init__(x=x, y=y, z=z, pitch=pitch, roll=roll, yaw=yaw, name=name, spt=spt)

        self.agent_radius = agent_radius
        self.controller = controller
        self.sensors = sensors

        self.rng = random.Random()

    def get_action(self, world_state) -> Tuple:
        # observations = (sensor.sense() for sensor in self.sensors)
        observations = [sensor.sense(
            agent_name=self.name,
            world_state=world_state
        ) for sensor in self.sensors]
        sensor_triggered, *_ = observations  # unpack first observation
        c = self.controller
        v, omega = c[0:2] if sensor_triggered else c[2:4]
        self.requested = v, omega
        return self.requested

    def tick(self, world_state: WorldState):
        """
        Process the agent forward in time by one tick

        Arguments:
        ----------
        1) world_state (WorldState): a holistic view of everything within
            the simulated world at a given time step

        Returns:
        --------
        1) self (EpuckRobot): a reference to the EpuckRobot after a tick
            has been completed
        """

        # todo: implement argument validation
        actions = np.asarray(self.get_action(world_state))
        v, d_yaw = actions * self.spt  # scale each action by timestep.

        # implement dynamics
        dx = v * math.cos(self.yaw)
        dy = v * math.sin(self.yaw)
        self.yaw += d_yaw

        # d_yaw = (left_wheel_angular_velocity + right_wheel_angular_velocity) /
        # x_scalar: float = self.half_wheel_radius * math.cos(self.yaw)
        # y_scalar: float = self.half_wheel_radius * math.sin(self.yaw)

        # calculation_matrix = np.ndarray(
        #     shape=(3,2),
        #     buffer=np.array([
        #         [x_scalar, x_scalar],
        #         [y_scalar, y_scalar],
        #         [self.yaw_scalar, -self.yaw_scalar]
        # ]))
        # velocity_vector = np.ndarray(
        #     shape=(2,1),
        #     buffer=np.array([
        #         right_wheel_angular_velocity, left_wheel_angular_velocity
        # ]))

        # delta_vector = np.matmul(calculation_matrix, velocity_vector)

        x = dx + self.x
        y = dy + self.y
        # yaw = self.d_yaw + self.yaw

        field_polygon: Polygon = world_state.field_polygon
        new_location = Point([x, y])

        disable_walls = True

        newstate_is_valid = field_polygon.contains(new_location) or disable_walls

        if newstate_is_valid:
            self.dx = dx
            self.dy = dy
            self.d_yaw = d_yaw
            self.x = x
            self.y = y
            # self.yaw =
            # print(f"{self.tick_count} \t | a: {self.name} \t x: {self.x: 8.5}, y: {self.y: 8.5},")

        self.tick_count += 1
        return self
