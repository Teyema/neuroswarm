import numpy as np
from novel_swarms.metrics.Circliness import RadialVarianceHelper

# typing
from typing import Any, override

from functools import cached_property


class CoordinateTest(RadialVarianceHelper):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = self.__class__.__name__

    @override
    def _calculate(self):
        # This function creates a metric for the agents to follow

        distances = [self.distance(agent.getPosition(), np.asarray([0, 0])) for agent in self.population]
        
        runner_agent = next(agent for agent in self.population if agent.name == "runner")
        runner_position = runner_agent.getPosition()
        goal_position = np.asarray(runner_agent.controller.goal_object.pos) 
        dist_to_goal = self.distance(runner_position, goal_position)

        for agent in self.population:
            if agent.agent_in_sight.name == "runner": 
                return 1
            elif dist_to_goal > runner_agent.radius:
                return 2

        return 0
     
    @staticmethod
    def distance(a, b):
        return np.linalg.norm(a - b)
    
