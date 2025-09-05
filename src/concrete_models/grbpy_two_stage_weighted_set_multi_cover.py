import random
from itertools import chain, combinations
from typing import Any, Optional

import cvxpy as cp
import gurobipy as gp
import numpy as np
from gurobipy import GRB

from src.abstract_models.grbpy_two_stage import GRBPYTwoStageModel


class WeightedSetMultiCover(GRBPYTwoStageModel):
    """
    This weighted set multi cover problem is designed such that all sets are relevant, i.e. there are no sets that
    are equal to another set but with higher costs, and no sets such that the union of some of its subsets is cheaper.
    It has an additional recourse action being able to recover costs from sets that are unused.
    """

    def __init__(
        self,
        num_items: int,
        num_covers: int,
        penalty: float,
        cover_costs_lb: int,
        cover_costs_ub: int,
        recovery_ratio: float = 0,
        seed: int = 0,
        silvestri2023: bool = False,
        density: float = 0.25,
        num_scenarios: int = 1,
    ):
        # Setting input parameters
        self.num_items = num_items
        self.num_covers = num_covers
        self.penalty = penalty
        self.cover_costs_lb = cover_costs_lb
        self.cover_costs_ub = cover_costs_ub
        self.recovery_ratio = recovery_ratio
        self.seed = seed
        self.silvestri2023 = silvestri2023
        self.density = density
        self.num_scenarios = num_scenarios

        # Setting basic model parameters
        model_sense = "MIN"
        decision_variables = {"x": (self.num_covers,)}
        #'y': (self.num_covers, num_scenarios)}
        _shape = (self.num_items, num_scenarios) if num_scenarios > 1 else (num_items,)
        param_to_predict_shapes = {"coverage_requirements": _shape}
        extra_param_shapes = None

        # Setting additional model parameters
        if self.silvestri2023:
            self.cover_costs, self.item_cover_matrix = self._set_fixed_parameters_silvestri2023(cover_costs_lb, cover_costs_ub, density, seed)
        else:
            self.cover_costs, self.item_cover_matrix = self._set_fixed_parameters(cover_costs_lb, cover_costs_ub, seed)
        self.max_cover_costs = (self.item_cover_matrix * self.cover_costs).max(axis=1)

        GRBPYTwoStageModel.__init__(
            self,
            decision_variables,
            param_to_predict_shapes,
            model_sense,
            extra_param_shapes=extra_param_shapes,
        )

    def _create_model(self):
        """Creates model AND variables_dict"""
        # Create a GP model
        self.gp_model = gp.Model("wsmc")
        self.vars_dict = {}
        self.second_stage_vars_dict = {}

        # Define variables
        # number of each cover that is picked
        x = self.gp_model.addMVar((self.num_covers,), vtype=GRB.INTEGER, name="x")
        self.vars_dict["x"] = x
        # Coverage an item missed in x
        y = self.gp_model.addMVar((self.num_items, self.num_scenarios), vtype=GRB.INTEGER, name="y")
        self.second_stage_vars_dict["y"] = y

        if self.recovery_ratio > 0:  # Over-coverage that is taken away
            z = self.gp_model.addMVar((self.num_items, self.num_scenarios), vtype=GRB.INTEGER, name="z")
            self.second_stage_vars_dict["z"] = z

        # It is a minimization problem
        self.gp_model.modelSense = GRB.MINIMIZE
        assert self.model_sense_int == self.gp_model.modelSense, "Is it a maximization or minimization problem? Check model sense."

        # Set objective (there are no first stage constraints in this problem)
        obj = gp.quicksum(self.cover_costs[j] * x[j] for j in range(self.num_covers)) + gp.quicksum(
            self.penalty * self.max_cover_costs[i] * y[i, k] / self.num_scenarios for i in range(self.num_items) for k in range(self.num_scenarios)
        )

        if self.recovery_ratio > 0:
            obj -= gp.quicksum(
                self.recovery_ratio * self.cover_costs[i] * z[i, k] / self.num_scenarios for i in range(self.num_items) for k in range(self.num_scenarios)
            )

            # The first num_items of covers are the single item covers, in order
            self.gp_model.addConstrs(z[i, k] <= x[i] for i in range(self.num_items) for k in range(self.num_scenarios))

        self.gp_model.setObjective(obj)

    def _set_params(self, *parameters_i: np.ndarray):
        """
        Set parameters for the second stage, this corresponds to adjusting the cover_requirement scenarios in the constraints.
        """
        # Obtain the weight parameters
        cover_requirements = parameters_i[0]

        cover_requirements = cover_requirements.reshape(-1, self.num_scenarios)

        # Remove existing constraints
        self.gp_model.remove(self.gp_model.getConstrs())

        # Set new constraints
        x = self.vars_dict["x"]
        y = self.second_stage_vars_dict["y"]
        if self.recovery_ratio > 0:
            z = self.second_stage_vars_dict["z"]
            self.gp_model.addConstrs(z[i, k] <= x[i] for i in range(self.num_items) for k in range(self.num_scenarios))
            self.gp_model.addConstrs(
                gp.quicksum(self.item_cover_matrix[i, j] * x[j] for j in range(self.num_covers)) + y[i, k] - z[i, k] >= cover_requirements[i, k]
                for i in range(self.num_items)
                for k in range(self.num_scenarios)
            )
        else:
            self.gp_model.addConstrs(
                gp.quicksum(self.item_cover_matrix[i, j] * x[j] for j in range(self.num_covers)) + y[i, k] >= cover_requirements[i, k]
                for i in range(self.num_items)
                for k in range(self.num_scenarios)
            )

    def _set_fixed_parameters(self, cover_costs_lb: int, cover_costs_ub: int, seed: int):
        np.random.seed(seed)
        random.seed(seed)

        # We iterate through all possible combinations
        cover_costs_dict = {}
        cover_costs = np.zeros(self.num_covers)
        item_cover_matrix = np.zeros((self.num_items, self.num_covers))
        cover_idx = 0
        num_items_to_cover = 0
        while cover_idx < self.num_covers:
            num_items_to_cover += 1
            # Generate all possible combinations of `num_ones` positions, go over them randomly
            combinations_of_positions = list(combinations(range(self.num_items), num_items_to_cover))
            if num_items_to_cover > 1:  # we shuffle the combinations for set that covers multiple items
                random.shuffle(combinations_of_positions)
            for ones_positions in combinations_of_positions:
                if cover_idx >= self.num_covers:
                    break  # Stop if we filled all columns
                item_cover_matrix[list(ones_positions), cover_idx] = 1
                if num_items_to_cover == 1:
                    costs = np.random.randint(cover_costs_lb, cover_costs_ub + 1)
                else:  # num_items_to_cover > 1
                    subsets = list(chain.from_iterable(combinations(ones_positions, r) for r in range(1, len(ones_positions))))
                    disjoint_union_subsets = []
                    for r in range(1, len(subsets) + 1):
                        for combo in combinations(subsets, r):
                            # Check if they cover the ones_positions and are disjoint
                            if ones_positions == tuple(set().union(*combo)) and all(
                                set(x).isdisjoint(set(y)) for i, x in enumerate(combo) for y in combo[i + 1 :]
                            ):
                                disjoint_union_subsets.append(combo)
                    min_costs = max([cover_costs_dict[s] for s in subsets])
                    max_costs = min([sum([cover_costs_dict[s] for s in dus]) for dus in disjoint_union_subsets])
                    costs = int((max_costs + min_costs) / 2)
                    # np.random.randint(min_costs, max_costs)
                cover_costs[cover_idx] = costs
                cover_costs_dict[ones_positions] = costs
                cover_idx += 1

        return cover_costs, item_cover_matrix

    def _set_fixed_parameters_silvestri2023(self, cover_costs_lb: float, cover_costs_ub: float, density: float, seed: int):
        np.random.seed(seed)
        cover_costs = np.random.uniform(cover_costs_lb, cover_costs_ub, self.num_covers)
        item_cover_matrix = np.zeros((self.num_items, self.num_covers))

        for item in range(self.num_items):  # get two covers for each item
            cover_1 = np.random.randint(0, self.num_covers)
            leftover_covers = [i for i in range(0, self.num_covers) if i != cover_1]
            cover_2 = leftover_covers[np.random.randint(0, self.num_covers - 1)]
            item_cover_matrix[item, cover_1] = 1
            item_cover_matrix[item, cover_2] = 1
        for cover in range(self.num_covers):  # cover an item
            item = np.random.randint(0, self.num_items)
            item_cover_matrix[item, cover] = 1

        # add until density is reached, note that with small problem cases density is often already a lot higher
        while item_cover_matrix.mean() < density:
            item = np.random.randint(0, self.num_items)
            cover = np.random.randint(0, self.num_covers)
            item_cover_matrix[item, cover] = 1

        return cover_costs, item_cover_matrix

    @staticmethod
    def get_var_domains() -> dict[str, dict[str, bool]]:
        # Since this function is to create a quadratic variant, we only care about first stage variables
        var_domain_dict = {"x": {"integer": True}}  # boolean, integer, nonneg, nonpos, pos, imag, complex
        return var_domain_dict

    @staticmethod
    def get_constraints(vars_dict: dict[str, cp.Variable]) -> Optional[list[Any]]:
        x = vars_dict["x"]
        return [x >= 0]
