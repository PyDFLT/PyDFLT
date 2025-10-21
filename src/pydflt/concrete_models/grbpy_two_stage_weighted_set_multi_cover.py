import random
from itertools import chain, combinations
from typing import Any, Optional

import cvxpy as cp
import gurobipy as gp
import numpy as np
from gurobipy import GRB

from pydflt.abstract_models.grbpy_two_stage import GRBPYTwoStageModel


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
        seed: int = 5,
        silvestri2023: bool = False,
        density: float = 0.25,
        num_scenarios: int = 1,
    ):
        """
        Initializes the WeightedSetMultiCover model.

        Args:
            num_items (int): Number of items that need to be covered.
            num_covers (int): Number of available covers (sets).
            penalty (float): Penalty for unmet coverage requirements.
            cover_costs_lb (int): Lower bound for cover costs.
            cover_costs_ub (int): Upper bound for cover costs.
            recovery_ratio (float): Ratio for recovering costs from unused covers. Defaults to 0.
            seed (int): Random seed for reproducible generation. Defaults to 0.
            silvestri2023 (bool): Whether to use Silvestri2023 parameter generation method. Defaults to False.
            density (float): Density of the item-cover matrix when using Silvestri2023 method. Defaults to 0.25.
            num_scenarios (int): Number of scenarios for multi-scenario optimization. Defaults to 1.
        """
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
        decision_variables = {"select_cover": (self.num_covers,)}
        # "unmet_coverage": (self.num_covers, num_scenarios)}
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
        """
        Creates the Gurobi optimization model for the weighted set multi-cover problem.
        This method defines the first and second stage variables, constraints, and objective function.

        Returns:
            tuple: A tuple containing the Gurobi model and the variables dictionary.
        """
        # Create a GP model
        gp_model = gp.Model("wsmc")
        vars_dict = {}
        second_stage_vars_dict = {}

        # Define variables
        # number of each cover that is picked
        x = gp_model.addMVar((self.num_covers,), vtype=GRB.INTEGER, name="select_cover")
        vars_dict["select_cover"] = x
        # Unmet coverage based on cover selection
        y = gp_model.addMVar((self.num_items, self.num_scenarios), vtype=GRB.INTEGER, name="unmet_coverage")
        second_stage_vars_dict["unmet_coverage"] = y

        if self.recovery_ratio > 0:  # Excess coverage that is not needed
            z = gp_model.addMVar((self.num_items, self.num_scenarios), vtype=GRB.INTEGER, name="excess_coverage")
            second_stage_vars_dict["excess_coverage"] = z

        # It is a minimization problem
        gp_model.modelSense = GRB.MINIMIZE
        assert self.model_sense_int == gp_model.modelSense, "Is it a maximization or minimization problem? Check model sense."

        # Set objective (there are no first stage constraints in this problem)
        obj = gp.quicksum(self.cover_costs[j] * x[j] for j in range(self.num_covers)) + gp.quicksum(
            self.penalty * self.max_cover_costs[i] * y[i, k] / self.num_scenarios for i in range(self.num_items) for k in range(self.num_scenarios)
        )

        if self.recovery_ratio > 0:
            obj -= gp.quicksum(
                self.recovery_ratio * self.cover_costs[i] * z[i, k] / self.num_scenarios for i in range(self.num_items) for k in range(self.num_scenarios)
            )

            # The first num_items of covers are the single item covers, in order
            gp_model.addConstrs(z[i, k] <= x[i] for i in range(self.num_items) for k in range(self.num_scenarios))

        gp_model.setObjective(obj)

        self.second_stage_vars_dict = second_stage_vars_dict

        return gp_model, vars_dict

    def _set_params(self, *parameters_i: np.ndarray):
        """
        Sets the parameters for the weighted set multi-cover model for a single instance.
        This corresponds to adjusting the coverage requirement scenarios in the constraints.

        Args:
            *parameters_i (np.ndarray): Coverage requirements for the current instance scenarios.
        """
        # Obtain the weight parameters
        cover_requirements = parameters_i[0]

        cover_requirements = cover_requirements.reshape(-1, self.num_scenarios)

        # Remove existing constraints
        self.gp_model.remove(self.gp_model.getConstrs())

        # Set new constraints
        x = self.vars_dict["select_cover"]
        y = self.second_stage_vars_dict["unmet_coverage"]
        if self.recovery_ratio > 0:
            z = self.second_stage_vars_dict["excess_coverage"]
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
        """
        Generates fixed parameters for the weighted set multi-cover problem.
        This method creates covers and their costs such that all sets are relevant.

        Args:
            cover_costs_lb (int): Lower bound for cover costs.
            cover_costs_ub (int): Upper bound for cover costs.
            seed (int): Random seed for reproducible generation.

        Returns:
            tuple: A tuple containing cover costs and item-cover matrix.
        """
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
        """
        Generates fixed parameters using the Silvestri2023 method.
        This method creates covers and their costs using a different approach.

        Args:
            cover_costs_lb (float): Lower bound for cover costs.
            cover_costs_ub (float): Upper bound for cover costs.
            density (float): Target density for the item-cover matrix.
            seed (int): Random seed for reproducible generation.

        Returns:
            tuple: A tuple containing cover costs and item-cover matrix.
        """
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
        """
        Returns the variable domains for the weighted set multi-cover problem when creating a quadratic variant.
        Only considers first stage variables since that's what's relevant for the quadratic variant.

        Returns:
            dict[str, dict[str, bool]]: A dictionary specifying variable domains for first stage variables.
        """
        # Since this function is to create a quadratic variant, we only care about first stage variables
        var_domain_dict = {"select_cover": {"integer": True}}  # boolean, integer, nonneg, nonpos, pos, imag, complex
        return var_domain_dict

    @staticmethod
    def get_constraints(vars_dict: dict[str, cp.Variable]) -> Optional[list[Any]]:
        """
        Returns the constraints for the weighted set multi-cover problem in CVXPY format.
        Used when creating a quadratic variant.

        Args:
            vars_dict (dict[str, cp.Variable]): A dictionary mapping variable names to CVXPY variables.

        Returns:
            Optional[list[Any]]: A list of CVXPY constraints ensuring non-negativity.
        """
        x = vars_dict["select_cover"]
        return [x >= 0]
