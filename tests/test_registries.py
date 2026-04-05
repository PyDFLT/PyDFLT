import copy
import unittest

# Import the registries and their corresponding make/get functions
from pydflt.concrete_models import GRBPYKnapsackModel
from pydflt.generate_data_functions import gen_data_knapsack
from pydflt.problem import Problem
from pydflt.registries.data import data_registry, get_data
from pydflt.registries.decision_makers import decision_maker_registry, make_decision_maker
from pydflt.registries.models import make_model, model_registry
from pydflt.utils.load import load_data_from_dict


class TestRegistries(unittest.TestCase):
    """
    This test suite verifies that the parameters stored in the model, decision maker,
    and data registries are not accidentally modified when their respective
    'make' or 'get' functions are called with overriding parameters. It does not currently test to see if
    everything registered is properly registered!
    It specifically checks for deep copy correctness for mutable parameters.
    """

    def setUp(self):
        """
        Set up fresh registries before each test to ensure isolation.
        This is crucial because the registries are global dictionaries.
        """
        # Save original states of registries
        self._original_model_registry = copy.deepcopy(model_registry)
        self._original_decision_maker_registry = copy.deepcopy(decision_maker_registry)
        self._original_data_registry = copy.deepcopy(data_registry)

    def tearDown(self):
        """
        Restore original states of registries after each test.
        """
        model_registry.clear()
        model_registry.update(self._original_model_registry)

        decision_maker_registry.clear()
        decision_maker_registry.update(self._original_decision_maker_registry)

        data_registry.clear()
        data_registry.update(self._original_data_registry)

    def test_model_registry_parameters_not_overwritten(self):
        """
        Verify that calling make_model with override_params does not modify
        the original parameters stored in the model_registry.
        """
        model_name = "knapsack_2D_Tang2024"
        original_params = copy.deepcopy(model_registry[model_name][1])  # Get original registered params

        # Call make_model with some overriding parameters
        override_capacity = 25
        override_weights_lb = 5.0
        _, final_params = make_model(model_name, capacity=override_capacity, weights_lb=override_weights_lb)

        # Assert that the original parameters in the registry remain unchanged
        assert model_registry[model_name][1]["capacity"] == original_params["capacity"]
        assert model_registry[model_name][1]["weights_lb"] == original_params["weights_lb"]
        assert model_registry[model_name][1]["weights_ub"] == original_params["weights_ub"]
        assert model_registry[model_name][1]["num_decisions"] == original_params["num_decisions"]

        # Verify that the final_params indeed have the overrides
        assert final_params["capacity"] == override_capacity
        assert final_params["weights_lb"] == override_weights_lb
        # Ensure other params are from original
        assert final_params["weights_ub"] == original_params["weights_ub"]

    def test_data_registry_parameters_not_overwritten(self):
        """
        Verify that calling get_data with override_params does not modify
        the original parameters stored in the data_registry.
        """
        data_name = "knapsack"
        original_params = copy.deepcopy(data_registry[data_name][1])

        # Call get_data with some overriding parameters
        override_num_data = 100
        override_noise_width = 0.1
        _, final_params = get_data(
            data_name,
            num_data=override_num_data,
            noise_width=override_noise_width,
        )

        # Assert that the original parameters in the registry remain unchanged
        assert data_registry[data_name][1]["num_data"] == original_params["num_data"]
        assert data_registry[data_name][1]["noise_width"] == original_params["noise_width"]
        assert data_registry[data_name][1]["num_features"] == original_params["num_features"]

        # Verify that the final_params indeed have the overrides
        assert final_params["num_data"] == override_num_data
        assert final_params["noise_width"] == override_noise_width
        # Ensure other params are from original
        assert final_params["num_features"] == original_params["num_features"]

    def test_all_models_instantiate(self):
        """
        Verify that every model registered in the model registry can be instantiated
        without raising an error.
        """
        for name in model_registry:
            with self.subTest(model=name):
                make_model(name)

    def test_all_data_functions_call(self):
        """
        Verify that every data function registered in the data registry can be called
        without raising an error. Entries that require an external file path (e.g.
        load_data_from_dict) are skipped because they have no valid default path.
        """
        for name in data_registry:
            with self.subTest(data=name):
                data_fn, _ = data_registry[name]
                if data_fn is load_data_from_dict:
                    # This entry requires a caller-supplied file path and cannot be
                    # smoke-tested without one; skip it.
                    continue
                get_data(name, num_data=10)

    def test_all_decision_makers_instantiate(self):
        """
        Verify that every decision maker registered in the decision maker registry can
        be instantiated without raising an error.

        A small knapsack problem is used as the shared problem instance. It is
        compatible with all registered decision makers, including those that require
        PyEPO-style losses (SPOPlus) and those that run solver calls during __init__
        (cave, lava).
        """
        opt_model = GRBPYKnapsackModel(
            num_decisions=5,
            capacity=10,
            weights_lb=1,
            weights_ub=3,
            dimension=1,
            seed=1,
        )
        data_dict = gen_data_knapsack(
            seed=1,
            num_data=10,
            num_features=2,
            num_items=5,
            dimension=1,
            polynomial_degree=1,
            noise_width=0.5,
        )
        problem = Problem(
            data_dict=data_dict,
            opt_model=opt_model,
            train_ratio=0.6,
            val_ratio=0.2,
            compute_optimal_decisions=True,
            compute_optimal_objectives=True,
            seed=1,
        )

        for name in decision_maker_registry:
            with self.subTest(decision_maker=name):
                make_decision_maker(problem, name)


# This allows running the tests directly from the file
if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
