import copy
import unittest

# Import the registries and their corresponding make/get functions
from src import (
    data_registry,
    decision_maker_registry,
    get_data,
    make_model,
    model_registry,
)


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
        model_name = "knapsack_2D_Tang2022"
        original_params = copy.deepcopy(model_registry[model_name][1])  # Get original registered params

        # Call make_model with some overriding parameters
        override_capacity = 25
        override_weights_lb = 5.0
        _, final_params = make_model(model_name, capacity=override_capacity, weights_lb=override_weights_lb)

        # Assert that the original parameters in the registry remain unchanged
        self.assertEqual(model_registry[model_name][1]["capacity"], original_params["capacity"])
        self.assertEqual(model_registry[model_name][1]["weights_lb"], original_params["weights_lb"])
        self.assertEqual(model_registry[model_name][1]["weights_ub"], original_params["weights_ub"])
        self.assertEqual(
            model_registry[model_name][1]["num_decisions"],
            original_params["num_decisions"],
        )

        # Verify that the final_params indeed have the overrides
        self.assertEqual(final_params["capacity"], override_capacity)
        self.assertEqual(final_params["weights_lb"], override_weights_lb)
        # Ensure other params are from original
        self.assertEqual(final_params["weights_ub"], original_params["weights_ub"])

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
        self.assertEqual(data_registry[data_name][1]["num_data"], original_params["num_data"])
        self.assertEqual(data_registry[data_name][1]["noise_width"], original_params["noise_width"])
        self.assertEqual(data_registry[data_name][1]["num_features"], original_params["num_features"])

        # Verify that the final_params indeed have the overrides
        self.assertEqual(final_params["num_data"], override_num_data)
        self.assertEqual(final_params["noise_width"], override_noise_width)
        # Ensure other params are from original
        self.assertEqual(final_params["num_features"], original_params["num_features"])


# This allows running the tests directly from the file
if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
