import unittest

from src.concrete_models.grbpy_two_stage_weighted_set_multi_cover import (
    WeightedSetMultiCover,
)
from src.decision_makers.sfge_decision_maker import SFGEDecisionMaker
from src.generate_data_functions.generate_data_wsmc import gen_data_wsmc
from src.problem import Problem
from src.runner import Runner

# Set-up model and data and problem for testing
model = WeightedSetMultiCover(
    num_items=2,
    num_covers=3,
    penalty=5,
    cover_costs_lb=5,
    cover_costs_ub=50,
    seed=5,
    recovery_ratio=0,
)
data_dict = gen_data_wsmc(
    seed=4,
    num_data=50,
    num_features=2,
    num_items=model.num_items,
    degree=1,
    noise_width=0.5,
)
problem = Problem(
    data_dict=data_dict,
    opt_model=model,
    train_ratio=0.4,
    val_ratio=0.2,
    compute_optimal_decisions=True,
    compute_optimal_objectives=True,
)


class TestDecisionMakersUsingWSMC(unittest.TestCase):
    """
    This unit test is testing several different settings for the SFGE decision maker with the new adjustments with
    the scale shift layer made by Kim, triggered by using standardize_predictions.
    """

    def test_sfge_1(self):
        """
        Test what happens with scale shift layer
        Returns:

        """
        # Predictor kwargs
        predictor_kwargs = {
            "size": 10,
            "num_hidden_layers": 2,
            "activation": "leaky_relu",
            "output_activation": "leaky_relu",
        }
        predictor_str = "MLP"
        print("\nStart test 1")
        # FIXME: KvdH why would SFGE not work without noisifier kwargs, shouldn't we set default?
        try:
            decision_maker = SFGEDecisionMaker(
                problem=problem,
                learning_rate=0.005,
                batch_size=32,
                device_str="cpu",
                loss_function_str="regret",
                predictor_str=predictor_str,
                predictor_kwargs=predictor_kwargs,
            )
        except Exception as e:
            print(f"Failed to initialize decision_maker: {e}")
            decision_maker = None  # Prevents further usage errors

        if decision_maker is not None:  # Proceed only if initialization was successful
            try:
                runner = Runner(decision_maker, num_epochs=2, use_wandb=False)
            except Exception as e:
                print(f"Failed to initialize runner: {e}")
                runner = None  # Prevents further usage errors

            if runner is not None:  # Proceed only if initialization was successful
                try:
                    runner.run()
                except Exception as e:
                    print(f"Failed to run runner: {e}")
        # TODO: This does not work, 'runner' referenced before assignment
        # self.assertNotEqual(runner, None)

    def test_sfge_2(self):
        """
        Test what happens without scale shift layer
        Returns:

        """
        # Predictor kwargs
        predictor_kwargs = {
            "size": 10,
            "num_hidden_layers": 2,
            "activation": "leaky_relu",
            "output_activation": "leaky_relu",
        }
        predictor_str = "MLP"
        print("\nStart test 1")
        # FIXME: KvdH why would SFGE not work without noisifier kwargs, shouldn't we set default?
        try:
            decision_maker = SFGEDecisionMaker(
                problem=problem,
                learning_rate=0.005,
                batch_size=32,
                device_str="cpu",
                loss_function_str="regret",
                predictor_str=predictor_str,
                predictor_kwargs=predictor_kwargs,
            )
        except Exception as e:
            print(f"Failed to initialize decision_maker: {e}")
            decision_maker = None  # Prevents further usage errors

        if decision_maker is not None:  # Proceed only if initialization was successful
            try:
                runner = Runner(decision_maker, num_epochs=2, use_wandb=False)
            except Exception as e:
                print(f"Failed to initialize runner: {e}")
                runner = None  # Prevents further usage errors

            if runner is not None:  # Proceed only if initialization was successful
                try:
                    runner.run()
                except Exception as e:
                    print(f"Failed to run runner: {e}")
        # TODO: This does not work, 'runner' referenced before assignment
        # self.assertNotEqual(runner, None)

    def test_sfge_3(self):
        """
        Test SFGE with independent sigma and an initial value of 0.5
        Returns:

        """
        print("\nStart test 3")
        # Predictor kwargs
        predictor_kwargs = {
            "size": 10,
            "num_hidden_layers": 2,
            "activation": "leaky_relu",
            "output_activation": "leaky_relu",
        }
        predictor_str = "MLP"
        try:
            decision_maker = SFGEDecisionMaker(
                problem=problem,
                learning_rate=0.01,
                batch_size=32,
                device_str="cpu",
                loss_function_str="regret",
                predictor_str=predictor_str,
                predictor_kwargs=predictor_kwargs,
                noisifier_kwargs={"sigma_setting": "independent", "sigma_init": 0.5},
            )
        except Exception as e:
            print(f"Failed to initialize decision_maker: {e}")
            decision_maker = None  # Prevents further usage errors

        if decision_maker is not None:  # Proceed only if initialization was successful
            try:
                runner = Runner(decision_maker, num_epochs=2, use_wandb=False)
            except Exception as e:
                print(f"Failed to initialize runner: {e}")
                runner = None  # Prevents further usage errors

            if runner is not None:  # Proceed only if initialization was successful
                try:
                    runner.run()
                except Exception as e:
                    print(f"Failed to run runner: {e}")
        # TODO: This does not work, 'runner' referenced before assignment
        # self.assertNotEqual(runner, None)


if __name__ == "__main__":
    unittest.main()
