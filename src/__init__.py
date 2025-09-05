import torch

from src.registries.data import data_registry as data_registry
from src.registries.data import get_data as get_data
from src.registries.data import register_data as register_data
from src.registries.decision_makers import (
    decision_maker_registry as decision_maker_registry,
)
from src.registries.decision_makers import make_decision_maker as make_decision_maker
from src.registries.decision_makers import (
    register_decision_maker as register_decision_maker,
)
from src.registries.models import make_model as make_model
from src.registries.models import model_registry as model_registry
from src.registries.models import register_model as register_model

from .dataset import DFLDataset as DFLDataset
from .logger import Logger as Logger
from .noisifier import Noisifier as Noisifier
from .problem import Problem as Problem
from .runner import Runner as Runner

torch.set_default_dtype(torch.float32)
