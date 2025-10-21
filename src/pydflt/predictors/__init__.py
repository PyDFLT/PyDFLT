from .multilayer_normal import MLPNormalPredictor
from .multilayer_perceptron import MLPPredictor
from .multilayer_sample import SamplePredictor

IMPLEMENTED_PREDICTORS = {
    "MLP": MLPPredictor,
    "Normal": MLPNormalPredictor,
    "Sample": SamplePredictor,
}
