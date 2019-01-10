PYTORCH = "pytorch"
"""
Constant for `pytorch` framework.
"""
TENSORFLOW = "tensorflow"
"""
Constant for `tensorflow` framework.
"""
TRAIN_TAG = "TRAIN"
VALID_TAG = "VALID"
TEST_TAG = "TEST"

from .argument import Argument
from .trainer import Trainer
from .config import Config
from .logger import Logger
from .model_pytorch import BaseModel
from .tester import Tester
from .dataset_loader import DatasetLoader
from .evaluator import Evaluator
from .metrics_comparator import MetricsComparator
