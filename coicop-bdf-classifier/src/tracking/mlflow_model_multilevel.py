"""MLflow 'models from code' definition for the multi-level classifier."""

import mlflow

from .mlflow_utils import MultilevelCOICOPPyfuncWrapper

mlflow.models.set_model(MultilevelCOICOPPyfuncWrapper())
