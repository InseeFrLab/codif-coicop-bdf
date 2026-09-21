"""MLflow 'models from code' definition for the multi-head classifier."""

import mlflow

from .mlflow_utils import MultiHeadCOICOPPyfuncWrapper

mlflow.models.set_model(MultiHeadCOICOPPyfuncWrapper())
