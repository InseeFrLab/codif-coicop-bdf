"""MLflow 'models from code' definition for the basic classifier."""

import mlflow

from .mlflow_utils import COICOPPyfuncWrapper

mlflow.models.set_model(COICOPPyfuncWrapper())
