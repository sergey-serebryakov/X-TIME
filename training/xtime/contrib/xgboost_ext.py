###
# Copyright (2023) Hewlett Packard Enterprise Development LP
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###
import time
import json
import typing as t

import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBModel, XGBRFRegressor, DMatrix

from xtime.datasets import Dataset
from xtime.estimators.estimator import Estimator
from numba import jit


__all__ = ["TreeTraversal", "TreeModel"]


class TreeTraversal:
    """Traverse dictionary representing one XGBoost tree and collect some descriptive statistics."""

    def __init__(self) -> None:
        self.depth = 0
        """Depth of a tree."""
        self.num_nodes = 0
        """Number of nodes in a tree (including leaves)."""
        self.num_leaves = 0
        """Number of leaves in a tree."""

    def traverse(self, node: t.Dict) -> "TreeTraversal":
        """Traverse the tree starting with `node` node."""
        self.depth = 0
        self.num_leaves = 0
        self.num_nodes = 0
        self._traverse(node)
        self.depth += 1  # Leaf nodes do not have `depth` field.
        return self

    def __repr__(self) -> str:
        return f"TreeTraversal(depth={self.depth}, num_leaves={self.num_leaves}, num_nodes={self.num_nodes})"

    def as_dict(self, prefix: str = "") -> t.Dict:
        return {
            f"{prefix}depth": self.depth,
            f"{prefix}num_leaves": self.num_leaves,
            f"{prefix}num_nodes": self.num_nodes,
        }

    def _traverse(self, node: t.Dict) -> None:
        """Recursively traverse the tree."""
        if not isinstance(node, dict) or "nodeid" not in node:
            return
        self.num_nodes += 1
        if "depth" in node:
            self.depth = max(self.depth, node["depth"])
        if "leaf" in node:
            self.num_leaves += 1
        if isinstance(node.get("children", None), list):
            for child in node["children"]:
                self._traverse(child)


class TreeModel:
    def __init__(self, model: XGBModel) -> None:
        assert isinstance(model, XGBModel), f"Unexpected model type ({type(model)})."
        self._trees = pd.DataFrame(
            [
                TreeTraversal().traverse(json.loads(n)).as_dict()
                for n in model.get_booster().get_dump(dump_format="json")
            ]
        )
        if isinstance(model, XGBRFRegressor):
            self._impl = _XGBRFRegressorTreeModelImpl(model, self._trees)
        else:
            self._impl = _DefaultTreeModelImpl(model, self._trees)

    @property
    def ensemble(self) -> t.Optional[str]:
        """One of - `bagging` or `boosting`."""
        return self._impl.ensemble

    @property
    def ensemble_size(self) -> int:
        """Number of rounds (same as number of base models)."""
        return self._impl.ensemble_size

    @property
    def model_size(self) -> int:
        """Number of trees per one round (model).

        For multi-class classification problems (num classes > 2), this is equal to the number of classes. For all other
        cases (binary classification and uni-variate regression), it's one tree.
        """
        return self._impl.model_size

    @property
    def num_trees(self) -> int:
        """Total number of trees in the ensemble (not necessarily equal to number of gradient boosting rounds)."""
        return self.ensemble_size * self.model_size

    def evaluate(self, dataset: Dataset, num_models: int) -> t.Dict:
        return self._impl.evaluate(dataset, num_models)

    def describe(self, num_models: int = 0) -> t.Dict:
        """Compute some basic statistics of an XGBoost model.

        Args:
            num_models: Number of models to consider. If 0, then all models are considered. A model is a base estimator
                in the ensemble. It may contain one or multiple trees.
        Returns:
            Dictionary with some descriptive statistics of this model that include the following:
                - `max_depth`: maximal tree depth
                - `max_leaves`: maximal number of leaves
                - `num_trees`: total number of trees. If `num_trees` is specified, and it is smaller or equal
                  to actual number of trees, then number of trees will equal to this number.
        """
        if num_models <= 0:
            num_models = self.ensemble_size
        num_trees = min(num_models, self.ensemble_size) * self.model_size
        return {
            "max_depth": self._trees["depth"][0:num_trees].max(),
            "max_leaves": self._trees["num_leaves"][0:num_trees].max(),
            "num_trees": num_trees,
        }


@jit()
def _compute(outputs: np.ndarray, leaf_cache: np.ndarray, leaves: np.ndarray) -> None:
    num_trees: int = outputs.shape[0]
    num_examples: int = outputs.shape[1]

    for tree_id in range(num_trees):
        for example_id in range(num_examples):
            outputs[tree_id, example_id] = num_trees * leaf_cache[tree_id, leaves[tree_id, example_id]]


class _XGBRFRegressorTreeModelImpl:

    class _ModelWrapper:
        def __init__(self, predictions: np.ndarray, x_id: t.Any) -> None:
            # [TreesDim, ExamplesDim]
            self._predictions = predictions
            assert self._predictions.ndim == 2
            #
            self._x_id = x_id

        def predict(self, x: pd.DataFrame, num_models: int = -1) -> np.ndarray:
            if id(x) != self._x_id:
                raise ValueError("Unexpected input data.")
            if num_models <= 0:
                num_models = self._predictions.shape[0]
            assert x.shape[0] == self._predictions.shape[1], "Inconsistent # examples"
            return self._predictions[num_models - 1, :].flatten()

    def _build_estimator(self, dataset: Dataset) -> None:
        if len(dataset.splits) != 1 or "test" not in dataset.splits:
            raise ValueError("Only one split (test) is expected.")
        self._estimator = Estimator()

        dm = DMatrix(
            dataset.splits["test"].x,
            enable_categorical=dataset.metadata.has_categorical_features()
        )
        num_examples = dataset.splits["test"].x.shape[0]

        # The `leaves` is a numpy array of shape [ExamplesDim, TreesDim] -> [TreesDim, ExamplesDim]
        leaves = self._model.get_booster().predict(dm, pred_leaf=True).astype("int32").transpose()
        # We need array of shape [TreesDim, ExamplesDim]
        st = time.time()
        outputs = np.zeros(
            (self._num_trees, num_examples),
            dtype="float"
        )
        print(f"Outputs initialized in {time.time() - st} seconds (shape={outputs.shape}, isnan={np.isnan(outputs).any()}).")

        # Create leaves cache for fast access
        st = time.time()
        trees: pd.DataFrame = self._model.get_booster().trees_to_dataframe()
        if trees.Gain.isnull().any():
            raise ValueError("Gain column contains NULL values.")
        # Shape - (TreeDim, NodeDim)
        leaf_cache: np.ndarray = trees.pivot(index="Tree", columns="Node", values="Gain").values
        print(f"Leaves cache created in {time.time() - st} seconds.")

        # Compute output of each tree
        st = time.time()
        _compute(outputs, leaf_cache, leaves)
        print(f"Partial outputs computed in {time.time() - st} seconds (shape={outputs.shape}, isnan={np.isnan(outputs).any()}).")

        # Need to sum the outputs of all trees to get partial predictions of that ensemble
        st = time.time()
        outputs = np.cumsum(outputs, axis=0)
        print(f"Partial outputs summed in {time.time() - st} seconds (shape={outputs.shape}, isnan={np.isnan(outputs).any()}).")

        # Now we just average and add global bias (aka base score).
        st = time.time()
        outputs = self._global_bias + outputs / np.arange(1, self._num_trees + 1).reshape(-1, 1)
        print(f"Partial outputs averaged and bias added in {time.time() - st} seconds (shape={outputs.shape}, isnan={np.isnan(outputs).any()}).")

        # Create estimator
        self._estimator = Estimator()
        self._estimator.model = self._ModelWrapper(outputs, id(dataset.splits["test"].x))

    def __init__(self, model: XGBRFRegressor, trees: pd.DataFrame) -> None:
        self._model = model
        self._estimator = None
        self._num_trees = trees.shape[0]

        config = json.loads(model.get_booster().save_config())
        assert int(config["learner"]["gradient_booster"]["gbtree_model_param"]["num_parallel_tree"]) == self._num_trees
        assert int(config["learner"]["gradient_booster"]["gbtree_model_param"]["num_trees"]) == self._num_trees

        self._global_bias = float(config["learner"]["learner_model_param"]["base_score"])
        print(f"num_trees={self._num_trees}, global_bias={self._global_bias}")

    @property
    def ensemble(self) -> t.Optional[str]:
        """One of - `bagging` or `boosting`."""
        return "bagging"

    @property
    def ensemble_size(self) -> int:
        """Number of rounds (same as number of base models)."""
        return self._num_trees

    @property
    def model_size(self) -> int:
        """Number of trees per one round (model).

        For multi-class classification problems (num classes > 2), this is equal to the number of classes. For all other
        cases (binary classification and uni-variate regression), it's one tree.
        """
        return 1

    def evaluate(self, dataset: Dataset, num_models: int) -> t.Dict:
        if self._estimator is None:
            self._build_estimator(dataset)
        return self._estimator.evaluate(dataset, predict_kwargs={"num_models": num_models})


class _DefaultTreeModelImpl:
    def __init__(self, model: XGBModel, trees: pd.DataFrame) -> None:
        self._estimator = Estimator()
        self._estimator.model = model

        self._num_rounds = trees.shape[0]
        self._num_trees_per_round = 1

        if isinstance(model, XGBClassifier):
            num_classes: int = getattr(model, "n_classes_", None)
            if num_classes is None:
                raise ValueError("Number of classes is None for XGBClassifier (unfitted model?).")
            if num_classes > 2:
                self._num_trees_per_round = num_classes
                self._num_rounds = self._num_rounds // self._num_trees_per_round

    @property
    def ensemble(self) -> t.Optional[str]:
        """One of - `bagging` or `boosting`."""
        return "boosting"

    @property
    def ensemble_size(self) -> int:
        """Number of rounds (same as number of base models)."""
        return self._num_rounds

    @property
    def model_size(self) -> int:
        """Number of trees per one round (model).

        For multi-class classification problems (num classes > 2), this is equal to the number of classes. For all other
        cases (binary classification and uni-variate regression), it's one tree.
        """
        return self._num_trees_per_round

    def evaluate(self, dataset: Dataset, num_models: int) -> t.Dict:
        if dataset.metadata.task.type.classification():
            kwargs = {"predict_proba_kwargs": {"iteration_range": (0, num_models)}}
        else:
            kwargs = {"predict_kwargs": {"iteration_range": (0, num_models)}}
        return self._estimator.evaluate(
            dataset, **kwargs
        )


def main() -> None:
    from pathlib import Path
    from xtime.estimators.estimator import Model

    trial_path = Path(
        "/lustre/data/mlflow/mlruns/2/e456d76b969e492ab36c5144defa03a4/artifacts/ray_tune/fit_18f81_01973_1973_colsample_bylevel=0.6898,colsample_bytree=0.5084,gamma=0.0001,max_depth=4,max_leaves=256,min_child_weight=39._2024-06-04_15-15-17"
    )
    xgb_model = Model.load_model(trial_path)
    tree_model = TreeModel(xgb_model)

    dataset = Dataset.create("year_prediction_msd:default")
    if "train" in dataset.splits:
        del dataset.splits["train"]
    if "valid" in dataset.splits:
        del dataset.splits["valid"]

    eval_result = tree_model.evaluate(dataset, 10)
    print(eval_result)


if __name__ == "__main__":
    main()
