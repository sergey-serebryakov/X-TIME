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
import json
import typing as t

import pandas as pd
from xgboost import XGBClassifier, XGBModel

from xtime.estimators.estimator import Estimator

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
        self._estimator = Estimator()
        self._estimator.model = model
        self._trees = pd.DataFrame(
            [
                TreeTraversal().traverse(json.loads(n)).as_dict()
                for n in model.get_booster().get_dump(dump_format="json")
            ]
        )
        self._num_rounds = self._trees.shape[0]
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
        return "boosting"

    @property
    def ensemble_size(self) -> int:
        return self._num_rounds

    @property
    def model_size(self) -> int:
        return self._num_trees_per_round

    @property
    def num_trees(self) -> int:
        return self.ensemble_size * self.model_size

    @property
    def estimator(self) -> Estimator:
        return self._estimator

    def describe(self, num_models: int = 0) -> t.Dict:
        """Compute some basic statistics of an XGBoost model.

        Args:
            model_dir: Directory that contains an XGBoost model.
            task_type: Task type this model was trained for (need to be able to load the right model).
            num_trees: Number of trees to use to compute statistics. For multi-class classification problems (num classes
                > 2), this must be `num_rounds * num_classes`. The value of `0` means use all trees.
            cache: If this function needs to be called multiple times, if the cache is not None, the function will cache
                the trees data frame in this cache under the "trees" key.
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
