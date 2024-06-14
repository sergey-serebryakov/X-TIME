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
import ctypes
import os
import shutil
import typing as t
import orjson as json
import uuid

import treelite
from cuml import ForestInference
from cuml.ensemble import RandomForestClassifier
import tempfile
from pathlib import Path
from treelite.frontend import Model as TreeliteModel

from xtime.datasets import Dataset
from xtime.estimators import Estimator
import random
import pandas as pd
import numpy as np


__all__ = ["TreeModel"]


class TreeTraversal:
    def __init__(self) -> None:
        self.depth = 0
        """Depth of a tree."""
        self.num_nodes = 0
        """Number of nodes in a tree (including leaves)."""
        self.num_leaves = 0
        """Number of leaves in a tree."""

    def traverse(self, tree: t.Dict) -> "TreeTraversal":
        """..."""
        # Init values
        self.num_leaves = len([node for node in tree["nodes"] if "leaf_value" in node])
        self.num_nodes = tree["num_nodes"]

        # Perform some basic checks so that I understand the structure of the tree
        assert tree["num_nodes"] == len(tree["nodes"]), \
            f"Unexpected tree structure: num_nodes={tree['num_nodes']}, len(nodes)={len(tree['nodes'])}"
        for idx, node in enumerate(tree["nodes"]):
            assert idx == node["node_id"], f"Unexpected node ID: idx={idx}, node_id={node['node_id']}."

        self.depth = self._compute_tree_depth(tree["nodes"], 0)
        return self

    def __repr__(self) -> str:
        return f"TreeTraversal(depth={self.depth}, num_leaves={self.num_leaves}, num_nodes={self.num_nodes})"

    def as_dict(self, prefix: str = "") -> t.Dict:
        return {
            f"{prefix}depth": self.depth,
            f"{prefix}num_leaves": self.num_leaves,
            f"{prefix}num_nodes": self.num_nodes,
        }

    def _compute_tree_depth(self, nodes: t.List[t.Dict], node_id: int) -> int:
        node: t.Dict = nodes[node_id]
        assert node["node_id"] == node_id, "Unexpected node ID"
        assert 0 <= node_id < len(nodes), "Unexpected node ID"

        if "leaf_value" in node:
            assert "left_child" not in node and "right_child" not in node, "Unexpected node structure"
            return 1

        assert "left_child" in node and "right_child" in node, "Unexpected node structure"
        return 1 + max(
            self._compute_tree_depth(nodes, node["left_child"]),
            self._compute_tree_depth(nodes, node["right_child"])
        )


class TreeModelTmp:
    """Compute statistics related to cuml RandomForest models.

    Args:
        model: An instance of the cuml.RandomForest model.
    """
    def __init__(self, model: RandomForestClassifier) -> None:
        self._checkpoint_file = Path("/dev/shm") / f"model-{uuid.uuid4()}.tl"
        model.convert_to_treelite_model().to_treelite_checkpoint(self._checkpoint_file.as_posix())

        treelite_model: TreeliteModel = TreeliteModel.deserialize(self._checkpoint_file.as_posix())
        model: t.Dict = json.loads(treelite_model.dump_as_json(pretty_print=False))
        trees: t.List[t.Dict] = model.pop("trees")

        self._ensemble_size = len(trees)

        # This seems to be working, need this to be able to create fil.ForestInference from json
        for tree in trees:
            tree["root_id"] = 0

        self._max_depths: t.List[int] = [0] * self.ensemble_size  # Cumulative max depths of models.
        self._max_leaves: t.List[int] = [0] * self.ensemble_size  # Cumulative max leaves of models.

        traverser = TreeTraversal()
        if self.ensemble_size > 0:
            _ = traverser.traverse(trees[0])
            self._max_depths[0], self._max_leaves[0] = traverser.depth, traverser.num_leaves

            for idx, tree in enumerate(trees[1:]):
                _ = traverser.traverse(tree)
                self._max_depths[idx] = max(traverser.depth, self._max_depths[idx-1])
                self._max_leaves[idx] = max(traverser.num_leaves, self._max_leaves[idx - 1])

        del trees, model, treelite_model

    def __del__(self) -> None:
        if self._checkpoint_file.exists():
            self._checkpoint_file.unlink()

    @property
    def ensemble(self) -> t.Optional[str]:
        return "bagging"

    @property
    def ensemble_size(self) -> int:
        # Same # estimators for binary and multi-class models
        return self._ensemble_size

    @property
    def model_size(self) -> int:
        """cuml.RandomForest uses one tree for multi-class problems."""
        num_trees_per_round = 1
        return num_trees_per_round

    @property
    def num_trees(self) -> int:
        return self.ensemble_size

    def evaluate(self, dataset: Dataset, num_models: int) -> t.Dict:
        """Evaluate sub-ensemble on the given dataset.

        Args:
            dataset: The dataset to evaluate the model on.
            num_models: Size of the sub-ensemble to use. If num_samples is 1, it will be first models.
        """

        treelite_model: TreeliteModel = TreeliteModel.deserialize(self._checkpoint_file.as_posix())
        treelite_model.set_tree_limit(num_models)

        _estimator = Estimator()
        _estimator.model = ForestInference().load_from_treelite_model(
            treelite_model,
            output_class=dataset.metadata.task.type.classification()
        )

        metrics: t.Dict = _estimator.evaluate(dataset)

        del _estimator, treelite_model
        return metrics

    def describe(self, num_models: int = 0) -> t.Dict:
        """Return maximal depth and maximal number of leaves for an ensemble of the specified size.

        Args:
            num_models: Size of the ensemble.
        """
        if num_models <= 0:
            num_models = self.ensemble_size
        num_trees = min(num_models, self.ensemble_size)
        return {
            "max_depth": self._max_depths[num_trees - 1],
            "max_leaves": self._max_leaves[num_trees - 1],
            "num_trees": num_trees,
        }


class _DatasetSplit(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_float)),
        ("num_examples", ctypes.c_size_t),
        ("num_features", ctypes.c_size_t),
        ("outputs", ctypes.POINTER(ctypes.c_float)),
        ("num_outputs", ctypes.c_size_t)
    ]

    def set(self, data: pd.DataFrame, num_trees: int, output_size: int) -> np.ndarray:
        assert isinstance(data, pd.DataFrame), \
            f"Unexpected data type (type={type(data)})."
        assert isinstance(num_trees, int) and num_trees > 0, \
            f"Unexpected num_trees (type={type(num_trees)}, num_trees={num_trees})."
        assert isinstance(output_size, int) and output_size > 0, \
            f"Unexpected num_classes (type={type(output_size)}, output_size={output_size})."

        raw_data: np.ndarray = data.values.flatten()
        assert raw_data.dtype == np.dtype(np.float32), f"Unexpected feature types (dtype={raw_data.dtype})."

        self.data = np.ctypeslib.as_ctypes(raw_data)
        self.num_examples = data.shape[0]
        self.num_features = data.shape[1]

        output_size = num_trees * self.num_examples * output_size
        outputs_buf = np.zeros(shape=(output_size,), dtype=np.float32)
        self.outputs = np.ctypeslib.as_ctypes(outputs_buf)
        self.num_outputs = output_size

        print(
            f"_DatasetSplit data=(dtype={raw_data.dtype}, shape={raw_data.shape}, num_examples={self.num_examples}, "
            f"num_features={self.num_features}), outputs_buf=(dtype={outputs_buf.dtype}, "
            f"shape={outputs_buf.shape}), num_trees={num_trees}, num_classes={output_size}."
        )

        return outputs_buf


class TreeModelV02:

    class PredictModel:
        def __init__(self, ids: t.Dict[int, str], probas: t.Dict[str, np.ndarray]) -> None:
            self._ids = ids
            self._probas = probas

        def predict_proba(self, x: t.Any, num_trees: int = -1) -> np.ndarray:
            id_x = id(x)
            if id_x not in self._ids:
                raise ValueError(f"Unknown split (id(x) = {id_x}).")
            split: str = self._ids[id_x]

            split_probas = self._probas[split]  # shape - (self.ensemble_size, -1, self._num_classes)
            if num_trees <= 0:
                num_trees = split_probas.shape[0]

            probas: np.ndarray = split_probas[num_trees-1, :, :]
            assert probas.ndim == 2, f"Internal bug - unexpected shape: {probas.shape}"
            return probas

    def __init__(self, model: RandomForestClassifier) -> None:
        # Create an empty working directory for this model (that corresponds to one trial / training run).
        self._work_dir = Path(f"/dev/shm/xtime/{os.getpid()}/{uuid.uuid4()}")
        if self._work_dir.exists():
            shutil.rmtree(self._work_dir)
        self._work_dir.mkdir(parents=True, exist_ok=True)

        # Get TreeLite model that provides API we need to convert this model to JSON
        model.convert_to_treelite_model().to_treelite_checkpoint((self._work_dir / "model.tl").as_posix())
        tl_model: TreeliteModel = treelite.Model.deserialize((self._work_dir / "model.tl").as_posix())
        (self._work_dir / "model.tl").unlink()

        # Save JSON model for interactions with `treelib` tool
        self.model_def: str = tl_model.dump_as_json(pretty_print=False)

        # Find parameters of sub-ensembles
        model_def: t.Dict = json.loads(self.model_def)
        trees_def: t.List[t.Dict] = model_def.pop("trees")

        self._ensemble_size = len(trees_def)
        self._num_classes = tl_model.num_class

        # This seems to be working, need this to be able to create fil.ForestInference from json
        for tree in trees_def:
            tree["root_id"] = 0

        self._max_depths: t.List[int] = [0] * self.ensemble_size  # Cumulative max depths of models.
        self._max_leaves: t.List[int] = [0] * self.ensemble_size  # Cumulative max leaves of models.

        self._predict_model: t.Optional[TreeModelV02.PredictModel] = None

        traverser = TreeTraversal()
        if self.ensemble_size > 0:
            _ = traverser.traverse(trees_def[0])
            self._max_depths[0], self._max_leaves[0] = traverser.depth, traverser.num_leaves

            for idx, tree in enumerate(trees_def[1:]):
                _ = traverser.traverse(tree)
                self._max_depths[idx] = max(traverser.depth, self._max_depths[idx - 1])
                self._max_leaves[idx] = max(traverser.num_leaves, self._max_leaves[idx - 1])

        # Free some memory?
        del tl_model, model_def, trees_def

    def __del__(self) -> None:
        if self._work_dir.exists():
            print(f"[V02::__del__] removing working directory: {self._work_dir}.")
            shutil.rmtree(self._work_dir)

    @property
    def ensemble(self) -> t.Optional[str]:
        return "bagging"

    @property
    def ensemble_size(self) -> int:
        # Same # estimators for binary and multi-class models
        return self._ensemble_size

    @property
    def model_size(self) -> int:
        """cuml.RandomForest uses one tree for multi-class problems."""
        num_trees_per_round = 1
        return num_trees_per_round

    @property
    def num_trees(self) -> int:
        return self.ensemble_size

    def evaluate(self, dataset: Dataset, num_models: int) -> t.Dict:
        if self._predict_model is None:
            self._explore_model_with_treelib(dataset)

        _estimator = Estimator()
        _estimator.model = self._predict_model

        metrics: t.Dict = _estimator.evaluate(
            dataset,
            predict_proba_kwargs={"num_trees": num_models}
        )

        del _estimator
        return metrics

    def _explore_model_with_treelib(self, dataset: Dataset) -> None:
        treelib_path = Path(_get_treelib_path())
        treelib_path = treelib_path.parent / f"lib{treelib_path.name}.so"
        treelib = ctypes.CDLL(treelib_path.as_posix())

        num_splits: int = len(dataset.splits)
        splits = (_DatasetSplit * num_splits)()

        process = treelib.process
        process.restype = None
        process.argtypes = [ctypes.c_char_p, _DatasetSplit * num_splits, ctypes.c_size_t]

        output_buffers: t.List[np.ndarray] = []
        for idx, (split_name, split_data) in enumerate(dataset.splits.items()):
            output_buf = splits[idx].set(split_data.x, num_trees=self._ensemble_size, output_size=self._num_classes)
            output_buffers.append(output_buf)
        process(self.model_def.encode("utf-8"), splits, num_splits)

        probas = {}
        ids: t.Dict[int, str] = {}
        for idx, split_name in enumerate(dataset.splits.keys()):
            try:
                # self.num_examples * num_trees * num_classes
                probas[split_name] = output_buffers[idx].reshape(self.ensemble_size, -1, self._num_classes)
            except ValueError as err:
                print("[_explore_model_with_treelib] " + str(err))
                print(
                    f"[_explore_model_with_treelib] split={split_name}, x={dataset.splits[split_name].x.shape}, "
                    f"split_probas={splits[idx].outputs_buf.shape}, "
                    f"new_shape={(self.ensemble_size, -1, self._num_classes)}."
                )
                raise
            ids[id(dataset.splits[split_name].x)] = split_name

        self._predict_model = TreeModelV02.PredictModel(ids, probas)

    def describe(self, num_models: int = 0) -> t.Dict:
        """Return maximal depth and maximal number of leaves for an ensemble of the specified size.

        Args:
            num_models: Size of the ensemble.
        """
        if num_models <= 0:
            num_models = self.ensemble_size
        num_trees = min(num_models, self.ensemble_size)
        return {
            "max_depth": self._max_depths[num_trees - 1],
            "max_leaves": self._max_leaves[num_trees - 1],
            "num_trees": num_trees,
        }


class TreeModelV01:
    """Compute statistics related to cuml RandomForest models.

    Args:
        model: An instance of the cuml.RandomForest model.
    """
    def __init__(self, model: RandomForestClassifier) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_file = (Path(temp_dir) / "model.tl").as_posix()
            model.convert_to_treelite_model().to_treelite_checkpoint(checkpoint_file)
            treelite_model: TreeliteModel = TreeliteModel.deserialize(checkpoint_file)

        self._model: t.Dict = json.loads(treelite_model.dump_as_json(pretty_print=False))
        self._trees: t.List[t.Dict] = self._model.pop("trees")

        # This seems to be working, need this to be able to create fil.ForestInference from json
        for tree in self._trees:
            tree["root_id"] = 0

        self._max_depths: t.List[int] = [0] * self.ensemble_size  # Cumulative max depths of models.
        self._max_leaves: t.List[int] = [0] * self.ensemble_size  # Cumulative max leaves of models.

        traverser = TreeTraversal()
        if self.ensemble_size > 0:
            _ = traverser.traverse(self._trees[0])
            self._max_depths[0], self._max_leaves[0] = traverser.depth, traverser.num_leaves

            for idx, tree in enumerate(self._trees[1:]):
                _ = traverser.traverse(tree)
                self._max_depths[idx] = max(traverser.depth, self._max_depths[idx-1])
                self._max_leaves[idx] = max(traverser.num_leaves, self._max_leaves[idx - 1])

    @property
    def ensemble(self) -> t.Optional[str]:
        return "bagging"

    @property
    def ensemble_size(self) -> int:
        # Same # estimators for binary and multi-class models
        return len(self._trees)

    @property
    def model_size(self) -> int:
        """cuml.RandomForest uses one tree for multi-class problems."""
        num_trees_per_round = 1
        return num_trees_per_round

    @property
    def num_trees(self) -> int:
        return self.ensemble_size

    def evaluate(self, dataset: Dataset, num_models: int, num_samples: int = 1) -> t.Dict:
        """Evaluate sub-ensemble on the given dataset.

        Args:
            dataset: The dataset to evaluate the model on.
            num_models: Size of the sub-ensemble to use. If num_samples is 1, it will be first models.
            num_samples: If greater than 1, average performance metrics over num_samples random sub-ensembles. This does
                not work now with the implementation of the describe method.
        """

        def _estimate(_trees: t.List[t.Dict]) -> t.Dict:
            _model_spec = self._model.copy()
            _model_spec["trees"] = _trees

            _estimator = Estimator()
            _estimator.model = ForestInference().load_from_treelite_model(
                TreeliteModel.import_from_json(json.dumps(_model_spec).decode()),
                output_class=dataset.metadata.task.type.classification()
            )

            return _estimator.evaluate(dataset)

        if num_samples == 1:
            return _estimate(self._trees[0:num_models])
        else:
            df = pd.DataFrame([
                _estimate(random.sample(self._trees, num_models)) for _ in range(num_samples)
            ])
            return {metric: df[metric].mean() for metric in df.columns}

    def describe(self, num_models: int = 0) -> t.Dict:
        """Return maximal depth and maximal number of leaves for an ensemble of the specified size.

        Args:
            num_models: Size of the ensemble.
        """
        if num_models <= 0:
            num_models = self.ensemble_size
        num_trees = min(num_models, self.ensemble_size)
        return {
            "max_depth": self._max_depths[num_trees - 1],
            "max_leaves": self._max_leaves[num_trees - 1],
            "num_trees": num_trees,
        }


TreeModel = TreeModelV02


def _save_dataset_as_csv(dataset: Dataset, work_dir: Path) -> t.List[str]:
    split_files: t.List[str] = []
    for split_name, split_data in dataset.splits.items():
        split_files.append((work_dir / f"{split_name}.csv").as_posix())
        split_data.x.to_csv(split_files[-1], index=False, sep=",")
    return split_files


def _get_treelib_path() -> str:
    # contrib -> xtime -> training
    treelib = Path(__file__).parent.parent.parent / "tools" / "treelib" / "build" / "treelib"
    if treelib.is_file():
        return treelib.as_posix()
    return "treelib"


def _get_directory_items(path: Path) -> t.List[str]:
    return [item.name for item in path.iterdir()]
