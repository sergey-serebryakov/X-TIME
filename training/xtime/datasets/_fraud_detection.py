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

import logging
import os
import typing as t
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from xtime.datasets import Dataset, DatasetBuilder, DatasetMetadata, DatasetSplit
from xtime.datasets.dataset import DatasetPrerequisites
from xtime.datasets.preprocessing import TimeSeriesEncoderV1
from xtime.errors import DatasetError
from xtime.ml import ClassificationTask, Feature, FeatureType, TaskType

__all__ = ["FDBuilder"]

logger = logging.getLogger(__name__)


_XTIME_DATASETS_FD = "XTIME_DATASETS_FD"
"""Environment variable that points to a directory with FD (Fraud Detection) dataset."""

_FD_HOME_PAGE = (
    "https://www.kaggle.com/datasets/volodymyrgavrysh/fraud-detection-bank-dataset-20k-records-binary?resource=download"  # noqa
)
"""Dataset home page."""

_FD_DATASET_FILE = "fraud_detection_bank_dataset.csv"
"""File containing raw (unprocessed) FD dataset that is located inside _XTIME_DATASETS_FD directory."""


class FDBuilder(DatasetBuilder):
    """FD: Fraud detection.

    Fraud Detection bank dataset 20K records binary classification.
    20k records of customer transactions with 112 features:
        https://www.kaggle.com/datasets/volodymyrgavrysh/fraud-detection-bank-dataset-20k-records-binary?resource=download
    """

    NAME = "fraud_detection"

    def __init__(self) -> None:
        super().__init__()
        self.builders.update(default=self._build_default_dataset)
        self.encoder = TimeSeriesEncoderV1()
        self._dataset_dir: t.Optional[Path] = None

    def _check_pre_requisites(self) -> None:
        # Check raw dataset exists.
        if _XTIME_DATASETS_FD not in os.environ:
            raise DatasetError.missing_prerequisites(
                f"No environment variable found ({_XTIME_DATASETS_FD}) that should point to a directory with "
                f"FD (Fraud Detection) dataset that can be downloaded from `{_FD_HOME_PAGE}`."
            )
        dataset_dir: Path = Path(os.environ[_XTIME_DATASETS_FD]).absolute()
        if dataset_dir.is_file():
            dataset_dir = dataset_dir.parent
        if not (dataset_dir / _FD_DATASET_FILE).is_file():
            raise DatasetError.missing_prerequisites(
                f"FD dataset location was identified as `{dataset_dir}`, but this is either not a directory "
                f"or dataset file (`{_FD_DATASET_FILE}`) not found in this location. Please, download v1.1 of this "
                f"dataset from its home page `{_FD_HOME_PAGE}`."
            )
        # Check `tsfresh` library can be imported.
        DatasetPrerequisites.check_tsfresh(self.NAME)
        #
        self._dataset_dir = dataset_dir

    def _build_default_dataset(self, **kwargs) -> Dataset:
        if kwargs:
            raise ValueError(f"{self.__class__.__name__}: `default` dataset does not accept arguments.")
        self._create_default_dataset()

        assert self._dataset_dir is not None, "Dataset directory is none."
        dataset_dir: Path = self._dataset_dir

        train_df = pd.read_csv(dataset_dir / (_FD_DATASET_FILE[0:-4] + "-default-train.csv"))
        test_df = pd.read_csv(dataset_dir / (_FD_DATASET_FILE[0:-4] + "-default-test.csv"))

        features = [
            Feature(col, FeatureType.CONTINUOUS, cardinality=int(train_df[col].nunique()))
            for col in train_df.columns
            if col != "targets"
        ]

        # Check that data frames contains expected columns (112 features and 1 is for label).
        assert (
            train_df.shape[1] == len(features) + 1
        ), f"Train data frame contains wrong number of columns (shape={train_df.shape})."
        assert (
            test_df.shape[1] == len(features) + 1
        ), f"Test data frame contains wrong number of columns (shape={test_df.shape})."

        label: str = "targets"

        dataset = Dataset(
            metadata=DatasetMetadata(
                name=FDBuilder.NAME,
                version="default",
                task=ClassificationTask(TaskType.BINARY_CLASSIFICATION, num_classes=2),
                features=features,
                properties={"source": dataset_dir.as_uri()},
            ),
            splits={
                DatasetSplit.TRAIN: DatasetSplit(x=train_df.drop(label, axis=1, inplace=False), y=train_df[label]),
                DatasetSplit.TEST: DatasetSplit(x=test_df.drop(label, axis=1, inplace=False), y=test_df[label]),
            },
        )
        return dataset

    def _create_default_dataset(self) -> None:
        """Create default train/test splits and save them to files.

        Input to this function is the clean dataset created by the `_clean_dataset` method of this class.
        """
        # Do not generate datasets if they have already been generated.
        assert self._dataset_dir is not None, "Dataset directory is none."
        dataset_dir: Path = self._dataset_dir

        default_train_dataset_file = dataset_dir / (_FD_DATASET_FILE[0:-4] + "-default-train.csv")
        default_test_dataset_file = dataset_dir / (_FD_DATASET_FILE[0:-4] + "-default-test.csv")
        if default_train_dataset_file.is_file() and default_test_dataset_file.is_file():
            return

        # Load clean dataset into a data frame (user_id,activity,timestamp,x,y,z)
        dataset_file = (dataset_dir / _FD_DATASET_FILE).with_suffix(".csv")
        assert dataset_file.is_file(), "Dataset file does not exist (this is internal error)."

        # df: pd.DataFrame = pd.read_csv(dataset_file, dtype=dtypes)
        df: pd.DataFrame = pd.read_csv(dataset_file)

        # Check for missing values
        assert not df.isna().any().any(), "There are missing values in the DataFrame"

        # Raw file has 114 columns (index, 112 features, labels `targets` )
        assert df.shape[1] == 114, f"Dataset expected to have 114 columns (shape={df.shape})."

        # Drop the first (index) column
        df = df.drop(df.columns[0], axis=1)

        # Split train and test dataframes
        df_train, df_test = train_test_split(df, test_size=0.2, random_state=0)

        df_train.to_csv(default_train_dataset_file, index=False)
        df_test.to_csv(default_test_dataset_file, index=False)
