#!/usr/bin/env python
# Copyright 2025 The Grail Simulation Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""XGBoost-facing re-exports of the shared dataset helpers."""

from __future__ import annotations

from knn.core import data as knn_data_module

DEFAULT_DATASET_SOURCE = knn_data_module.DEFAULT_DATASET_SOURCE
EVAL_SPLIT = knn_data_module.EVAL_SPLIT
PROMPT_COLUMN = knn_data_module.PROMPT_COLUMN
PROMPT_MAX_HISTORY = knn_data_module.PROMPT_MAX_HISTORY
SOLUTION_COLUMN = knn_data_module.SOLUTION_COLUMN
TRAIN_SPLIT = knn_data_module.TRAIN_SPLIT
filter_dataset_for_issue = knn_data_module.filter_dataset_for_issue
filter_dataset_for_participant_studies = knn_data_module.filter_dataset_for_participant_studies
filter_split_for_participant_studies = knn_data_module.filter_split_for_participant_studies
issues_in_dataset = knn_data_module.issues_in_dataset
load_dataset_source = knn_data_module.load_dataset_source

__all__ = list(knn_data_module.__all__)
