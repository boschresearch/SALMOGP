"""
// Copyright (c) 2022 Robert Bosch GmbH
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published
// by the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
from typing import Union
import numpy as np

class history_manager_dataExclusive():
    
    def history_init(self, length: int, data_args: Union[list, np.ndarray] = None):
        self.history = {
            "training_time": [],
            "point_selection_time": [],
            "data_history": np.zeros(length, dtype=int),
            "data_args": np.array([None] * length) if data_args is None else data_args
            }# data_args: args in the dataset
    
    def history_update(
        self, data_args: Union[list, np.ndarray]
    ):
        self.history["data_history"] = np.append(
                    self.history["data_history"],
                    [self.history["data_history"][-1]+1] * np.shape(data_args)[-1])
        print(data_args)
        self.history["data_args"] = np.hstack((self.history["data_args"], data_args))


class history_manager_dataInclusive(history_manager_dataExclusive):
    r"""
    the model need to have self.num_data
    """
    def history_init(self, data_args: Union[list, np.ndarray]=None):
        super().history_init(self.num_data, data_args)