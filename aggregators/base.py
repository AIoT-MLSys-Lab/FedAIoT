from typing import Union

import numpy as np
import torch

from aggregators.optreo import OptRepo


class FederatedAveraging:
    def __init__(self, global_model: torch.nn.Module,
                 server_optimizer='sgd',
                 server_lr=1e-2,
                 server_momentum=0.9,
                 eps=1e-3):

        self.global_model = global_model
        self.server_optimizer = server_optimizer
        self.server_lr = server_lr
        self.optimizer = OptRepo.name2cls(self.server_optimizer)(
            filter(lambda p: p.requires_grad, global_model.parameters()),
            lr=server_lr,
        )

    def step(self,
             updated_parameter_list: list[dict[str:np.array]],
             weights: Union[None | list[float]],
             round_idx: int = 0):
        self.optimizer.zero_grad()

        params_n_plus_1 = self._average_updates(updated_parameter_list, weights)
        named_params = dict(self.global_model.cpu().named_parameters())
        state_n_plus_1 = self.global_model.cpu().state_dict()
        with torch.no_grad():
            for parameter_name, parameter_n_plus_1 in params_n_plus_1.items():
                if parameter_name in named_params.keys():
                    parameter_n = named_params[parameter_name]
                    parameter_n.grad = parameter_n.data - parameter_n_plus_1.data
                else:
                    state_n_plus_1[parameter_name] = params_n_plus_1[parameter_name]
        self.global_model.load_state_dict(state_n_plus_1)
        self.optimizer.step()
        return self.global_model.cpu().state_dict()

    @staticmethod
    def _average_updates(update_list, weights=None):
        if weights is None:
            weights = [1 / len(update_list) for _ in range(len(update_list))]
        weights = np.array(weights, dtype=float)
        weights /= weights.sum()
        averaged_params = {k: v * weights[0] for k, v in update_list[0].items()}
        if len(update_list) > 1:
            for local_model_params, weight in zip(update_list[1:], weights[1:]):
                for k in averaged_params.keys():
                    averaged_params[k] += local_model_params[k] * weight
        return averaged_params
