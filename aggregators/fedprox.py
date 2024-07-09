from typing import Union

import numpy as np
import torch

from aggregators.base import FederatedAveraging


class FederatedProximal(FederatedAveraging):
    def __init__(self, global_model: torch.nn.Module,
                 server_optimizer='sgd',
                 server_lr=1e-2,
                 server_momentum=0.9,
                 mu=1e-2,  # Proximal term coefficient
                 eps=1e-3):

        super().__init__(global_model, server_optimizer, server_lr, server_momentum, eps)
        self.mu = mu

    def step(self,
             updated_parameter_list: list[dict[str, np.array]],
             weights: Union[None, list[float]],
             round_idx: int = 0):
        self.optimizer.zero_grad()

        params_n_plus_1 = self._average_updates(updated_parameter_list, weights)
        named_params = dict(self.global_model.cpu().named_parameters())
        state_n_plus_1 = self.global_model.cpu().state_dict()
        with torch.no_grad():
            for parameter_name, parameter_n_plus_1 in params_n_plus_1.items():
                if parameter_name in named_params.keys():
                    parameter_n = named_params[parameter_name]
                    parameter_n.grad = parameter_n.data - parameter_n_plus_1.data + self.mu * (
                                parameter_n.data - parameter_n_plus_1.data)
                else:
                    state_n_plus_1[parameter_name] = params_n_plus_1[parameter_name]
        self.global_model.load_state_dict(state_n_plus_1)
        self.optimizer.step()
        return self.global_model.cpu().state_dict()
