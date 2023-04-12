from typing import List, Union, Type

import torch


class TorchRepo:
    """A simplified utility class for working with subclasses of torch.optim.Optimizer."""

    @classmethod
    def get_names(cls, module) -> List[str]:
        """Returns a list of supported optimizer names."""
        return [x.__name__.lower() for x in module.__subclasses__()]

    @classmethod
    def name2cls(cls, name: str, module):
        """Returns the optimizer class belonging to the name."""
        opt_class = next((x for x in module.__subclasses__() if x.__name__.lower() == name.lower()),
                         None)
        if not opt_class:
            raise KeyError(f"Invalid optimizer: {name}! Available optimizers: {cls.get_names(module)}")
        return opt_class

    @classmethod
    def supported_parameters(cls, opt: Union[str, torch.optim.Optimizer], module) -> List[str]:
        """Returns a list of __init__ function parameters of an optimizer."""
        opt_class = cls.name2cls(opt, module) if isinstance(opt, str) else opt
        params = opt_class.__init__.__code__.co_varnames
        return [p for p in params if p not in {"defaults", "self", "params"}]


if __name__ == '__main__':
    print(TorchRepo.get_names(torch.optim.Optimizer))
    print(TorchRepo.get_names(torch.optim.lr_scheduler._LRScheduler))
    print(TorchRepo.name2cls("adam", torch.optim.Optimizer))
    print(TorchRepo.name2cls("linearlr", torch.optim.lr_scheduler._LRScheduler))
    print(TorchRepo.supported_parameters("adam", torch.optim.Optimizer))
    print(TorchRepo.supported_parameters(torch.optim.Adam, torch.optim.Optimizer))
    print(TorchRepo.supported_parameters(torch.optim.lr_scheduler.StepLR, torch.optim.lr_scheduler._LRScheduler))