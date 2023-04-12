from typing import List, Union, Type

import torch


class OptRepo:
    """A simplified utility class for working with subclasses of torch.optim.Optimizer."""

    @classmethod
    def get_opt_names(cls) -> List[str]:
        """Returns a list of supported optimizer names."""
        return [x.__name__.lower() for x in torch.optim.Optimizer.__subclasses__()]

    @classmethod
    def name2cls(cls, name: str) -> Type[torch.optim.Optimizer]:
        """Returns the optimizer class belonging to the name."""
        opt_class = next((x for x in torch.optim.Optimizer.__subclasses__() if x.__name__.lower() == name.lower()),
                         None)
        if not opt_class:
            raise KeyError(f"Invalid optimizer: {name}! Available optimizers: {cls.get_opt_names()}")
        return opt_class

    @classmethod
    def supported_parameters(cls, opt: Union[str, torch.optim.Optimizer]) -> List[str]:
        """Returns a list of __init__ function parameters of an optimizer."""
        opt_class = cls.name2cls(opt) if isinstance(opt, str) else opt
        params = opt_class.__init__.__code__.co_varnames
        return [p for p in params if p not in {"defaults", "self", "params"}]
