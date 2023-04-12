from typing import List, Union, Type

import torch


class TorchComponentRepository:
    """A utility class for working with subclasses of PyTorch components,
    such as torch.optim.Optimizer and torch.optim.lr_scheduler._LRScheduler."""

    @classmethod
    def get_supported_names(cls, module) -> List[str]:
        """Returns a list of supported component names."""
        return [component.__name__.lower() for component in module.__subclasses__()]

    @classmethod
    def get_class_by_name(cls, name: str, module):
        """Returns the component class corresponding to the given name."""
        component_class = next((component for component in module.__subclasses__()
                                if component.__name__.lower() == name.lower()), None)
        if not component_class:
            raise KeyError(f"Invalid component: {name}! Available components: {cls.get_supported_names(module)}")
        return component_class

    @classmethod
    def get_supported_parameters(cls, component: Union[str, Type], module=None) -> List[str]:
        """Returns a list of __init__ function parameters for a given component and module."""
        component_class = cls.get_class_by_name(component, module) if isinstance(component, str) else component
        params = component_class.__init__.__code__.co_varnames
        return [param for param in params if param not in {"defaults", "self", "params"}]


if __name__ == '__main__':
    print(TorchComponentRepository.get_supported_names(torch.optim.Optimizer))
    print(TorchComponentRepository.get_supported_names(torch.optim.lr_scheduler._LRScheduler))
    print(TorchComponentRepository.get_class_by_name("adam", torch.optim.Optimizer))
    print(TorchComponentRepository.get_class_by_name("linearlr", torch.optim.lr_scheduler._LRScheduler))
    print(TorchComponentRepository.get_supported_parameters(torch.optim.Adam))
    print(TorchComponentRepository.get_supported_parameters(torch.optim.lr_scheduler.StepLR))
    print(TorchComponentRepository.get_supported_parameters("adam", torch.optim.Optimizer))

