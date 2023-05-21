import inspect

from torch import nn
from ultralytics.nn.tasks import DetectionModel

from models import widar, wisdm, ut_har, emognition, casas, energy, ego4d, epic_sounds

MODULE_MAP = {
    'wisdm_phone': wisdm,
    'wisdm_watch': wisdm,
    'widar': widar,
    'ut_har': ut_har,
    'emognition': emognition,
    'casas': casas,
    'energy': energy,
    'ego4d': ego4d,
    'epic_sounds': epic_sounds
}


def find_subclasses_and_factory_functions(module, parent_class):
    results = []

    for _, obj in inspect.getmembers(module):
        # Check if it's a class and a subclass of the parent_class
        if inspect.isclass(obj) and issubclass(obj, parent_class) and obj != parent_class:
            results.append(obj)
        # Check if it's a function
        elif inspect.isfunction(obj):
            try:
                # Get the function's return type annotation
                return_annotation = inspect.signature(obj).return_annotation

                # Check if the return type annotation is a subclass of the parent_class
                if inspect.isclass(return_annotation) and issubclass(return_annotation,
                                                                     parent_class) and return_annotation != parent_class:
                    results.append(obj)
            except (TypeError, ValueError, KeyError):
                # Ignore the function if the return type annotation is missing or not valid
                pass

    return results


def find_class_by_name(class_list, target_name):
    return next((cls for cls in class_list if cls.__name__ == target_name), None)


def load_model(model_name, trainer, dataset_name):
    if trainer == 'ultralytics':
        return DetectionModel(cfg=model_name)

    if dataset_name not in MODULE_MAP:
        raise ValueError('Dataset not supported')

    modules = find_subclasses_and_factory_functions(MODULE_MAP[dataset_name], nn.Module)
    model_cls = find_class_by_name(modules, model_name)

    if not model_cls:
        raise ValueError(f'No class found with the given name: {model_name}')

    return model_cls()


if __name__ == '__main__':
    model = load_model('UT_HAR_ResNet18', 'BaseTrainer', 'ut_har')
    print(model)
