import json
import pandas as pd

class ParameterDict:
    def __init__(self, parameter_dict):
        for key, value in parameter_dict.items():
            setattr(self, key, value)