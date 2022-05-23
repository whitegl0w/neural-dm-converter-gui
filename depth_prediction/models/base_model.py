import torch
import torch.nn as nn


class BaseModel(torch.nn.Module):
    def load(self, path):
        """Load model from file.

        Args:
            path (str): file path
        """
        map_location = "cuda" if torch.cuda.is_available() else "cpu"
        parameters = torch.load(path, map_location=map_location)

        if "optimizer" in parameters:
            parameters = parameters["model"]

        self.load_state_dict(parameters)
