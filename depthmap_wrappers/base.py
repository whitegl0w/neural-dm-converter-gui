from abc import ABC, abstractmethod

from depthmap_wrappers.models import Model


class BaseDmWrapper(ABC):
    @abstractmethod
    def prepare_model(self, model: Model, *args): ...

    @abstractmethod
    def process(self, image, *args): ...
