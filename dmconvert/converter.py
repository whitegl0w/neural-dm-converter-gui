import numpy.typing as npt
from dataclasses import dataclass
from typing import Callable, Optional

from depthmap_wrappers.base import BaseDmWrapper
from depthmap_wrappers.midas import MidasDmWrapper
from abc import ABC, abstractmethod
from depthmap_wrappers.models import Model

RED = 2
GREEN = 1
BLUE = 0


@dataclass
class DmMediaParams:
    fps: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    frame_count: Optional[int] = None


class ReaderError(RuntimeError):
    pass


class DmMediaReader(ABC):
    def is_ready(self) -> bool:
        return True

    @staticmethod
    @abstractmethod
    def display_name() -> str: ...

    @abstractmethod
    def prepare_and_get_params(self) -> DmMediaParams: ...

    @abstractmethod
    def data(self) -> Optional[npt.NDArray]: ...

    def close(self): ...


class DmMediaSeekableReader(DmMediaReader):
    @abstractmethod
    def seek(self, position_ms: int): ...

    @property
    @abstractmethod
    def duration(self) -> int: ...

    @property
    @abstractmethod
    def progress(self) -> int: ...

    def replay(self):
        self.seek(0)


class DmMediaWriter(ABC):
    @staticmethod
    @abstractmethod
    def display_name() -> str: ...

    def prepare(self, media_params: DmMediaParams): ...

    @abstractmethod
    def write(self, img: npt.NDArray, dm: npt.NDArray): ...

    def close(self): ...


class DmMediaConverter:
    preprocessors: list[Callable[[npt.NDArray], npt.NDArray]] = []
    postprocessors: list[Callable[[npt.NDArray, npt.NDArray], tuple[npt.NDArray, npt.NDArray]]] = []
    writers: list[DmMediaWriter] = []

    def __init__(self, model: Model, reader: DmMediaReader, model_loader: BaseDmWrapper):
        self._reader = reader
        self._model = model
        self._is_running = False
        self._wrapper = model_loader

    def start(self):
        self._is_running = True

        media_params = self._reader.prepare_and_get_params()
        for writer in self.writers:
            writer.prepare(media_params)

        self._wrapper.prepare_model(self._model)

        try:
            for img in self._reader.data():
                if not self._is_running:
                    break

                for preprocessor in self.preprocessors:
                    img = preprocessor(img)

                dm = self._wrapper.process(img)

                for postprocessor in self.postprocessors:
                    img, dm = postprocessor(img, dm)

                for writer in self.writers:
                    writer.write(img, dm)
        except KeyboardInterrupt:
            pass
        finally:
            self._reader.close()
            for writer in self.writers:
                writer.close()

    def stop(self):
        self._is_running = False

    @property
    def reader(self):
        return self._reader
