import numpy.typing as npt

from dataclasses import dataclass
from typing import Callable, Optional
from depth_prediction.run import create_depth_map, prepare_model
from abc import ABC, abstractmethod

from models.settings import Models

RED = 2
GREEN = 1
BLUE = 0


@dataclass
class DmMediaParams:
    fps: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    frame_count: Optional[int] = None


class DmMediaReader(ABC):
    def prepare(self) -> DmMediaParams: ...

    @abstractmethod
    def data(self) -> Optional[npt.NDArray]: ...

    def close(self): ...


class DmMediaSeekableReader(DmMediaReader):
    @abstractmethod
    def seek(self, position_ms: int): ...

    @property
    @abstractmethod
    def duration(self) -> int: ...

    def replay(self):
        self.seek(0)


class DmMediaWriter(ABC):
    def prepare(self, media_params: DmMediaParams): ...

    @abstractmethod
    def write(self, img: npt.NDArray, dm: npt.NDArray): ...

    def close(self): ...


class DmMediaConverter:
    preprocessors: list[Callable[[npt.NDArray], npt.NDArray]] = []
    postprocessors: list[Callable[[npt.NDArray, npt.NDArray], tuple[npt.NDArray, npt.NDArray]]] = []
    writers: list[DmMediaWriter] = []

    def __init__(self, model: Models, reader: DmMediaReader):
        self._reader = reader
        self._model_type = model.value.type
        self._model_path = model.value.path
        self._is_running = False

    def start(self):
        self._is_running = True

        media_params = self._reader.prepare()
        for writer in self.writers:
            writer.prepare(media_params)

        prepare_model(self._model_type, self._model_path)

        try:
            for img in self._reader.data():
                if not self._is_running:
                    break

                for preprocessor in self.preprocessors:
                    img = preprocessor(img)

                dm = create_depth_map(img)

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
