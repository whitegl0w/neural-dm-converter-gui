import glob
import os
from dataclasses import dataclass
from typing import Optional

from aenum import extend_enum, Enum
from pathlib import Path
from settings import MODELS_DIR


@dataclass
class Model:
    type: str
    path: str


class Models(Enum):
    @classmethod
    def autodetect(cls):
        files = glob.glob(os.path.join(MODELS_DIR, "*.pt"))
        models = map(lambda file: Model(Path(file).stem, file), files)
        for model in models:
            extend_enum(cls, model.type, model)

    @classmethod
    def find_by_model_type(cls, type: str) -> Optional[Model]:
        models = [m.value for m in cls]
        for model in models:
            if model.type == type:
                return model
        return None

