import enum
from dataclasses import dataclass


@dataclass
class Model:
    type: str
    path: str


class Models(enum.Enum):
    DPT_LARGE = Model('dpt_large', 'models/dpt_large-midas-2f21e586.pt')
    V21_NORMAL = Model('midas_v21', 'models/model.pt')
    V21_SMALL = Model('midas_v21_small', 'models/small.pt')

    DEFAULT_LARGE = DPT_LARGE
    DEFAULT_SMALL = V21_NORMAL
