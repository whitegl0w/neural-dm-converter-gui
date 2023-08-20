import enum
from dataclasses import dataclass


@dataclass
class   Model:
    type: str
    path: str


class Models(enum.Enum):
    DPT_LARGE = Model('dpt_large', 'midas/dpt_large-midas-2f21e586.pt')
    V21_NORMAL = Model('midas_v21', 'midas/model.pt')
    V21_SMALL = Model('midas_v21_small', 'midas/small.pt')
    V31_LARGE = Model('dpt_beit_large_512', 'models/dpt_beit_large_512.pt')
    V31_SMALL = Model('dpt_swin2_tiny_256', 'models/dpt_swin2_tiny_256.pt')

    DEFAULT_LARGE = DPT_LARGE
    DEFAULT_SMALL = V21_NORMAL
