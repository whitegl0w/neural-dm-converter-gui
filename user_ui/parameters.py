from dataclasses import dataclass
from typing import Callable


@dataclass
class ControlProperty:
    caption: str
    name: str
    min_value: int
    max_value: int


@dataclass
class ControlElement:
    name: str
    builder: Callable[[...], Callable]
    properties: list[ControlProperty]