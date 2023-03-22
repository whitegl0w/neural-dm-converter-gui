import cv2

from dmconvert.postprocessors import create_anaglyph_processor
from .parameters import ControlElement, ControlProperty

POSTPROCESSOR_ELEMENTS = [
    ControlElement(
        name="Аналгиф",
        builder=create_anaglyph_processor,
        properties=[
            ControlProperty(name="max_offset", caption="Максимальный сдвиг", min_value=0, max_value=30),
            ControlProperty(name="direction", caption="Направление", min_value=-1, max_value=1),
        ]
    ),
    ControlElement(
        name="Размытие карты глубины",
        builder=lambda factor: lambda img, dm: (img, cv2.blur(dm, (factor, factor))),
        properties=[
            ControlProperty(name="factor", caption="Сила", min_value=1, max_value=30)
        ]
    )
]
