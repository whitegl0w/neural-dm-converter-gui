import cv2

from dmconvert.postprocessors import create_anaglyph_processor, create_dm_correcter
from .parameters import ControlElement, ControlProperty

POSTPROCESSOR_ELEMENTS = [
    ControlElement(
        name="DM Corrector",
        builder=create_dm_correcter,
        properties=[
            ControlProperty(name="windows_size", caption="Окно (кадров)", min_value=1, max_value=50),
            ControlProperty(name="return_num", caption="Номер кадра", min_value=1, max_value=50),
            ControlProperty(name="move_factor", caption="Порог движения", min_value=50, max_value=700)
        ]
    ),
    ControlElement(
        name="Конверт в анаглиф",
        builder=create_anaglyph_processor,
        properties=[
            ControlProperty(name="max_offset", caption="Cдвиг", min_value=0, max_value=30),
        ]
    ),
    ControlElement(
        name="Размытие карты глубины",
        builder=lambda factor: lambda img, dm: (img, cv2.blur(dm, (factor, factor))),
        properties=[
            ControlProperty(name="factor", caption="Сила", min_value=1, max_value=50)
        ]
    )
]


PREPROCESSOR_ELEMENTS = [
    ControlElement(
        name="Повернуть",
        builder=lambda angle: lambda img: cv2.rotate(img, angle),
        properties=[
            ControlProperty(name="angle", caption="Угол", min_value=0, max_value=2)
        ]
    ),
    ControlElement(
        name="Сжатие",
        builder=lambda factor: lambda img: cv2.resize(img, (factor, int(factor / img.shape[1] * img.shape[0]))),
        properties=[
            ControlProperty(name="factor", caption="Коэффициент", min_value=64, max_value=1920)
        ]
    ),
    ControlElement(
        name="Размытие",
        builder=lambda factor: lambda img: cv2.blur(img, (factor, factor)),
        properties=[
            ControlProperty(name="factor", caption="Сила", min_value=1, max_value=50)
        ]
    ),
    ControlElement(
        name="Добавить контуры",
        builder=lambda t1, t2: lambda img: cv2.subtract(img, cv2.cvtColor(cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), t1, t2), cv2.COLOR_GRAY2BGR)),
        properties=[
            ControlProperty(name="t1", caption="Порог 1", min_value=0, max_value=100),
            ControlProperty(name="t2", caption="Порог 2", min_value=0, max_value=100),
        ]
    ),
    ControlElement(
        name="Преобразовать в ЧБ",
        builder=lambda: lambda img: cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR),
        properties=[
        ]
    ),
]
