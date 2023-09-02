import sys

"""
    Настройки моделей
"""

# Директория с файлами моделей
MODELS_DIR = 'models'

# Автоматически регистрировать файлы моделей (*.pt) из директории [MODELS_DIR]
AUTODETECT_MODELS = True


"""
    Сторонние модули
"""
# Пути к сторонним модулям для настройки нейронной сети
sys.path.extend(['./depthmap/MiDaS'])

# Название класса - загрузчика модели, должен наследоваться от [BaseDmWrapper]
from depthmap_wrappers.midas import MidasDmWrapper
MODEL_LOADER = MidasDmWrapper
