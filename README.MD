# Depth map converter

## Установка

1. Скачать последний релиз с раздела релизов
или склонировать проект
```bash
git clone --recurse-submodules https://github.com/whitegl0w/neural_video_3D_converter.git
```

2. Открыть консоль в папке с проектом

3. Рекомендуется создать venv (виртуальную среду)
```bash
python -m venv envDepthmap
```

4. Активировать созданный venv:
```bash
source envDepthmap/bin/activate # (для Linux/MacOS)
envDepthmap/Scripts/activate.bat # (для Windows)
```

5. Установить пакеты
```bash
pip install -r requirements.txt
```
При наличии поддержки CUDA в видеокарте можно установить версию PyTorch с поддержкой GPU вычислений. 
Это обеспечит более высокую скорость работы нейронной сети.

Для установки версии CUDA необходимо предварительно в файле requirements.txt поменять строку
"--extra-index-url https://download.pytorch.org/whl/cpu" 
на
"--extra-index-url https://download.pytorch.org/whl/cu118" 

6. [Скачать](https://github.com/isl-org/MiDaS/releases/tag/v3_1) требуемые модели MiDaS и положить в папку "models"

## Пример работы

Запустить приложение в режиме UI
```bash
python main.py
```

Взять изображение с камеры и вывести на экран в анаглифном формате
```bash
python main.py cam 0 -t screen -a
```

Считать видеофайл, конвертировать видео в анаглифный формат и записать в файл out.avi
```bash
python main.py vid input.mp4 -t screen video -a
```

Считать видеофайл, конвертировать видео в анаглифный формат и записать в файл out.avi, процесс конвертации дублировать на экран

```bash
python main.py vid input.mp4 -t screen video -a
```

Считать изображения из папки input, создать для них карты глубины и записать в папку output, использовать модель DPT_LARGE для максимального качества
```bash 
python main.py img input -t images -l
```
