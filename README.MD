## Depth map converter

Взять изображение с камеры и вывести на экран в анаглифном формате

`python main.py cam 0 -t screen -a`

Считать видеофайл, конвертировать видео в анаглифный формат и записать в файл out.avi

`python main.py vid input.mp4 -t screen video -a`

Считать видеофайл, конвертировать видео в анаглифный формат и записать в файл out.avi, процесс конвертации дублировать на экран

`python main.py vid input.mp4 -t screen video -a`

Считать изображения из папки input, создать для них карты глубины и записать в папку output, использовать модель DPT_LARGE для максимального качества

`python main.py img input -t images -l`

Запустить приложение в режиме UI

`python main.py`