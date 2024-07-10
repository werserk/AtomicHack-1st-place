# Поиск дефектов сварных швов

## 1. Подготовка весов

Для запуска модели надо веса скачать веса
с [диска](https://drive.google.com/drive/folders/1jpuMyeurWTFfz9V53p-9iKq0o_DcJ_cU?usp=sharing) в папку `models`
Должно получиться так:

```
models/
├── ckpt_best_s1.pth
├── ckpt_best_sm.pth
├── ckpt_best_m1.pth
├── ckpt_latest_m1.pth
...
```

## 2. Подготовка тестовых данных

Тестовые данные должны лежать в папке `data` и путь должен быть указан в `config.py` \
Папка `data` будет выглядеть так:

```
data/Тестовый датасет
├── 0.jpg
├── 100.jpg
├── 101.jpg
...
```

`config.py`:

```python
class CONFIG:
    path_to_images = 'data/Тестовый датасет'
    model_path = 'models/ckpt_best_m1.pth'
    model_type = 'yolo_nas_m'
    submit_file_name = 'submit_best_m1.csv'
```

## 3. Выбор весов

Для запуска разных моделей следует изменить файл `config.py`
В нём нужно изменить параметры `model_type` и `model_path` в зависимости от модели, например:

`model_type = 'yolo_nas_m'`, тогда путь к весам `model_path = `

- ckpt_best_m1.pth - **Лучшие веса на нашей валидации**
- ckpt_latest_m1.pth

`model_type = 'yolo_nas_s'`, тогда путь к весам `model_path = `

- ckpt_best_s1.pth
- ckpt_best_sm.pth

## 4. Запуск

Запускаем докер и получаем файл с предсказанием в папке `submits` (мы уже положили туда 2 сабмита)

```bash
docker compose up
```


