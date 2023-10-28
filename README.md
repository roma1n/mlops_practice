# ML Ops Practice

Autogenerated docs: [link](https://roma1n.github.io/mlops_practice/index.html)

Учебный проект по курсу MLOps. Решается относительно простая задача (классификации FashionMNIST) и на ее примере изучается использование технолгий MLOps:

- `dvc`, `git`
- `hydra`
- `poetry`
- `pytorch_lighting`
- `sphinx (autodoc)`

Как запустить:
1. Установить `git`, `dvc`, `poetry`
2. Склонировать этот репозиторий
3. Загрузить данные через dvc: `dvc pull -r dvcreader`
4. Войти в окружение через poetry: `poetry install && poetry shell`
5. Запустить одно из:
    - `mlops train` - обучение модели
    - `mlops infer` - инференс модели с записью результатов применения на тесте в csv-файл (перед запуском инференса нужно запустить обучение)
    - `mlops train_infer` - последовательный запуск `train` и `infer`
