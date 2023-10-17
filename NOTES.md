#  Ссылки

## Репо, из которых брались примеры

# Команды

## DVC

Добавить gdrive в качестве remote
```bash
dvc remote add -d storage gdrive://18ExTxuzi0beJOtZ10DM4lgkcAauV4o59
```

Добавить датасет
```bash
dvc add data/fashion_mnist
```

Залить датасет в remote (для gdrive редиректит в браузер, там нужно залогиниться и дать dvc доступы):
```bash
dvc push
```

Статус
```bash
dvc status
```

Привести данные в соответствие с `*.dvc` файлами
```bash
dvc checkout
```
