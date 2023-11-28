FROM python:3.11.6-bullseye AS builder

# Setup external dependancies
ENV POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1
ENV PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH"
RUN apt update && \
    apt install build-essential -y && \
    curl -sSL https://install.python-poetry.org | python3 - && \
    pip install -U pip && \
    pip install dvc dvc-gdrive

# Setup project dependencies
COPY poetry.lock pyproject.toml /app/
WORKDIR /app
RUN poetry install --no-interaction --no-ansi -vvv --no-root

# Setup project
COPY . /app
RUN poetry install --only-root

CMD poetry run mlops train_infer
