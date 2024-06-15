# AtomicHack

Link to Atomichack: https://atomichack.ru/ \
Link to GitHub: https://github.com/werserk/AtomicHack

## Описание

Создать программный модуль с использованием ИИ, который автоматически выявляет и классифицирует дефекты сварных швов по
фотографиям. Модуль должен обрабатывать загруженные фотографии и определять типы дефектов.

## Setup

Clone repo:

```bash
git clone https://github.com/werserk/AtomicHack.git && cd AtomicHack
```

Install dependencies:

```bash
poetry install
```

Run streamlit:

```bash
streamlit run streamlit_app.py
```

Run backend (if you want to use API)

```bash
uvicorn run_backend:app --host 0.0.0.0 --port 8000
```