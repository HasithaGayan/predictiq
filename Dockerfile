FROM python:3.14.2-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    PORT=10000

WORKDIR /app

COPY requirements-runtime.txt .

RUN pip install --no-cache-dir --only-binary=:all: -r requirements-runtime.txt

COPY backend/ ./backend/
COPY src/ ./src/
COPY frontend/ ./frontend/

COPY reports/generated/day66/selected_pipeline.joblib ./reports/generated/day66/selected_pipeline.joblib

RUN echo "3d9058804e718a7c18f62e603dbc15e0b64790efcbd2d06330a47c273e9950d1  reports/generated/day66/selected_pipeline.joblib" | sha256sum -c -

RUN useradd --create-home --uid 10001 appuser

USER appuser

EXPOSE 10000

CMD ["sh", "-c", "exec python -m uvicorn backend.app:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1"]
