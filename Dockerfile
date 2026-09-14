FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && useradd --create-home --uid 10001 appuser
COPY --chown=appuser:appuser backend/ backend/
COPY --chown=appuser:appuser frontend/ frontend/
COPY --chown=appuser:appuser fixtures/ fixtures/
COPY --chown=appuser:appuser tests/ tests/
COPY --chown=appuser:appuser eval/ eval/
COPY --chown=appuser:appuser train_ppo.py train_customer_policy.py ./
COPY --chown=appuser:appuser artifacts/ artifacts/
USER appuser
EXPOSE 8080
CMD ["python", "-m", "uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8080"]
