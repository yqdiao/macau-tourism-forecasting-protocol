FROM python:3.12-slim

WORKDIR /app
COPY docker/requirements-protocol-smoke.txt ./requirements.txt
RUN pip install --no-cache-dir --timeout 120 --retries 10 -r requirements.txt

COPY data_pipeline.py protocol_config.py models.py run_backtest.py evaluate_predictions.py ./
COPY configs/protocol.yaml ./configs/protocol.yaml

ENTRYPOINT ["python", "run_backtest.py"]
