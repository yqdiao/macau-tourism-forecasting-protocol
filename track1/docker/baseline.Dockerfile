FROM python:3.12-slim

WORKDIR /workspace
COPY track1/docker/requirements-protocol-smoke.txt ./requirements.txt
RUN pip install --no-cache-dir --timeout 120 --retries 10 -r requirements.txt

COPY track1/code/ ./track1/code/
COPY track1/configs/ ./track1/configs/
COPY shared/configs/ ./shared/configs/

ENTRYPOINT ["python", "track1/code/run_backtest.py"]
