FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime
ARG UPSTREAM_COMMIT
RUN test -n "$UPSTREAM_COMMIT"
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/time-series-foundation-models/lag-llama.git /opt/upstream \
 && git -C /opt/upstream checkout --detach "$UPSTREAM_COMMIT" \
 && test "$(git -C /opt/upstream rev-parse HEAD)" = "$UPSTREAM_COMMIT" \
 && pip install --no-cache-dir -e /opt/upstream
WORKDIR /app
COPY docker/requirements-lag-llama.txt /tmp/common.txt
RUN pip install --no-cache-dir -r /tmp/common.txt
COPY data_pipeline.py run_foundation_backtest.py ./
COPY foundation_models/ ./foundation_models/
COPY configs/ ./configs/
ENTRYPOINT ["python", "run_foundation_backtest.py"]
