FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    WYOMING_URI=tcp://0.0.0.0:10200 \
    DATA_DIR=/data \
    LANGUAGE=en \
    CROP_SILENCE=300 \
    STEPS=5 \
    SPEED=1.0 \
    THREADS=4

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/mitrokun/wyoming_supertonic.git .

RUN pip install --no-cache-dir \
    supertonic \
    wyoming \
    sentence-stream \
    num2words \
    onnxruntime \
    numpy

# Optional directory for local models
VOLUME ["/data"]

EXPOSE 10200

ENTRYPOINT ["python3", "-m", "wyoming_supertonic"]

CMD [
    "--uri", "tcp://0.0.0.0:10200"
]
