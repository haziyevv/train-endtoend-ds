FROM python:3.11-slim-bullseye

ENV WORKDIR=/usr/src/app/

RUN mkdir -p $WORKDIR
WORKDIR $WORKDIR

COPY requirements.txt ${WORKDIR}/requirements.txt
COPY config.yaml ${WORKDIR}/config.yaml
COPY gcp_training ${WORKDIR}/gcp_training

RUN pip install -r requirements.txt
ENTRYPOINT ["python3", "-m", "gcp_training"]
