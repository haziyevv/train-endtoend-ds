FROM python:3.11-slim-bullseye

ENV WORKDIR=/usr/src/app/

RUN mkdir -p $WORKDIR
WORKDIR $WORKDIR

COPY requriements.txt ${WORKDIR}/req_ml-model.txt
COPY gcp_training ${WORKDIR}/gcp_training

RUN pip install -r requirements.txt
ENTRYPOINT ["python3", "-m", "gcp_training.train"]
