ARG BASE_IMAGE
FROM ${BASE_IMAGE}
RUN apt-get update && apt-get install -y zip unzip swig=3.0.12-2 sysstat && pip install --no-cache --upgrade pip
ARG REQUIREMENTS_TXT
COPY ${REQUIREMENTS_TXT} /tmp/requirements.txt
RUN pip install --no-cache -r /tmp/requirements.txt && rm -rf /root/.cache
COPY bayesmark-0.0.8.tar.gz /tmp/bayesmark-0.0.8.tar.gz
RUN pip install --force-reinstall /tmp/bayesmark-0.0.8.tar.gz
