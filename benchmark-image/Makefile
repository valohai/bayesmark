.PHONY: default

VERSION := $(shell date +%Y%m%d)-$(shell git describe --always --dirty --long --exclude '*')
BASE_IMAGE := python:3.6
DOCKER_IMAGE_NAME := valohai/amd-benchmark
BAYESMARK_VERSION := 0.0.8

default:
	@echo "make pin-requirements:  pin requirements from requirements.in to requirements.txt"
	@echo "make docker-image       build the competition image"

pin-requirements:
	mkdir -p $(PWD)/cache
	docker run --rm -it -v $(PWD):/wd --workdir /wd ${BASE_IMAGE} sh -c "pip install -q pip-tools && pip-compile --cache-dir /wd/cache -v requirements.in"

docker-image:
	cd .. && sh build_wheel.sh
	cp ../dist/bayesmark-${BAYESMARK_VERSION}.tar.gz .
	DOCKER_BUILDKIT=1 docker build --build-arg REQUIREMENTS_TXT=requirements.txt --build-arg BASE_IMAGE=${BASE_IMAGE} -t ${DOCKER_IMAGE_NAME}:${VERSION} .
	docker tag ${DOCKER_IMAGE_NAME}:${VERSION} ${DOCKER_IMAGE_NAME}:latest
