PROBER_IMAGE=prober-image
MODEL_IMAGE=model-image

#dir
PROBER_DIR=prober
MODEL_DIR=DNNmodels

#docker files
PROBER_DOCKERFILE=$(PROBER_DIR)/Dockerfile.prober
MODEL_DOCKERFILE=$(MODEL_DIR)/Dockerfile.model

.PHONY: all build clean build-prober build-model run

all: build

build: build-prober build-model

build-prober:
	docker build -f $(PROBER_DOCKERFILE) -t $(PROBER_IMAGE) $(PROBER_DIR)

build-model:
	docker build -f $(MODEL_DOCKERFILE) -t $(MODEL_IMAGE) $(MODEL_DIR)

clean:
	docker rmi -f $(PROBER_IMAGE) || true
	docker rmi -f $(MODEL_IMAGE) || true
run:
	./run.sh

