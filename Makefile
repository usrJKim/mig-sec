PROBER_IMAGE=prober-image
MODEL_IMAGE=model-image
TEST_IMAGE=test-image

#dir
PROBER_DIR=prober
MODEL_DIR=DNNmodels

#docker files
PROBER_DOCKERFILE=$(PROBER_DIR)/Dockerfile.prober
MODEL_DOCKERFILE=$(MODEL_DIR)/Dockerfile.model
TEST_DOCKERFILE=$(MODEL_DIR)/Dockerfile.test

.PHONY: all build clean build-prober build-model build-test cnn trans

all: build

build: build-prober build-model

build-prober:
	sudo docker build -f $(PROBER_DOCKERFILE) -t $(PROBER_IMAGE) $(PROBER_DIR)

build-model:
	sudo docker build -f $(MODEL_DOCKERFILE) -t $(MODEL_IMAGE) $(MODEL_DIR)

build-test:
	sudo docker build -f $(TEST_DOCKERFILE) -t $(TEST_IMAGE) $(MODEL_DIR)

clean:
	sudo docker rmi -f $(PROBER_IMAGE) || true
	sudo docker rmi -f $(MODEL_IMAGE) || true
cnn:
	./run_cnn.sh
trans:
	./run_trans.sh

