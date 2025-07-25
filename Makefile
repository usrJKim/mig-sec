PROBER_IMAGE=prober-image
CNN_IMAGE=cnn-image
LLM_IMAGE=llm-image
TEST_IMAGE=test-image

#dir
PROBER_DIR=prober
MODEL_DIR=DNNmodels

#docker files
PROBER_DOCKERFILE=$(PROBER_DIR)/Dockerfile.prober
CNN_DOCKERFILE=$(MODEL_DIR)/Dockerfile.model
LLM_DOCKERFILE=$(MODEL_DIR)/Dockerfile.model
TEST_DOCKERFILE=$(MODEL_DIR)/Dockerfile.test

.PHONY: all build clean build-prober build-model build-test cnn trans

all: build

build: build-prober build-cnn build-llm

build-prober:
	sudo docker build -f $(PROBER_DOCKERFILE) -t $(PROBER_IMAGE) $(PROBER_DIR)

build-cnn:
	sudo docker build -f $(CNN_DOCKERFILE) -t $(CNN_IMAGE) $(MODEL_DIR)

build-llm:
	sudo docker build -f $(LLM_DOCKERFILE) -t $(LLM_IMAGE) $(MODEL_DIR)

build-test:
	sudo docker build -f $(TEST_DOCKERFILE) -t $(TEST_IMAGE) $(MODEL_DIR)

clean:
	sudo docker rmi -f $(PROBER_IMAGE) || true
	sudo docker rmi -f $(MODEL_IMAGE) || true
cnn:
	./run_cnn.sh
trans:
	./run_trans.sh

