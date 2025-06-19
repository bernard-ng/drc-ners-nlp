.PHONY: default
default: help

.PHONY: help
help:
	@echo Tasks:
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

.PHONY: download
download:
	@if [ ! -f dataset/names.csv ]; then \
		set -a; [ -f .env.local ] && . .env.local; set +a; \
		[ -z "$$DATASET_URL" ] && . .env; \
		mkdir -p dataset; \
		curl -L "$${DATASET_URL}" -o dataset/names.csv; \
	else \
		echo "dataset/names.csv already exists. Skipping download."; \
	fi

.PHONY: clean
clean:
    rm -rf ./models
	rm -rf ./results
	rm -rf ./dataset/spacy/train.spacy
	rm -rf ./dataset/spacy/dev.spacy
