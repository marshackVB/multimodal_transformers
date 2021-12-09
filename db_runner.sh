#!/bin/bash

export MLFLOW_TRACKING_URI=databricks://e2-demo-east

mlflow run https://github.com/marshackVB/multimodal_transformers \
    -P config_file=./datasets/ecommerce_reviews/training_config.json \
    --backend-config cluster_config.json --experiment-id 648050317765410