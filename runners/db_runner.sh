#!/bin/bash

export MLFLOW_TRACKING_URI=databricks://e2-demo-east

mlflow run https://github.com/marshackVB/multimodal_transformers \
    -P max_length=100 -P num_train_epochs=3 -P per_device_train_batch_size=128 -P per_device_eval_batch_size=128 \
    --backend databricks \
    --backend-config cluster_config.json \
    --experiment-id 648050317765410