#!/bin/bash

mlflow run ./ -P max_length=50 -P num_train_epochs=1 --experiment-id 0