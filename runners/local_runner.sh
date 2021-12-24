#!/bin/bash

mlflow run ./ -P max_length=50 -P num_train_epochs=5 -P training_sample_record_num=200 --experiment-id 0