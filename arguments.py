import sys
import os
import json
from dataclasses import dataclass, field
from typing import Optional
from transformers.training_args import TrainingArguments
from transformers import HfArgumentParser



@dataclass
class DataArguments:

    train_data_path_or_table_name:str = field(
        metadata={"help": "Location of training data"}
    )

    test_data_path_or_table_name:str = field(
        metadata={"help": "Location of testing data"}
    )

    column_info_path:str = field(
        metadata={"help": "File containing colum-level information"}
    )

    experiment_name:str = field(
        metadata={"help": "Location of MLflow experiment"}
    )

    def __post_init__(self):
        with open(os.path.abspath(self.column_info_path), 'r') as f:
            self.column_info = json.load(f)


   
@dataclass
class ModelArgs:


    max_length:int = field(
        metadata={"help": "Max number of tokens passed to tokenizer"}
    )

    tokenizer_batch_size:int = field(
        metadata={"help": "Batch size fed to tokenizer"}
    )

    num_labels:int = field(
        metadata={"help": "Number of labels to predict"}
    )

    model_name_or_path:str = field(
        metadata={"help": "Path to pretrained model of model identifier"}
    )

    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )



@dataclass
class MultiModelTrainingArgs(TrainingArguments):
 
    databricks_profile: Optional[str] = field(
        default=None, metadata={"help": "The Databricks CLI profile used to autenticate to a Workspace"}
    )

    do_train: bool = field(
        default=True, metadata={"help": "Indicates if model training should occure"}
    )

    per_device_train_batch_size: int = field(
        default=64, metadata={"help": "Training loop batch size"}
    )

    per_device_eval_batch_size: int = field(
        default=64, metadata={"help": "Evaluation loop batch size"}
    )

    save_strategy: str = field(
        default="epoch", metadata={"help": "The interval at which checkpoints will be saved"}
    )

    save_total_limit: int = field(
        default=10, metadata={"help": "The maximum number of checkpoint files that will be saved"}
    )

    evaluation_strategy: str = field(
        default="epoch", metadata={"help": "The Databricks CLI profile used to autenticate to a Workspace"}
    )

    logging_strategy: str = field(
        default="steps", metadata={"help": "The interval at which logs will be generated"}
    )

    logging_steps: int = field(
        default=50, metadata={"help": "The frequency at which logs will be written"}
    )

    overwrite_output_dir: bool = field(
        default=True, metadata={"help": "The Databricks CLI profile used to autenticate to a Workspace"}
    )

    seed: int = field(
        default=123, metadata={"help": "Random seed"}
    )

    report_to: str = field(
        default="none", metadata={"help": "Logging to services, such as MLflow"}
    )

    disable_tqdm: bool = field(
        default=False, metadata={"help": "Disable or enable this logging feature"}
    )

    load_best_model_at_end: bool = field(
        default=True, metadata={"help": "Load the best model checkpoint after training"}
    )

    metric_for_best_model: str = field(
        default="f1", metadata={"help": "Metric for choosing the best model"}
    )

    greater_is_better: bool = field(
        default=True, metadata={"help": "Indicates how best metric should be assessed"}
    )

    log_to_managed_mlflow: bool = field(
        default=False, metadata={"help": "Indicates if artifacts should be logged to a remote tracking server"}
    )

    dataloader_pin_memory: bool = field(
        default=False, metadata={"help": "Indicates if dataloader should be pinned to memory"}
    )

