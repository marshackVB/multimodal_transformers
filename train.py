import os
import sys
from sys import version_info
import json
import pandas as pd
from itertools import islice
import torch
from transformers import (DistilBertForSequenceClassification, DistilBertTokenizerFast, Trainer, TrainingArguments,
                          AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, DistilBertConfig, 
                          HfArgumentParser)
from arguments import DataArguments, ModelArgs, MultiModelTrainingArgs
from transformations import AutoTokenize, get_feature_names, get_transformer
from model import MultiModalLoader, MultiModelTransformer, compute_metrics
from helpers import get_best_metrics
import mlflow


#%load_ext autoreload
#%autoreload 2
#%aimport transformations,  model, arguments


def main():

    # Parse arguments
    parser = HfArgumentParser([DataArguments, ModelArgs, MultiModelTrainingArgs])

    # Temp load json config
    data_args, model_args, training_args = parser.parse_json_file(json_file=os.path.abspath("datasets/ecommerce_reviews/training_config.json"))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        data_args, model_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        raise Exception("You must provide a json configuration file as the single argument to the program")

    data_args, model_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    # Ingest
    train_df = pd.read_csv(os.path.abspath(data_args.train_data_path_or_table_name), 
                        index_col=0)

    test_df = pd.read_csv(os.path.abspath(data_args.test_data_path_or_table_name), 
                        index_col=0)

    train_df = train_df.sample(frac=1).reset_index(drop=True)
    test_df = test_df.sample(frac=1).reset_index(drop=True)
    # Data sample for local testing...
    #train_df = train_df.sample(frac=1).reset_index(drop=True)[:200]
    #test_df = train_df.copy(deep=True)


    # Transform features
    transformer = (get_transformer(model_type =       model_args.model_name_or_path, 
                                   text_cols =        data_args.column_info["text_cols"],
                                   numeric_cols =     data_args.column_info["num_cols"], 
                                   categorical_cols = data_args.column_info["cat_cols"])
                                    .fit(train_df))

    label = data_args.column_info['label_col']

    x_train = transformer.transform(train_df)
    x_train = pd.DataFrame(x_train, columns=get_feature_names(transformer))
    y_train = train_df[label]

    x_test = transformer.transform(test_df)
    x_test = pd.DataFrame(x_test, columns=get_feature_names(transformer))
    y_test = test_df[label]

    # Data loader
    train_data = MultiModalLoader(x_train, y_train, 'autotokenize__input_ids', 'autotokenize__attention_mask')
    test_data = MultiModalLoader(x_test, y_test, 'autotokenize__input_ids', 'autotokenize__attention_mask')

    config = AutoConfig.from_pretrained(
                model_args.model_name_or_path,
                mum_labels = model_args.num_labels,
                cache_dir=model_args.cache_dir,
            )

    config.num_cat_features_dim = len(train_data.num_cat_features)

    model = MultiModelTransformer.from_pretrained(model_args.model_name_or_path, config=config)

    trainer = Trainer(model = model,
                      args = training_args,
                      train_dataset = train_data,
                      eval_dataset = test_data,
                      compute_metrics=compute_metrics)


    with mlflow.start_run() as run:
        trainer.train()

        best_metrics = get_best_metrics(trainer)

        mlflow.log_metrics(best_metrics)

        python_version = "{major}.{minor}.{micro}".format(major=version_info.major,
                                                    minor=version_info.minor,
                                                    micro=version_info.micro)

        try:
            runtime_version = spark.conf.get("spark.databricks.clusterUsageTags.sparkVersion")
        except NameError:
            runtime_version="local"


        params = {"eval_batch_size":        trainer.args.eval_batch_size,
                 "train_batch_size":       trainer.args.train_batch_size,
                 "gpus":                   trainer.args._n_gpu,
                 "epochs":                 trainer.args.num_train_epochs,
                 "metric_for_best_model":  trainer.args.metric_for_best_model,
                 "best_checkpoint":        trainer.state.best_model_checkpoint.split('/')[-1],
                 "runtime_version":        runtime_version,
                 "python_version":         python_version}
    
        mlflow.log_params(params)
        trainer.save_model('./huggingface_model')
        mlflow.log_artifacts('./huggingface_model', artifact_path='huggingface_model')

    
    #trainer.evaluate()

    #trainer.train(model_path = training_args.logging_dir if os.path.isdir(training_args.logging_dir) else None)
    #trainer.save_model()

if __name__ == "__main__":

    main()