import os
from pickle import dump
from sys import version_info
import json
import pandas as pd
import torch
from transformers import Trainer, AutoConfig, HfArgumentParser, EarlyStoppingCallback
import mlflow
from arguments import DataArguments, ModelArgs, MultiModelTrainingArgs
from transformations import AutoTokenize, get_feature_names, get_transformer
from model import MultiModalLoader, MultiModalTransformer, compute_metrics, get_best_metrics


def main(data_args, model_args, training_args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_df = pd.read_csv(os.path.abspath(data_args.train_data_path_or_table_name), 
                        index_col=0)

    test_df = pd.read_csv(os.path.abspath(data_args.test_data_path_or_table_name), 
                        index_col=0)


    if data_args.training_sample_record_num > 0:
        train_df = train_df.sample(frac=1).reset_index(drop=True)[:data_args.training_sample_record_num]
        test_df = train_df.copy(deep=True)
    else:    
        train_df = train_df.sample(frac=1).reset_index(drop=True)
        test_df = test_df.sample(frac=1).reset_index(drop=True)


    transformer = (get_transformer(model_type =           model_args.model_name_or_path, 
                                   text_cols =            data_args.column_info["text_cols"],
                                   numeric_cols =         data_args.column_info["num_cols"], 
                                   categorical_cols =     data_args.column_info["cat_cols"],
                                   max_length =           model_args.max_length, 
                                   tokenizer_batch_size = model_args.tokenizer_batch_size)
                                   .fit(train_df))

    label = data_args.column_info['label_col']

    x_train = transformer.transform(train_df)
    x_train = pd.DataFrame(x_train, columns=get_feature_names(transformer))
    y_train = train_df[label]

    x_test = transformer.transform(test_df)
    x_test = pd.DataFrame(x_test, columns=get_feature_names(transformer))
    y_test = test_df[label]

    train_data = MultiModalLoader(x_train, y_train, 'autotokenize__input_ids', 'autotokenize__attention_mask')
    test_data = MultiModalLoader(x_test, y_test, 'autotokenize__input_ids', 'autotokenize__attention_mask')

    config = AutoConfig.from_pretrained(
                model_args.model_name_or_path,
                mum_labels = model_args.num_labels,
                cache_dir=model_args.cache_dir,
            )

    config.num_cat_features_dim = len(train_data.num_cat_feature_cols)

    model = MultiModalTransformer.from_pretrained(model_args.model_name_or_path, config=config)

    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=2, 
                                                    early_stopping_threshold=.01)

    trainer = Trainer(model = model,
                      args = training_args,
                      train_dataset = train_data,
                      eval_dataset = test_data,
                      compute_metrics=compute_metrics,
                      callbacks=[early_stopping_callback])


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


        params = {"eval_batch_size":       trainer.args.eval_batch_size,
                 "train_batch_size":       trainer.args.train_batch_size,
                 "gpus":                   trainer.args._n_gpu,
                 "epochs":                 trainer.args.num_train_epochs,
                 "metric_for_best_model":  trainer.args.metric_for_best_model,
                 "best_checkpoint":        trainer.state.best_model_checkpoint.split('/')[-1],
                 "runtime_version":        runtime_version,
                 "python_version":         python_version}
    
        mlflow.log_params(params)

        dump(transformer, open('transformer.pkl', 'wb'))
        trainer.save_model('./huggingface_model')

        mlflow.log_artifacts('./huggingface_model', artifact_path='huggingface_model')
        mlflow.log_artifact('transformer.pkl')


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--model_name_or_path",             help="Pretrained model type")
    parser.add_argument("--output_dir",                     help="Model checkpoint directory")
    parser.add_argument("--column_info_path",               help="Model checkpoint directory")
    parser.add_argument("--experiment_name",                help="Name of MLflow experiment")
    parser.add_argument("--train_data_path_or_table_name",  help="Location of training data file")
    parser.add_argument("--test_data_path_or_table_name",   help="Location of test data file")
    parser.add_argument("--databricks_profile",             help="Databricks CLI profile for authentication")
    parser.add_argument("--num_labels",                     help="Number of labels to predict")
    parser.add_argument("--max_length",                     help="Number of tokens to retain")
    parser.add_argument("--tokenizer_batch_size",           help="Batch size for tokenizer")
    parser.add_argument("--num_train_epochs",               help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size",    help="Training batch size")
    parser.add_argument("--per_device_eval_batch_size",     help="Evaluation batch size")
    parser.add_argument("--save_strategy",                  help="Checkpoint save strategy")
    parser.add_argument("--save_total_limit",               help="Maximum number of saved checkpoints")
    parser.add_argument("--evaluation_strategy",            help="Evaluation strategy")
    parser.add_argument("--logging_strategy",               help="Logging strategy")
    parser.add_argument("--logging_steps",                  help="Number of logging steps")
    parser.add_argument("--training_sample_record_num",     help="Sample records for testing/dev")
    parser.add_argument("--early_stopping_patience",        help="Early stopping epochs")
    parser.add_argument("--early_stopping_threshold",       help="Early stopping threshold")


    hf_argparser = HfArgumentParser([DataArguments, ModelArgs, MultiModelTrainingArgs])

    data_args, model_args, training_args = hf_argparser.parse_args_into_dataclasses(look_for_args_file=False)

    main(data_args, model_args, training_args)