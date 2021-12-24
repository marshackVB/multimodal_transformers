from typing import List, Tuple, Dict
import json
import torch
from torch import nn
from transformers import DistilBertForSequenceClassification
from sklearn.metrics import precision_recall_fscore_support

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def input_ids_tensors(df, input_ids_col):
    return (torch.tensor(df[input_ids_col])
                          .to(torch.int64)
                          .to(device))

def attention_mask_tensors(df, attention_mask_col):
    return (torch.tensor(df[attention_mask_col])
                          .to(torch.int64)
                          .to(device))

def num_cat_feature_tensors(df, num_cat_feature_cols):
    return (torch.tensor(df[num_cat_feature_cols]
                          .to_numpy(dtype='float32'))
                          .float()
                          .to(device))


class MultiModalLoader(torch.utils.data.Dataset):
    def __init__(self, x_train, y_train, input_ids_col, attention_mask_col):
        self.x_train = x_train
        self.y_train = y_train
        self.input_ids_col = input_ids_col
        self.attention_mask_col = attention_mask_col
        self.num_cat_feature_cols = [feature for feature in x_train.columns if feature not in
                                     [input_ids_col, attention_mask_col]]
        

    def __len__(self):
        return self.x_train.shape[0]


    def __getitem__(self, idx):
        x_train_row = self.x_train.iloc[idx]

        input_ids = input_ids_tensors(x_train_row, self.input_ids_col)

        attention_mask = attention_mask_tensors(x_train_row, self.attention_mask_col)

        num_cat_features = num_cat_feature_tensors(x_train_row, self.num_cat_feature_cols)
        
        label = (torch.tensor(self.y_train.iloc[idx])
                      .unsqueeze(0)
                      .to(torch.int64)
                      .to(device))

        return {"input_ids": input_ids, "attention_mask": attention_mask, 
                "num_cat_features": num_cat_features, "labels": label}


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super(MLP, self).__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        output = self.layer_1(x)
        output = nn.ReLU()(output)
        output = self.dropout(output)
        output = self.layer_2(output)
        return output


class MultiModalTransformer(DistilBertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        """
        self.classifier = self.weight_init(
                                nn.Linear(config.dim + config.num_cat_features_dim, config.num_labels)
                                )
        """
        self.classifier = MLP(config.dim + config.num_cat_features_dim, 200, config.num_labels)
        
                            
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        class_weights=None,
        num_cat_features=None
        ):

        # Get last hidden state for [CLS] token embedding; this is a single
        # 768 dimension vector for each input record

        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        hidden_state = outputs.last_hidden_state
        #hidden_state = outputs[0]
        # Extract the [CLS] token embedding from the full array of 
        # word embeddings
        pooled_output = hidden_state[:, 0]
        pooled_output = self.pre_classifier(pooled_output)
        pooled_output = nn.ReLU()(pooled_output)
        pooled_output = self.dropout(pooled_output)

        combined_features = torch.cat((pooled_output, num_cat_features), dim=1)
        logits = self.classifier(combined_features)


        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            output = (logits,) + outputs[1:]
            return ((loss,) + output)

        softmax = nn.Softmax(dim=1)
        return softmax(logits)


def weight_init(self,  m, activation='linear'):
    torch.nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain(activation))
    return m


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    return {
        'f1': f1,
        'precision': precision,
        'recall': recall
            }


def get_best_metrics(trainer) -> Dict[str, float]:
    """
    Extract metrics from a fitted Trainer instance.
    Args:
    trainer: A Trainer instance that has been trained on data.

    Returns:
    A dictionary of metrics and their values.
    """

    # Best model metrics
    best_checkpoint = f'{trainer.state.best_model_checkpoint}/trainer_state.json' 

    with open(best_checkpoint) as f:
        metrics = json.load(f)

    best_step = metrics['global_step']

    all_log_history = enumerate(metrics['log_history'])

    best_log_idx = [idx for idx, values in all_log_history if values['step'] == best_step][0]

    best_log = metrics['log_history'][best_log_idx]
    best_log['early_stopping_epoch'] = best_log.pop('epoch')
    #best_log.pop('epoch')

    # Overal runtime metrics
    runtime_logs_idx = [idx for idx, values in enumerate(trainer.state.log_history) if values.get('train_runtime') is not None][0]
    runtime_logs = trainer.state.log_history[runtime_logs_idx]

    best_log['train_runtime'] = runtime_logs['train_runtime']
    best_log['train_loss'] = runtime_logs['train_loss']

    return best_log