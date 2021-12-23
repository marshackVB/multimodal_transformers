import mlflow
from mlflow.tracking import MlflowClient
from pickle import load
import torch
from transformers import AutoModelForSequenceClassification
from model import MultiModalTransformer
from transformations import get_feature_names

client = MlflowClient()

# Copy artifacts to driver
run_id = '9f3b9164bbb440b59f00679077cbba9c'
client.download_artifacts(run_id=run_id, path="transformer.pkl", dst_path='./loaded_artifacts')
client.download_artifacts(run_id=run_id, path="huggingface_model", dst_path='./loaded_artifacts')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load artifacts
transformer = load(open('transformer.pkl', 'rb'))
model_dir = './loaded_artifacts/huggingface_model'
model = MultiModalTransformer.from_pretrained(model_dir).to(device)

# Generate sample data
inference_sample = train_df[:20]
x_to_score = transformer.transform(inference_sample)
x_to_score = pd.DataFrame(x_to_score, columns=get_feature_names(transformer))

# Specify features names
input_ids_col = 'autotokenize__input_ids'
attention_mask_col = 'autotokenize__attention_mask'
num_cat_feature_cols = [feature for feature in x_to_score.columns if feature not in
                       [input_ids_col, attention_mask_col]]


input_ids = input_ids_tensors(x_to_score, input_ids_col)

attention_mask = attention_mask_tensors(x_to_score, attention_mask_col)

num_cat_features = num_cat_feature_tensors(x_to_score, num_cat_feature_cols)


with torch.no_grad():
    predictions = model(input_ids=input_ids, 
                        attention_mask=attention_mask, 
                        num_cat_features=num_cat_features)
                
predictions