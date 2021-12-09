from typing import List, Tuple, Dict
import json

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
  best_log.pop('epoch')

  # Overal runtime metrics
  runtime_logs_idx = [idx for idx, values in enumerate(trainer.state.log_history) if values.get('train_runtime') is not None][0]
  runtime_logs = trainer.state.log_history[runtime_logs_idx]

  best_log['train_runtime'] = runtime_logs['train_runtime']
  best_log['train_loss'] = runtime_logs['train_loss']

  return best_log