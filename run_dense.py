# train a sparse model
from transformers import Trainer, TrainingArguments 
import evaluate
import torch
from datasets import DatasetDict
from torch.nn.utils.rnn import pad_sequence
import os
from transformers import RobertaModel
from transformers import AutoTokenizer
from datasets import load_dataset
from models import Dense
from utils import CustomTrainer

# set configs
os.environ["WANDB_PROJECT"] = "moe_classifier"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == "__main__":

    model = RobertaModel.from_pretrained("roberta-base")
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    data = load_dataset("imdb")
    data = data.shuffle(seed=42)
    data["test"] = data["test"].select(range(4000))

    def tokenize_data(examples):
        tok = tokenizer(examples['text'], return_tensors='pt', truncation=True)
        examples.update(tok)
        examples.pop('text')
        return examples

    def collate_fn(features):
        input_ids = pad_sequence([torch.LongTensor(f['input_ids']).squeeze() for f in features], batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_mask = pad_sequence([torch.LongTensor(f['attention_mask']).squeeze() for f in features], batch_first=True, padding_value = 0)
        class_label = torch.LongTensor([f['label'] for f in features])
        return {
            "input_ids" : input_ids,
            "class_label" : class_label,
            "attention_mask" : attention_mask
        }
    
    moe_model = Dense(model)

    tokenized_data = DatasetDict({k:v for k,v in data.items() if k != "unsupervised"})
    tokenized_data = data.map(tokenize_data, num_proc=5)


    args = TrainingArguments(output_dir = "./moe_output", remove_unused_columns=False,
                            evaluation_strategy="steps", logging_strategy="steps", logging_steps=1, eval_steps=50, 
                            max_steps=1000,
                            per_device_train_batch_size = 128,
                            per_device_eval_batch_size = 256,
                            learning_rate=float(os.environ.get("LR", 2e-4)),
                            report_to="wandb",
                            run_name=os.environ.get("RUN_NAME","dense_model"),
                            lr_scheduler_type="cosine",
                            warmup_ratio=0.1)

    trainer = CustomTrainer(moe_model, args=args, train_dataset = tokenized_data['train'], eval_dataset = tokenized_data['test'], data_collator=collate_fn)
    trainer.train()