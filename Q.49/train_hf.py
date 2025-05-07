# Hugging Face Training Script for SageMaker
# Description: Trains a DistilBERT model on medical text data for classification.

import os
import argparse
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch

def load_data(train_path, test_path):
    """Load and prepare datasets."""
    train_df = pd.read_csv(os.path.join(train_path, 'train.csv'))
    test_df = pd.read_csv(os.path.join(test_path, 'test.csv'))
    
    # Assume 'text' column for input and 'target' for labels
    train_dataset = Dataset.from_pandas(train_df[['text', 'target']])
    test_dataset = Dataset.from_pandas(test_df[['text', 'target']])
    return train_dataset, test_dataset

def tokenize_data(dataset, tokenizer):
    """Tokenize text data."""
    return dataset.map(
        lambda x: tokenizer(x['text'], truncation=True, padding='max_length', max_length=128),
        batched=True
    )

def train_model(args):
    """Train DistilBERT model using Hugging Face Trainer."""
    # Load data
    train_dataset, test_dataset = load_data(args.train_data, args.test_data)
    
    # Initialize tokenizer and model
    tokenizer = DistilBertTokenizer.from_pretrained(args.model_name)
    model = DistilBertForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
    
    # Tokenize datasets
    train_dataset = tokenize_data(train_dataset, tokenizer)
    test_dataset = tokenize_data(test_dataset, tokenizer)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        logging_dir=f'{args.output_dir}/logs',
        report_to='none'  # SageMaker handles metrics
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )
    
    # Train model
    trainer.train()
    
    # Save model
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str, default='distilbert-base-uncased')
    parser.add_argument('--train-data', type=str, default='/opt/ml/input/data/train')
    parser.add_argument('--test-data', type=str, default='/opt/ml/input/data/test')
    parser.add_argument('--output-dir', type=str, default='/opt/ml/model')
    parser.add_argument('--num-train-epochs', type=int, default=3)
    parser.add_argument('--per-device-train-batch-size', type=int, default=16)
    args = parser.parse_args()
    
    train_model(args)