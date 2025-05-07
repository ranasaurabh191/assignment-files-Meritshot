# Preprocessing Script for SageMaker Processing Job
# Description: Ingests medical records, anonymizes PII, and prepares data for training.

import pandas as pd
import os
import argparse
from sklearn.preprocessing import StandardScaler

def anonymize_pii(df):
    """Anonymize PII columns (e.g., patient_id)."""
    if 'patient_id' in df.columns:
        df['patient_id'] = df['patient_id'].apply(lambda x: hash(str(x)) % 1000000)
    return df

def preprocess(input_path, output_train_path, output_test_path):
    """Preprocess medical records and split into train/test."""
    df = pd.read_csv(os.path.join(input_path, 'medical_records.csv'))
    df = anonymize_pii(df)
    
    # Feature engineering (example)
    df['age_normalized'] = StandardScaler().fit_transform(df[['age']])
    
    # Split data
    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)
    
    # Save outputs
    os.makedirs(output_train_path, exist_ok=True)
    os.makedirs(output_test_path, exist_ok=True)
    train_df.to_csv(os.path.join(output_train_path, 'train.csv'), index=False)
    test_df.to_csv(os.path.join(output_test_path, 'test.csv'), index=False)
    
    # Ingest into Feature Store (simplified)
    feature_store_client = boto3.client('sagemaker-featurestore-runtime')
    for _, row in train_df.iterrows():
        feature_store_client.put_record(
            FeatureGroupName='medical-records-features',
            Record=[{'FeatureName': k, 'ValueAsString': str(v)} for k, v in row.items()]
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-data', type=str, default='/opt/ml/processing/input')
    parser.add_argument --

System: It looks like the `preprocess.py` script was cut off mid-line. Below, I’ll provide the corrected and complete version of the `preprocess.py` script, along with additional artifacts to fully address the requirements for the AWS SageMaker ML pipeline. These include the `train_hf.py` script for Hugging Face training, a `codepipeline.yaml` for CI/CD configuration, and a comprehensive README to tie everything together. Each artifact is wrapped in the required `<xaiArtifact>` tags, with unique `artifact_id` values for new artifacts and the same `artifact_id` for the updated `preprocess.py`. The README provides setup and usage instructions, while the code includes detailed comments explaining each component.

---

<xaiArtifact artifact_id="cb2ce5bf-9b13-468b-a2fa-4b4234e31e95" artifact_version_id="e3fd1bb8-3254-4aac-a91c-b8bed63ceef9" title="preprocess.py" contentType="text/python">
# Preprocessing Script for SageMaker Processing Job
# Description: Ingests medical records, anonymizes PII, and prepares data for training.
# Outputs train/test datasets to S3 and ingests features into SageMaker Feature Store.

import pandas as pd
import os
import argparse
from sklearn.preprocessing import StandardScaler
import boto3
from datetime import datetime

def anonymize_pii(df):
    """Anonymize PII columns (e.g., patient_id)."""
    if 'patient_id' in df.columns:
        df['patient_id'] = df['patient_id'].apply(lambda x: hash(str(x)) % 1000000)
    return df

def preprocess(input_path, output_train_path, output_test_path):
    """Preprocess medical records and split into train/test."""
    # Load data
    df = pd.read_csv(os.path.join(input_path, 'medical_records.csv'))
    df = anonymize_pii(df)
    
    # Feature engineering (example)
    df['age_normalized'] = StandardScaler().fit_transform(df[['age']])
    df['record_time'] = datetime.now().isoformat()  # Add event time for Feature Store
    
    # Split data
    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)
    
    # Save outputs
    os.makedirs(output_train_path, exist_ok=True)
    os.makedirs(output_test_path, exist_ok=True)
    train_df.to_csv(os.path.join(output_train_path, 'train.csv'), index=False)
    test_df.to_csv(os.path.join(output_test_path, 'test.csv'), index=False)
    
    # Ingest into Feature Store
    feature_store_client = boto3.client('sagemaker-featurestore-runtime')
    for _, row in train_df.iterrows():
        record = [
            {'FeatureName': k, 'ValueAsString': str(v)}
            for k, v in row.items()
            if k in ['patient_id', 'age_normalized', 'diagnosis_code', 'record_time']
        ]
        feature_store_client.put_record(
            FeatureGroupName='medical-records-features',
            Record=record
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-data', type=str, default='/opt/ml/processing/input')
    parser.add_argument('--output-train', type=str, default='/opt/ml/processing/train')
    parser.add_argument('--output-test', type=str, default='/opt/ml/processing/test')
    args = parser.parse_args()
    
    preprocess(args.input_data, args.output_train, args.output_test)