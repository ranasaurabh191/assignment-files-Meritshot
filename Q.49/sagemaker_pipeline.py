# AWS SageMaker ML Pipeline for Health-Tech Startup
# This script defines the end-to-end pipeline for data ingestion, preprocessing,
# feature store integration, model training, deployment, monitoring, and retraining
# using SageMaker, XGBoost, and Hugging Face Transformers.

import boto3
import sagemaker
from sagemaker import get_execution_role
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.feature_store.feature_group import FeatureGroup
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.tuner import HyperparameterTuner
from sagemaker.model import Model
from sagemaker.pipeline import PipelineModel
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep, TransformStep, CreateModelStep, EndpointConfigStep, EndpointStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.model_monitor import DataCaptureConfig, ModelMonitor
from sagemaker.huggingface import HuggingFace
import json
import os
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize SageMaker session and role
session = sagemaker.Session()
role = get_execution_role()
region = session.boto_region_name
bucket = session.default_bucket()

# ------------------- Data Ingestion and Preprocessing -------------------
# Description: Use SageMaker Processing to ingest and preprocess medical records from S3.
# Output is stored in S3 and ingested into SageMaker Feature Store.

def create_preprocessing_job():
    """Define a SageMaker Processing job for data ingestion and preprocessing."""
    processor = ScriptProcessor(
        image_uri=sagemaker.image_uris.retrieve('sklearn', region, '0.23-1'),
        command=['python3'],
        instance_type='ml.m5.xlarge',
        instance_count=2,
        role=role
    )
    
    processor.run(
        code='preprocess.py',
        inputs=[ProcessingInput(source=f's3://{bucket}/raw/medical_records/', destination='/opt/ml/processing/input')],
        outputs=[
            ProcessingOutput(output_name='train', source='/opt/ml/processing/train', destination=f's3://{bucket}/processed/train/'),
            ProcessingOutput(output_name='test', source='/opt/ml/processing/test', destination=f's3://{bucket}/processed/test/')
        ]
    )
    return processor

# ------------------- Feature Store Integration -------------------
# Description: Store engineered features in SageMaker Feature Store for reuse.

def create_feature_group():
    """Create and ingest data into SageMaker Feature Store."""
    feature_group_name = 'medical-records-features'
    feature_group = FeatureGroup(name=feature_group_name, sagemaker_session=session)
    
    # Define feature definitions (example)
    feature_definitions = [
        {'FeatureName': 'patient_id', 'FeatureType': 'String'},
        {'FeatureName': 'age', 'FeatureType': 'Fractional'},
        {'FeatureName': 'diagnosis_code', 'FeatureType': 'String'},
        {'FeatureName': 'record_time', 'FeatureType': 'String'}
    ]
    
    feature_group.load_feature_definitions(data_frame=None)  # Define schema
    feature_group.create(
        s3_uri=f's3://{bucket}/feature_store/',
        record_identifier_name='patient_id',
        event_time_feature_name='record_time',
        role_arn=role,
        enable_online_store=True
    )
    
    # Ingest data (handled in preprocess.py)
    return feature_group_name

# ------------------- Model Training (XGBoost and Hugging Face) -------------------
# Description: Train XGBoost for structured data and Hugging Face Transformer for text data.
# Use SageMaker Experiments for tracking and Debugger for profiling.

def create_xgboost_training_job():
    """Define an XGBoost training job with hyperparameter tuning."""
    xgboost_image = sagemaker.image_uris.retrieve('xgboost', region, '1.5-1')
    estimator = Estimator(
        image_uri=xgboost_image,
        role=role,
        instance_count=1,
        instance_type='ml.m5.4xlarge',
        output_path=f's3://{bucket}/output/xgboost/',
        sagemaker_session=session,
        enable_sagemaker_metrics=True
    )
    
    estimator.set_hyperparameters(
        num_round=100,
        objective='binary:logistic',
        eval_metric='auc'
    )
    
    # Hyperparameter tuning
    tuner = HyperparameterTuner(
        estimator=estimator,
        objective_metric_name='validation:auc',
        hyperparameter_ranges={
            'max_depth': sagemaker.tuner.IntegerParameter(3, 10),
            'eta': sagemaker.tuner.ContinuousParameter(0.1, 0.5)
        },
        max_jobs=10,
        max_parallel_jobs=3
    )
    
    tuner.fit({
        'train': TrainingInput(s3_data=f's3://{bucket}/processed/train/', content_type='csv'),
        'validation': TrainingInput(s3_data=f's3://{bucket}/processed/test/', content_type='csv')
    })
    
    return tuner

def create_huggingface_training_job():
    """Define a Hugging Face Transformer training job."""
    huggingface_estimator = HuggingFace(
        entry_point='train_hf.py',
        source_dir='scripts/',
        role=role,
        instance_count=1,
        instance_type='ml.p3.2xlarge',
        transformers_version='4.26',
        pytorch_version='1.13',
        py_version='py39',
        hyperparameters={
            'model_name': 'distilbert-base-uncased',
            'num_train_epochs': 3,
            'per_device_train_batch_size': 16
        },
        output_path=f's3://{bucket}/output/huggingface/',
        sagemaker_session=session
    )
    
    huggingface_estimator.fit({
        'train': TrainingInput(s3_data=f's3://{bucket}/processed/train/', content_type='text/csv'),
        'test': TrainingInput(s3_data=f's3://{bucket}/processed/test/', content_type='text/csv')
    })
    
    return huggingface_estimator

# ------------------- Model Deployment (Batch and Real-Time) -------------------
# Description: Deploy models for batch inference (Batch Transform) and real-time inference (SageMaker Endpoint).

def deploy_batch_transform(estimator):
    """Deploy model for batch inference."""
    transformer = estimator.transformer(
        instance_count=1,
        instance_type='ml.m5.xlarge',
        output_path=f's3://{bucket}/batch_output/',
        accept='text/csv'
    )
    
    transformer.transform(
        data=f's3://{bucket}/processed/test/',
        content_type='text/csv',
        split_type='Line'
    )
    
    return transformer

def deploy_endpoint(estimator):
    """Deploy model to a real-time SageMaker Endpoint."""
    model = estimator.create_model()
    endpoint_config = model.deploy(
        initial_instance_count=1,
        instance_type='ml.m5.xlarge',
        data_capture_config=DataCaptureConfig(
            enable_capture=True,
            sampling_percentage=100,
            destination_s3_uri=f's3://{bucket}/data_capture/'
        )
    )
    return endpoint_config

# ------------------- Model Monitoring -------------------
# Description: Use SageMaker Model Monitor for data drift and performance monitoring.

def setup_model_monitor(endpoint_name):
    """Set up SageMaker Model Monitor for drift detection."""
    monitor = ModelMonitor(
        role=role,
        instance_count=1,
        instance_type='ml.m5.xlarge',
        output_s3_uri=f's3://{bucket}/monitor_output/',
        sagemaker_session=session
    )
    
    monitor.create_monitoring_schedule(
        endpoint_name=endpoint_name,
        schedule_expression='cron(0 0 * * ? *)',  # Daily
        statistics=MetricsSource(
            s3_uri=f's3://{bucket}/processed/test/baseline_stats.json',
            content_type='application/json'
        ),
        constraints=MetricsSource(
            s3_uri=f's3://{bucket}/processed/test/baseline_constraints.json',
            content_type='application/json'
        )
    )
    
    return monitor

# ------------------- SageMaker Pipeline -------------------
# Description: Define a SageMaker Pipeline for end-to-end workflow with conditional deployment.

def create_sagemaker_pipeline():
    """Define SageMaker Pipeline for preprocessing, training, and deployment."""
    # Preprocessing step
    preprocess_step = ProcessingStep(
        name='PreprocessData',
        processor=create_preprocessing_job(),
        outputs=[
            ProcessingOutput(output_name='train', source='/opt/ml/processing/train'),
            ProcessingOutput(output_name='test', source='/opt/ml/processing/test')
        ]
    )
    
    # Training step (XGBoost)
    xgboost_step = TrainingStep(
        name='TrainXGBoost',
        estimator=create_xgboost_training_job().estimator
    )
    
    # Model creation
    model_step = CreateModelStep(
        name='CreateModel',
        model=Model(
            model_data=xgboost_step.properties.ModelArtifacts.S3ModelArtifacts,
            role=role,
            sagemaker_session=session
        )
    )
    
    # Conditional deployment (deploy if AUC > 0.8)
    condition_step = ConditionStep(
        name='CheckAUC',
        conditions=[ConditionGreaterThanOrEqualTo(
            left=xgboost_step.properties.FinalMetricDataList['validation:auc'].Value,
            right=0.8
        )],
        if_steps=[
            EndpointConfigStep(
                name='CreateEndpointConfig',
                endpoint_config_name='medical-model-endpoint',
                model_name=model_step.properties.ModelName
            ),
            EndpointStep(
                name='DeployEndpoint',
                endpoint_name='medical-model-endpoint'
            )
        ],
        else_steps=[]
    )
    
    # Define pipeline
    pipeline = Pipeline(
        name='MedicalMLPipeline',
        steps=[preprocess_step, xgboost_step, model_step, condition_step],
        sagemaker_session=session
    )
    
    pipeline.upsert(role_arn=role)
    pipeline.start()
    
    return pipeline

# ------------------- CI/CD with CodePipeline and CodeBuild -------------------
# Description: Automate pipeline updates using CodePipeline and CodeBuild.

def setup_cicd():
    """Configure CodePipeline and CodeBuild (defined in YAML below)."""
    # CodeBuild project configuration
    codebuild_client = boto3.client('codebuild')
    codebuild_client.create_project(
        name='SageMakerPipelineBuild',
        source={
            'Type': 'GITHUB',
            'Location': '<your-github-repo-url>'
        },
        artifacts={'Type': 'S3', 'Location': bucket},
        environment={
            'Type': 'LINUX_CONTAINER',
            'Image': 'aws/codebuild/standard:5.0',
            'ComputeType': 'BUILD_GENERAL1_SMALL'
        },
        service_role=role
    )
    
    # CodePipeline configuration (see codepipeline.yaml)
    codepipeline_client = boto3.client('codepipeline')
    with open('codepipeline.yaml', 'r') as f:
        pipeline_def = yaml.safe_load(f)
    codepipeline_client.create_pipeline(pipeline=pipeline_def)

# ------------------- Automatic Retraining Trigger -------------------
# Description: Use CloudWatch Events to trigger weekly retraining or on drift detection.

def setup_retraining_trigger():
    """Set up CloudWatch Events for weekly retraining."""
    events_client = boto3.client('events')
    events_client.put_rule(
        Name='WeeklyRetraining',
        ScheduleExpression='cron(0 0 ? * MON *)',  # Every Monday
        State='ENABLED'
    )
    
    events_client.put_targets(
        Rule='WeeklyRetraining',
        Targets=[{
            'Id': 'SageMakerPipeline',
            'Arn': f'arn:aws:sagemaker:{region}:{session.account_id}:pipeline/MedicalMLPipeline',
            'RoleArn': role
        }]
    )

# ------------------- Main Execution -------------------
# Description: Run the full pipeline.

def run_pipeline():
    """Execute the end-to-end pipeline."""
    # Create feature group
    feature_group_name = create_feature_group()
    
    # Run SageMaker Pipeline
    pipeline = create_sagemaker_pipeline()
    
    # Deploy batch transform
    xgboost_estimator = create_xgboost_training_job().best_estimator
    deploy_batch_transform(xgboost_estimator)
    
    # Deploy real-time endpoint
    endpoint_config = deploy_endpoint(xgboost_estimator)
    
    # Set up monitoring
    setup_model_monitor('medical-model-endpoint')
    
    # Configure CI/CD
    setup_cicd()
    
    # Set up retraining trigger
    setup_retraining_trigger()

if __name__ == '__main__':
    run_pipeline()