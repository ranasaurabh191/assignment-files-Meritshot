# ML Pipeline for Fraud Detection Model Training
# Description: Trains an ML model using SageMaker, integrating with Delta Lake data.

import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model import Model
from sagemaker.model_monitor import DataCaptureConfig
from delta.tables import DeltaTable
from pyspark.sql import SparkSession
import boto3
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize SageMaker session
session = sagemaker.Session()
role = sagemaker.get_execution_role()
bucket = session.default_bucket()

# ------------------- Data Preparation -------------------
# Description: Extract features from Delta Lake for ML training.

def prepare_training_data():
    """Prepare training data from Delta Lake."""
    spark = SparkSession.builder \
        .appName('MLDataPrep') \
        .config('spark.sql.catalog.spark_catalog', 'org.apache.spark.sql.delta.catalog.DeltaCatalog') \
        .getOrCreate()
    
    # Read fraud alerts
    data = spark.read.format('delta').load('s3://telecom-lakehouse/raw/fraud_alerts/')
    data.write.mode('overwrite').save(f's3://{bucket}/training_data/')
    
    spark.stop()
    return f's3://{bucket}/training_data/'

# ------------------- Model Training and Deployment -------------------
# Description: Train and deploy an XGBoost model for fraud detection.

def train_and_deploy_model():
    """Train and deploy fraud detection model using SageMaker."""
    estimator = Estimator(
        image_uri=sagemaker.image_uris.retrieve('xgboost', session.boto_region_name, '1.5-1'),
        role=role,
        instance_count=1,
        instance_type='ml.m5.xlarge',
        output_path=f's3://{bucket}/output/',
        sagemaker_session=session
    )
    
    estimator.set_hyperparameters(
        num_round=100,
        objective='binary:logistic',
        eval_metric='auc'
    )
    
    estimator.fit({
        'train': TrainingInput(s3_data=prepare_training_data(), content_type='csv')
    })
    
    # Deploy to SageMaker Endpoint
    model = estimator.create_model()
    model.deploy(
        initial_instance_count=1,
        instance_type='ml.m5.xlarge',
        data_capture_config=DataCaptureConfig(
            enable_capture=True,
            sampling_percentage=100,
            destination_s3_uri=f's3://{bucket}/data_capture/'
        )
    )

# ------------------- Main Execution -------------------
if __name__ == '__main__':
    train_and_deploy_model()