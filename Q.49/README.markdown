# AWS SageMaker ML Pipeline for Health-Tech Startup

## Overview
This project implements an end-to-end production-ready ML pipeline on AWS SageMaker for a health-tech startup. The pipeline ingests millions of medical records daily, trains XGBoost and Hugging Face Transformer models, stores features in SageMaker Feature Store, deploys models for batch and real-time inference, monitors performance, and retrains models weekly or on drift detection. The solution is secure, cost-optimized, and compliant with healthcare regulations.

### Features
- **Data Ingestion & Preprocessing**: SageMaker Processing jobs ingest and preprocess medical records from S3, anonymizing PII.
- **Feature Store**: SageMaker Feature Store stores engineered features for reuse.
- **Model Training**: XGBoost for structured data and Hugging Face DistilBERT for text, with hyperparameter tuning and SageMaker Experiments.
- **Deployment**: Dual-path deployment for batch inference (Batch Transform) and real-time inference (SageMaker Endpoint).
- **Monitoring**: SageMaker Model Monitor detects data drift and performance issues.
- **Retraining**: SageMaker Pipelines with CloudWatch Events for weekly or triggered retraining.
- **CI/CD**: CodePipeline and CodeBuild automate pipeline updates.
- **Security**: VPC, encryption, and IAM roles ensure compliance.
- **Cost Optimization**: Spot instances, lifecycle policies, and resource cleanup.

## Prerequisites
- AWS account with SageMaker, S3, CodePipeline, CodeBuild, and CloudWatch permissions
- GitHub repository with OAuth token
- Python 3.9+
- AWS CLI and SageMaker SDK installed
- VPC configured for SageMaker with private subnets

## Setup Instructions
1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd sagemaker-pipeline
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure AWS**
   - Set up an S3 bucket for raw data, processed data, and outputs.
   - Create a SageMaker execution role with permissions for S3, SageMaker, and Feature Store.
   - Configure a VPC with private subnets for SageMaker jobs.

4. **Upload Data**
   - Place medical records in `s3://<bucket>/raw/medical_records/`.
   - Ensure data includes `patient_id`, `age`, `diagnosis_code`, and `text` columns.

5. **Set Up Feature Store**
   - Run `create_feature_group()` in `sagemaker_pipeline.py` to initialize the Feature Store.

6. **Run the Pipeline**
   ```bash
   python sagemaker_pipeline.py
   ```
   This sets up the SageMaker Pipeline, trains models, deploys endpoints, and configures monitoring.

7. **Configure CI/CD**
   - Update `codepipeline.yaml` with your GitHub repo details.
   - Add GitHub OAuth token to AWS Secrets Manager.
   - Deploy the pipeline using AWS CLI:
     ```bash
     aws codepipeline create-pipeline --cli-input-yaml file://codepipeline.yaml
     ```

8. **Set Up Monitoring**
   - Baseline statistics and constraints are generated in the preprocessing step.
   - Model Monitor schedules are created automatically.

## Usage
1. **Batch Inference**
   - Output is stored in `s3://<bucket>/batch_output/`.
   - Trigger manually or via pipeline.

2. **Real-Time Inference**
   - Access the endpoint at `medical-model-endpoint`.
   - Example request:
     ```bash
     aws sagemaker-runtime invoke-endpoint \
       --endpoint-name medical-model-endpoint \
       --body '{"features": [1, 2, 3]}' \
       output.json
     ```

3. **Monitor Performance**
   - Check Model Monitor reports in `s3://<bucket>/monitor_output/`.
   - View metrics in SageMaker Experiments.

## Project Structure
```
sagemaker-pipeline/
├── sagemaker_pipeline.py   # Main pipeline script
├── preprocess.py          # Preprocessing script for SageMaker Processing
├── train_hf.py            # Hugging Face training script
├── codepipeline.yaml      # CodePipeline configuration
├── requirements.txt       # Python dependencies
├── README.md              # This file
└── scripts/               # Directory for training scripts
```

## Security Measures
- **VPC**: All SageMaker jobs run in a VPC with private subnets and security groups.
- **Encryption**: S3 buckets use SSE-KMS, and data in transit uses TLS.
- **IAM Roles**: Least-privilege roles for SageMaker, S3, and Feature Store access.
- **PII Handling**: Patient IDs are anonymized using hashing in preprocessing.
- **Audit Logs**: CloudTrail logs all API calls for compliance.

## Cost Optimization
- **Spot Instances**: Use managed spot training for SageMaker jobs.
- **Lifecycle Policies**: Delete old S3 objects after 30 days.
- **Resource Cleanup**: Terminate idle endpoints and pipelines.
- **Instance Selection**: Use `ml.m5.xlarge` for processing and `ml.p3.2xlarge` for GPU training only when needed.

## Limitations
- **SageMaker Costs**: Training and hosting can be expensive for large-scale workloads.
- **Feature Store Latency**: Online store queries may have higher latency for millions of records.
- **Hugging Face Complexity**: Fine-tuning Transformers requires careful hyperparameter tuning.

## Troubleshooting
- **Pipeline Failures**: Check CloudWatch logs for SageMaker jobs.
- **CI/CD Issues**: Verify GitHub token and CodeBuild permissions.
- **Drift Alerts**: Review Model Monitor reports for data quality issues.

## License
MIT License. See `LICENSE` for details.

## Contact
For questions, contact the ML team at <your-email@example.com>.