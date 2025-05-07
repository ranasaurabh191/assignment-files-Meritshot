# MLOps Pipeline for Financial Services

## Overview
This project implements a fully automated MLOps pipeline for a financial services firm, enabling model training, versioning, hyperparameter tuning, interpretability, fairness checks, deployment, and monitoring. The pipeline supports large-scale tabular customer data with PII, complies with GDPR and SOC 2 regulations, and includes CI/CD, auto-scaling, and rollback mechanisms.

### Features
- **Dataset Versioning**: Uses DVC for versioning datasets stored in S3, with PII anonymization.
- **CI/CD**: GitHub Actions for automated testing, building, and deployment.
- **Orchestration**: Airflow schedules daily training and deployment.
- **Model Training**: RandomForest model with MLflow for experiment tracking and Optuna for hyperparameter tuning.
- **Interpretability & Fairness**: SHAP for feature importance and Fairlearn for bias mitigation.
- **Deployment**: Dockerized model deployed on Kubernetes with auto-scaling.
- **Monitoring**: EvidentlyAI for drift detection and Prometheus/Grafana for latency monitoring.
- **Governance**: MLflow ensures reproducibility, with rollback to the best model on drift detection.
- **Compliance**: PII anonymization, encrypted S3 storage, and audit logs for GDPR/SOC 2.

## Prerequisites
- Python 3.9+
- Docker
- Kubernetes (e.g., local Minikube or AWS EKS)
- AWS account with S3 bucket
- GitHub repository with Actions enabled
- Airflow instance
- Prometheus and Grafana for monitoring

## Setup Instructions
1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd mlops-pipeline
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment**
   - Set up AWS credentials for S3 access.
   - Update `mlops_pipeline.py` with your S3 bucket and key names.
   - Configure container registry details in `github_workflow.yml`.
   - Set up Kubernetes secrets for sensitive configurations.

4. **Initialize DVC**
   ```bash
   dvc init
   dvc remote add -d myremote s3://my-bucket/dvc
   ```

5. **Set Up Airflow**
   - Deploy `airflow_dag.py` to your Airflow instance.
   - Configure the DAG to run daily.

6. **Set Up Monitoring**
   - Deploy Prometheus and Grafana.
   - Configure EvidentlyAI reports for drift detection.

7. **CI/CD Configuration**
   - Add secrets (`DOCKER_USERNAME`, `DOCKER_PASSWORD`, `KUBECONFIG`) to GitHub Actions.
   - Push changes to trigger the CI/CD pipeline.

## Usage
1. **Run the Pipeline Locally**
   ```bash
   python mlops_pipeline.py
   ```
   This executes the full pipeline: data loading, training, interpretability, fairness checks, monitoring, and deployment.

2. **Access the Model Service**
   - After deployment, the model is exposed as a RESTful API via Kubernetes LoadBalancer.
   - Example request:
     ```bash
     curl -X POST http://<service-ip>/predict -H "Content-Type: application/json" -d '{"features": [1, 2, 3]}'
     ```

3. **Monitor Performance**
   - View drift reports in MLflow artifacts (`drift_report.html`).
   - Check latency metrics in Grafana.

## Project Structure
```
mlops-pipeline/
├── mlops_pipeline.py       # Main pipeline script
├── airflow_dag.py          # Airflow DAG for orchestration
├── Dockerfile              # Docker configuration for model service
├── github_workflow.yml     # GitHub Actions CI/CD workflow
├── k8s/
│   ├── deployment.yaml     # Kubernetes deployment manifest
│   ├── service.yaml        # Kubernetes service manifest
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── tests/                  # Unit tests (add your own)
```

## Compliance Notes
- **PII Handling**: PII is anonymized using hashing before storage.
- **Encryption**: S3 buckets are encrypted with SSE-KMS.
- **Audit Logs**: MLflow and Airflow maintain logs for traceability.
- **Access Control**: IAM roles restrict S3 and Kubernetes access.

## Extending the Pipeline
- **Hyperparameter Tuning**: Integrate Optuna for automated tuning (add to `train_model`).
- **Transformers**: Use Hugging Face Transformers for advanced models.
- **Multilingual Support**: Extend to support multiple languages with multilingual datasets.

## Troubleshooting
- **Drift Detected**: Check `drift_report.html` for details and verify data quality.
- **Deployment Issues**: Ensure Kubernetes secrets and container registry credentials are correct.
- **CI/CD Failures**: Review GitHub Actions logs for test or build errors.

## License
MIT License. See `LICENSE` for details.

## Contact
For questions, contact the MLOps team at <your-email@example.com>.