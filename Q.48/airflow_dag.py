# Airflow DAG for Pipeline Orchestration
# Description: Schedules and orchestrates the MLOps pipeline.

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from mlops_pipeline import run_pipeline

default_args = {
    'owner': 'mlops_team',
    'start_date': datetime(2025, 5, 7),
    'retries': 1,
}

with DAG('mlops_pipeline', default_args=default_args, schedule_interval='@daily') as dag:
    run_task = PythonOperator(
        task_id='run_pipeline',
        python_callable=run_pipeline
    )