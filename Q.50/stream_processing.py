# Stream Processing for Real-Time Fraud Detection
# Description: Uses Apache Flink on AWS Kinesis Data Analytics to process 10 TB/day of streaming data
# for fraud detection with <5s latency. Integrates with Delta Lake for storage.

import boto3
import json
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.table import StreamTableEnvironment, EnvironmentSettings
from pyflink.table.udf import udf
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Flink environment
env = StreamExecutionEnvironment.get_execution_environment()
env_settings = EnvironmentSettings.new_instance().in_streaming_mode().build()
t_env = StreamTableEnvironment.create(env, environment_settings=env_settings)

# ------------------- Fraud Detection Model -------------------
# Description: Simple rule-based model for demonstration (replace with ML model).

@udf(result_type='BOOLEAN')
def detect_fraud(call_record):
    """Detect fraudulent calls based on IP and duration."""
    compromised_ips = ['192.168.1.100', '10.0.0.1']  # Example
    return call_record['source_ip'] in compromised_ips or call_record['duration'] > 3600

# ------------------- Stream Processing Pipeline -------------------
# Description: Ingest data from Kinesis, process with Flink, and sink to Delta Lake.

def setup_stream_pipeline():
    """Configure Flink pipeline for real-time fraud detection."""
    # Define Kinesis source
    t_env.execute_sql("""
        CREATE TABLE call_records (
            call_id STRING,
            source_ip STRING,
            destination_ip STRING,
            duration INT,
            timestamp STRING
        ) WITH (
            'connector' = 'kinesis',
            'stream' = 'call-records-stream',
            'aws.region' = 'us-east-1',
            'format' = 'json'
        )
    """)

    # Define Delta Lake sink
    t_env.execute_sql("""
        CREATE TABLE fraud_alerts (
            call_id STRING,
            source_ip STRING,
            timestamp STRING,
            is_fraud BOOLEAN
        ) WITH (
            'connector' = 'delta',
            'table-path' = 's3://telecom-lakehouse/raw/fraud_alerts/',
            'format' = 'parquet'
        )
    """)

    # Process stream
    call_records = t_env.from_path('call_records')
    fraud_alerts = call_records.select(
        call_records.call_id,
        call_records.source_ip,
        call_records.timestamp,
        detect_fraud(call_records).alias('is_fraud')
    ).where(detect_fraud(call_records))

    # Sink to Delta Lake
    fraud_alerts.execute_insert('fraud_alerts')

# ------------------- Model Serving -------------------
# Description: Serve fraud detection results via Kinesis Streams for downstream apps.

def publish_alerts():
    """Publish fraud alerts to Kinesis for real-time consumption."""
    kinesis_client = boto3.client('kinesis', region_name='us-east-1')
    alerts = t_env.from_path('fraud_alerts').to_pandas()
    for _, row in alerts.iterrows():
        kinesis_client.put_record(
            StreamName='fraud-alerts-stream',
            Data=json.dumps(row.to_dict()),
            PartitionKey=row['call_id']
        )

# ------------------- Main Execution -------------------
if __name__ == '__main__':
    setup_stream_pipeline()
    env.execute('Fraud Detection Pipeline')
    publish_alerts()