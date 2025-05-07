# Batch Processing for Customer Behavior Analytics
# Description: Uses Apache Spark on AWS EMR to process 100 TB+ historical data
# for customer behavior analytics, stored in Delta Lake.

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, avg
from delta.tables import DeltaTable
import boto3
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Spark session
spark = SparkSession.builder \
    .appName('CustomerBehaviorAnalytics') \
    .config('spark.sql.catalog.spark_catalog', 'org.apache.spark.sql.delta.catalog.DeltaCatalog') \
    .config('spark.sql.extensions', 'io.delta.sql.DeltaSparkSessionExtension') \
    .getOrCreate()

# ------------------- Batch Processing Pipeline -------------------
# Description: Aggregate customer interactions and store results in Delta Lake.

def run_batch_processing():
    """Process historical data for customer behavior analytics."""
    # Read from Delta Lake
    customer_data = spark.read.format('delta').load('s3://telecom-lakehouse/raw/customer_interactions/')
    
    # Example transformation: Aggregate call counts and durations by customer
    analytics = customer_data.groupBy('customer_id').agg(
        count('call_id').alias('total_calls'),
        avg('duration').alias('avg_call_duration')
    )
    
    # Write to Delta Lake (partitioned by date)
    analytics.write.format('delta') \
        .mode('overwrite') \
        .partitionBy('date') \
        .save('s3://telecom-lakehouse/curated/customer_analytics/')
    
    # Optimize Delta table
    delta_table = DeltaTable.forPath(spark, 's3://telecom-lakehouse/curated/customer_analytics/')
    delta_table.vacuum(168)  # Retain 7 days
    delta_table.optimize()

# ------------------- Main Execution -------------------
if __name__ == '__main__':
    run_batch_processing()
    spark.stop()