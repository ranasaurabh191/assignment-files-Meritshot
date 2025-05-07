# AWS Glue Crawler for Data Cataloging
# Description: Configures AWS Glue to catalog Delta Lake tables and enforce schema.

import boto3
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Glue client
glue_client = boto3.client('glue', region_name='us-east-1')

# ------------------- Data Cataloging -------------------
# Description: Create a Glue crawler to catalog Delta Lake tables.

def create_glue_crawler():
    """Configure and run AWS Glue crawler for Delta Lake."""
    glue_client.create_crawler(
        Name='telecom-lakehouse-crawler',
        Role='arn:aws:iam::account-id:role/AWSGlueServiceRole',
        DatabaseName='telecom_lakehouse',
        Targets={
            'S3Targets': [
                {'Path': 's3://telecom-lakehouse/raw/'},
                {'Path': 's3://telecom-lakehouse/curated/'}
            ]
        },
        TablePrefix='delta_',
        SchemaChangePolicy={
            'UpdateBehavior': 'UPDATE_IN_DATABASE',
            'DeleteBehavior': 'LOG'
        }
    )
    
    glue_client.start_crawler(Name='telecom-lakehouse-crawler')

# ------------------- Main Execution -------------------
if __name__ == '__main__':
    create_glue_crawler()