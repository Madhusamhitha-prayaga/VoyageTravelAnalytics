FROM apache/airflow:2.8.4-python3.10

USER root

# Install system dependencies as root
RUN apt-get update && \
    apt-get install -y git && \
    apt-get clean

# Switch to airflow user before running pip
USER airflow

# Install Python packages as airflow user
RUN pip install --no-cache-dir --upgrade pip && \
    pip install pandas scikit-learn
