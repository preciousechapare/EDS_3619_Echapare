# Automated ECG-Based Heart Rate Recovery Analytics in MHEALTH Wearable Sensor Telemetry Using NumPy-Driven Python Pipelines

## Project Overview

This project implements a Python-based biomedical data analytics pipeline for the BIO-01 topic under Pillar 9: Biomedical & Wearable Systems. The project focuses on heart rate recovery analysis using ECG-related wearable sensor telemetry from the MHEALTH dataset.

The system performs automated data ingestion, cleaning, statistical analysis, comparative analysis, correlation mapping, static visualization, and animated signal visualization. The pipeline was developed using a modular Object-Oriented Programming structure to satisfy engineering data systems requirements.

## Assigned Topic

- Pillar: Pillar 9 — Biomedical & Wearable Systems
- Topic ID: BIO-01
- Topic: Heart Rate Recovery
- Dataset Search Theme: Wearable Sensor Data ECG

## Dataset

- Dataset Name: MHEALTH Dataset CSV
- Dataset Source: Kaggle
- Dataset Link: https://www.kaggle.com/datasets/nirmalsankalana/mhealth-dataset-data-set-csv
- Main Data Slice Used: Subject 1 wearable biomedical telemetry
- Unique Filter Logic: The pipeline uses Subject 1 ECG-related wearable sensor data as the unique dataset slice. ECG signal behavior is analyzed across physical activity labels to evaluate recovery-related biomedical telemetry patterns.

## Project Structure

```txt
EDS_3619_Echapare
├── main.py
├── requirements.txt
├── README.md
├── data
│   ├── dataset_original.csv
│   └── dataset_cleaned.csv
└── outputs
    ├── ecg_statistics.csv
    ├── activity_comparison.csv
    ├── correlation_matrix.csv
    ├── static_1_ecg_histogram.png
    ├── static_2_ecg_boxplot.png
    ├── static_3_correlation_heatmap.png
    ├── static_4_ecg_activity_scatter.png
    ├── animation_1_ecg_trend.gif
    └── animation_2_ecg_activity.gif