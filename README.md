# Cybersecurity Threat Detection System (Synthetic Demo)

This repository is a demo Cybersecurity Threat Detection System using a synthetic network traffic dataset.
It includes preprocessing, model training (Random Forest + SVM), alert generation, and a Flask API for real-time predictions.

## Quickstart

1. Create a virtual environment and activate it:
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate

2. Install dependencies:
   pip install -r requirements.txt

3. Train models:
   python src/train_model.py

4. Generate alerts:
   python src/alert_generation.py

5. Run API:
   python app.py
   # API endpoints:
   # GET /health
   # POST /predict  (JSON body with fields: src_port, dst_port, packet_count, byte_count, flow_duration, protocol)

