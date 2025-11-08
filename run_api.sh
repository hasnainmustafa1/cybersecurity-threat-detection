#!/bin/bash
export FLASK_APP=src/predict_api.py
python -m flask run --host=0.0.0.0 --port=5000
