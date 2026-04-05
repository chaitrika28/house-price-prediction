# 🏠 House Price Prediction

Predicts house prices in Indian Rupees using Machine Learning + Django backend.

## Tech Stack
- Python 3.12
- scikit-learn (Linear Regression)
- Django + Django REST Framework
- California Housing Dataset

## Setup Instructions

### 1. Install dependencies
pip install -r requirements.txt

### 2. Train the model
python ml/train.py

### 3. Run Django server
cd pricepredict
python manage.py runserver

### 4. Open browser
Go to: http://127.0.0.1:8000/predict/

## Features
- Predicts house price based on income, rooms, location, etc.
- Output in Indian Rupees (INR)
- Clean web UI

## Made by Chaitrika
