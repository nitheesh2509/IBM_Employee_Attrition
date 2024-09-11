# __init__.py
# This file makes src a package.

# Optionally, you can import functions for easy access
from .data_preprocessing import load_data, clean_data, encode_features, split_data, scale_features
from .model import train_model, evaluate_model
