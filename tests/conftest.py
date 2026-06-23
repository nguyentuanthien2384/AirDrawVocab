"""Cấu hình pytest: thêm thư mục gốc dự án vào sys.path để import được
config, stroke_features, image_preprocess và các package src.*"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
