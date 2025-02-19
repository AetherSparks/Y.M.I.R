try:
    import atexit
    print("atexit imported successfully")
except ImportError as e:
    print("Error importing atexit:", e)

try:
    import json
    print("json imported successfully")
except ImportError as e:
    print("Error importing json:", e)

try:
    import os
    print("os imported successfully")
except ImportError as e:
    print("Error importing os:", e)

try:
    import cv2
    print("cv2 imported successfully")
except ImportError as e:
    print("Error importing cv2:", e)

try:
    import numpy as np
    print("numpy imported successfully")
except ImportError as e:
    print("Error importing numpy:", e)

try:
    import threading
    print("threading imported successfully")
except ImportError as e:
    print("Error importing threading:", e)

try:
    import time
    print("time imported successfully")
except ImportError as e:
    print("Error importing time:", e)

try:
    import mediapipe as mp
    print("mediapipe imported successfully")
except ImportError as e:
    print("Error importing mediapipe:", e)

try:
    import dlib
    print("dlib imported successfully")
except ImportError as e:
    print("Error importing dlib:", e)

try:
    import warnings
    print("warnings imported successfully")
except ImportError as e:
    print("Error importing warnings:", e)

try:
    from deepface import DeepFace
    print("DeepFace imported successfully")
except ImportError as e:
    print("Error importing DeepFace:", e)

try:
    from concurrent.futures import ThreadPoolExecutor
    print("ThreadPoolExecutor imported successfully")
except ImportError as e:
    print("Error importing ThreadPoolExecutor:", e)

try:
    from flask import Flask, render_template, Response, jsonify, request
    print("Flask and necessary components imported successfully")
except ImportError as e:
    print("Error importing Flask:", e)

try:
    from flask_cors import CORS
    print("flask_cors imported successfully")
except ImportError as e:
    print("Error importing flask_cors:", e)

try:
    from scipy.spatial import distance as dist
    print("scipy.spatial imported successfully")
except ImportError as e:
    print("Error importing scipy.spatial:", e)

try:
    from collections import deque
    print("collections imported successfully")
except ImportError as e:
    print("Error importing collections:", e)

try:
    from transformers import pipeline
    print("transformers pipeline imported successfully")
except ImportError as e:
    print("Error importing transformers:", e)

try:
    from rich.console import Console
    print("rich.console imported successfully")
except ImportError as e:
    print("Error importing rich.console:", e)

try:
    from rich.panel import Panel
    print("rich.panel imported successfully")
except ImportError as e:
    print("Error importing rich.panel:", e)

try:
    import pandas as pd
    print("pandas imported successfully")
except ImportError as e:
    print("Error importing pandas:", e)

try:
    import torch
    print("torch imported successfully")
except ImportError as e:
    print("Error importing torch:", e)

try:
    import requests
    print("requests imported successfully")
except ImportError as e:
    print("Error importing requests:", e)
