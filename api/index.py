import os
import sys

# Ensure repository root and drs_opencv are on sys.path for Vercel serverless environment
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

drs_dir = os.path.join(root_dir, 'drs_opencv')
if drs_dir not in sys.path:
    sys.path.insert(0, drs_dir)

try:
    from app import app
except Exception as e:
    from flask import Flask, jsonify
    app = Flask(__name__)
    @app.route('/')
    @app.route('/<path:path>')
    def error_fallback(path=''):
        return jsonify({'error': 'Serverless Startup Error', 'details': str(e)}), 500
