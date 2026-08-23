# verify_pipeline.py
"""
Standalone CI/CD Pipeline Verifier for GitHub Actions.
Starts Flask server in background thread and verifies all 10 endpoints.
"""

import time
import threading
import requests
import sys
import os

# Ensure root workspace directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def run_flask_app():
    try:
        from app import app
        app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)
    except Exception as e:
        print(f"[Flask Server Exception]: {e}")

if __name__ == "__main__":
    print("=========================================================")
    print("  ICC REAL DRS HAWK-EYE 3D -- CI/CD PIPELINE VERIFIER    ")
    print("=========================================================")
    
    # Start Flask server in background thread
    server_thread = threading.Thread(target=run_flask_app, daemon=True)
    server_thread.start()
    time.sleep(7)  # Allow server to complete startup and bind to port

    BASE_URL = "http://127.0.0.1:5000"
    endpoints_to_test = [
        ("GET", "/", "Page 1: Live DRS Review Console"),
        ("GET", "/analytics", "Page 2: Precision Matrix"),
        ("GET", "/records", "Page 3: Historical Decision Database"),
        ("GET", "/admin", "Page 4: Admin Console"),
        ("GET", "/favicon.ico", "SVG Favicon Endpoint"),
        ("GET", "/manifest.json", "Progressive Web App Manifest"),
        ("GET", "/health", "API Health Check"),
        ("GET", "/api/history", "Decision History API"),
        ("GET", "/api/stats", "Session Totals Stats API"),
        ("GET", "/api/spatial_snicko", "Spatial Snicko Beamforming API"),
        ("GET", "/api/camera_calibration", "8-Cam Auto-Calibration API"),
        ("GET", "/api/action_legality", "ICC Action Legality Classifier API"),
        ("GET", "/api/stadium_stream", "WebSocket Stadium Telemetry API"),
        ("GET", "/api/super_slowmo", "2,000 FPS Super Slow-Mo Interpolator API"),
        ("GET", "/api/nerf_3d", "NeRF 3D Scene Reconstruction API"),
        ("GET", "/api/doppler_speed", "Doppler Effect Ball Speed Audio API"),
        ("GET", "/api/fielder_probability", "Fielder Catch Probability AI API"),
        ("GET", "/api/crypto_ledger", "SHA-256 Merkle Ledger Certificate API"),
    ]

    passed = 0
    failed = 0

    for method, path, desc in endpoints_to_test:
        time.sleep(0.2)
        try:
            res = requests.get(f"{BASE_URL}{path}")
            if res.status_code == 200:
                print(f"[PASS] {method} {path} -- 200 OK ({desc})")
                passed += 1
            else:
                print(f"[FAIL] {method} {path} -- Status {res.status_code} ({desc})")
                failed += 1
        except Exception as e:
            print(f"[ERROR] {method} {path} -- {e}")
            failed += 1

    # Test POST /process
    print("\n[Testing DRS Processing Pipeline POST /process]...")
    try:
        res = requests.post(f"{BASE_URL}/process", data={"color_mode": "AUTO"})
        if res.status_code == 200:
            data = res.json()
            print("[PASS] POST /process -- 200 OK")
            print(f"       Final Call: {data.get('final_call')}")
            passed += 1
        else:
            print(f"[FAIL] POST /process -- Status {res.status_code}")
            failed += 1
    except Exception as e:
        print(f"[ERROR] POST /process -- {e}")
        failed += 1

    print("\n=========================================================")
    print(f"  CI/CD PIPELINE VERIFIER: {passed} PASSED | {failed} FAILED")
    print("=========================================================")
    
    if failed > 0:
        sys.exit(1)
    else:
        sys.exit(0)
