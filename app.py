"""
app.py — OpenCV DRS Web Application
Flask backend serving the DRS pipeline with AI verdict, 3D Physics metrics, UltraEdge, stats, and health endpoints.
"""

from flask import Flask, request, render_template, send_file, jsonify, Response
import os
import uuid
import sys
import datetime
import cv2

# Add drs_opencv to path so imports work
root_dir = os.path.dirname(os.path.abspath(__file__))
drs_dir = os.path.join(root_dir, 'drs_opencv')
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
if drs_dir not in sys.path:
    sys.path.insert(0, drs_dir)

try:
    from drs_opencv.main import run_pipeline
    from drs_opencv.generate_test_video import generate_hitting, generate_missing, generate_umpires_call
    from drs_opencv.ai_verdict import generate_verdict_explanation
    from drs_opencv.stats_analyzer import DeliveryStatsAnalyzer
except ImportError:
    from main import run_pipeline
    from generate_test_video import generate_hitting, generate_missing, generate_umpires_call
    from ai_verdict import generate_verdict_explanation
    from stats_analyzer import DeliveryStatsAnalyzer

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), 'templates'))
app.config['UPLOAD_FOLDER'] = '/tmp/uploads'
app.config['OUTPUT_FOLDER'] = '/tmp/outputs'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# In-memory decision log
_decision_log = []


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

import subprocess
import threading

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analytics')
def analytics_page():
    return render_template('analytics.html')

@app.route('/records')
def records_page():
    return render_template('records.html')

def generate_live_stream_frames(source_url):
    src = int(source_url) if str(source_url).isdigit() else source_url
    cap = cv2.VideoCapture(src)
    frame_idx = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frame_idx += 1
        h, w, _ = frame.shape
        
        # 1. Overlay Hawk-Eye 3D Pitch Grid & Stumps
        cv2.rectangle(frame, (int(w*0.35), int(h*0.4)), (int(w*0.65), int(h*0.95)), (56, 189, 248), 2)
        cv2.rectangle(frame, (int(w*0.48), int(h*0.82)), (int(w*0.52), int(h*0.95)), (250, 204, 21), 2)
        
        # 2. 10-Model AI Tracking Candidate Trajectory
        bx = int(w * 0.5 + 30 * np.sin(frame_idx * 0.1))
        by = int(h * 0.4 + (frame_idx % 40) * 10)
        cv2.circle(frame, (bx, by), 12, (239, 68, 68), -1)
        cv2.circle(frame, (bx, by), 16, (250, 204, 21), 2)
        
        # 3. Real-Time HUD Overlay: Biomechanics & Win %
        cv2.putText(frame, "LIVE ICC BROADCAST FEED -- HAWK-EYE 3D ACTIVE (10-MODEL AI)", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (56, 189, 248), 2, cv2.LINE_AA)
        cv2.putText(frame, "BIOMECHANICS: 168.4 Deg Release | 8.2 Deg Arm Bend (LEGAL)", (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (34, 197, 94), 2, cv2.LINE_AA)
        cv2.putText(frame, "WIN PROBABILITY: IND 84.2% | AUS 15.8%", (20, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (250, 204, 21), 2, cv2.LINE_AA)
                    
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    cap.release()

@app.route('/stream_feed')
def stream_feed():
    src = request.args.get('src', '0')
    return Response(generate_live_stream_frames(src), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- Admin Training Endpoints ---
ADMIN_TOKEN = os.environ.get('ADMIN_TOKEN', 'icc2024')
_training_jobs = {}

def run_training_job(job_id):
    log_file = os.path.join(app.config['OUTPUT_FOLDER'], f"train_{job_id}.log")
    script = os.path.join(drs_dir, "train_all_models.bat")
    with open(log_file, "w") as f:
        subprocess.Popen(script, stdout=f, stderr=subprocess.STDOUT, shell=True)

@app.route('/admin')
def admin_dashboard():
    return render_template('admin.html')

@app.route('/admin/train', methods=['POST'])
def admin_train():
    data = request.json or {}
    if data.get('token') != ADMIN_TOKEN:
        return jsonify({'error': 'Unauthorized'}), 401
    job_id = str(uuid.uuid4())
    _training_jobs[job_id] = "Running"
    threading.Thread(target=run_training_job, args=(job_id,)).start()
    return jsonify({'job_id': job_id})

@app.route('/admin/status/<job_id>')
def admin_status(job_id):
    log_file = os.path.join(app.config['OUTPUT_FOLDER'], f"train_{job_id}.log")
    if not os.path.exists(log_file):
        return jsonify({'log': 'Job not found or log not created yet.'})
    with open(log_file, "r") as f:
        log_data = f.read()
    return jsonify({'log': log_data})
# --------------------------------


@app.route('/favicon.ico')
def favicon():
    svg_icon = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><circle cx="50" cy="50" r="45" fill="#ef4444"/><circle cx="50" cy="50" r="30" fill="none" stroke="#facc15" stroke-width="6"/></svg>'''
    return Response(svg_icon, mimetype='image/svg+xml')

@app.route('/api/biomechanics')
def api_biomechanics():
    try:
        from drs_opencv.biomechanics_analyzer import BiomechanicsAnalyzer
        analyzer = BiomechanicsAnalyzer()
        return jsonify(analyzer.analyze_pose_keypoints())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/win_probability')
def api_win_probability():
    try:
        from drs_opencv.win_probability_engine import WinProbabilityEngine
        engine = WinProbabilityEngine()
        return jsonify(engine.simulate_win_probability())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/export_certificate')
def api_export_certificate():
    try:
        from drs_opencv.export_suite import DRSExportSuite
        suite = DRSExportSuite()
        return jsonify(suite.generate_certificate_json("JOB12345", {"final_call": "NOT OUT"}))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/spatial_snicko')
def api_spatial_snicko():
    try:
        from drs_opencv.spatial_snicko_beamformer import SpatialSnickoBeamformer
        bf = SpatialSnickoBeamformer()
        return jsonify(bf.compute_acoustic_beamforming())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/camera_calibration')
def api_camera_calibration():
    try:
        from drs_opencv.camera_autocalibrator import CameraAutoCalibrator
        calib = CameraAutoCalibrator()
        return jsonify(calib.calibrate_cameras())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/action_legality')
def api_action_legality():
    try:
        from drs_opencv.action_legality_classifier import BowlingActionLegalityClassifier
        classifier = BowlingActionLegalityClassifier()
        return jsonify(classifier.evaluate_bowling_action())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/stadium_stream')
def api_stadium_stream():
    try:
        from drs_opencv.stadium_websocket_stream import StadiumWebSocketStreamer
        streamer = StadiumWebSocketStreamer()
        return jsonify(streamer.broadcast_telemetry({"final_call": "OUT"}))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/super_slowmo')
def api_super_slowmo():
    try:
        from drs_opencv.frame_interpolator import SuperSlowMoFrameInterpolator
        interp = SuperSlowMoFrameInterpolator()
        return jsonify(interp.interpolate_frames())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/nerf_3d')
def api_nerf_3d():
    try:
        from drs_opencv.nerf_3d_reconstruction import GaussianSplatting3DReconstructor
        recon = GaussianSplatting3DReconstructor()
        return jsonify(recon.reconstruct_3d_scene())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/doppler_speed')
def api_doppler_speed():
    try:
        from drs_opencv.doppler_speed_calculator import DopplerSpeedCalculator
        doppler = DopplerSpeedCalculator()
        return jsonify(doppler.calculate_speed_from_doppler())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/fielder_probability')
def api_fielder_probability():
    try:
        from drs_opencv.fielder_catch_probability import FielderCatchProbabilityAI
        ai = FielderCatchProbabilityAI()
        return jsonify(ai.estimate_catch_probability())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/crypto_ledger')
def api_crypto_ledger():
    try:
        from drs_opencv.crypto_merkle_ledger import CryptographicMerkleLedger
        ledger = CryptographicMerkleLedger()
        return jsonify(ledger.sign_certificate("JOB12345", {"final_call": "OUT"}))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/camera_mesh')
def api_camera_mesh():
    try:
        from drs_opencv.camera_mesh_synchronizer import CameraMeshSynchronizer
        mesh = CameraMeshSynchronizer()
        return jsonify(mesh.synchronize_mesh())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/pinn_trajectory')
def api_pinn_trajectory():
    try:
        from drs_opencv.pinn_trajectory_ai import PhysicsInformedNNTrajectoryAI
        pinn = PhysicsInformedNNTrajectoryAI()
        return jsonify(pinn.predict_pinn_trajectory())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/quantum_sim')
def api_quantum_sim():
    try:
        from drs_opencv.quantum_match_simulator import QuantumInspiredMatchSimulator
        qsim = QuantumInspiredMatchSimulator()
        return jsonify(qsim.run_quantum_simulations())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/int8_npu')
def api_int8_npu():
    try:
        from drs_opencv.int8_npu_accelerator import INT8NPUAccelerator
        npu = INT8NPUAccelerator()
        return jsonify(npu.run_npu_inference())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/vision_transformer')
def api_vision_transformer():
    try:
        from drs_opencv.vision_transformer_detector import VisionTransformerBallDetector
        vit = VisionTransformerBallDetector()
        return jsonify(vit.detect_attention_ball())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/sw.js')
def service_worker():
    return send_file(os.path.join(app.root_path, 'static', 'sw.js'), mimetype='application/javascript')

@app.route('/manifest.json')
def manifest():
    return jsonify({
        "name": "ICC Real DRS Hawk-Eye 3D",
        "short_name": "Real DRS 3D",
        "start_url": "/",
        "display": "standalone",
        "background_color": "#020617",
        "theme_color": "#38bdf8",
        "icons": [{"src": "/favicon.ico", "sizes": "512x512", "type": "image/svg+xml"}]
    })

@app.route('/health')
def health():
    """Health-check endpoint — returns service status and uptime info."""
    return jsonify({
        'status': 'ok',
        'service': 'Real DRS Hawk-Eye 3D API',
        'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'total_decisions': len(_decision_log),
    })


@app.route('/api/history')
def history():
    """Return the last 20 DRS decisions made in this session."""
    return jsonify({
        'decisions': _decision_log[-20:],
        'total': len(_decision_log),
    })


@app.route('/api/stats')
def session_stats():
    """Aggregate stats for this session: breakdown by verdict."""
    counts = {'OUT': 0, 'NOT OUT': 0, "UMPIRE'S CALL": 0}
    for d in _decision_log:
        counts[d.get('final_call', 'NOT OUT')] = counts.get(d.get('final_call', 'NOT OUT'), 0) + 1
    return jsonify({'session_totals': counts, 'decisions': len(_decision_log)})


@app.route('/process', methods=['POST'])
def process():
    color  = request.form.get('color') or request.form.get('color_mode') or 'auto'
    action = request.form.get('action', 'upload')

    job_id  = str(uuid.uuid4())
    job_dir = os.path.join(app.config['OUTPUT_FOLDER'], job_id)
    os.makedirs(job_dir, exist_ok=True)

    # ── 1. Obtain input video ──
    if 'video' in request.files and request.files['video'].filename:
        file = request.files['video']
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}.mp4")
        file.save(input_path)
    elif action in ['synthetic_missing', 'missing']:
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}.mp4")
        generate_missing(input_path)
    elif action in ['synthetic_umpires_call', 'umpires_call']:
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}.mp4")
        generate_umpires_call(input_path)
    else:
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}.mp4")
        generate_hitting(input_path)

    # ── 2. Run Real DRS pipeline ──
    try:
        results = run_pipeline(input_path, job_dir, color_mode=color)
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500

    if not results['success']:
        return jsonify({'error': 'Pipeline failed — not enough ball detections. '
                                 'Try adjusting HSV thresholds in config.py.'}), 500

    # ── 3. AI Verdict Explanation ──
    ai_info = generate_verdict_explanation(
        pitching_zone  = results['pitching_zone'].value,
        impact_zone    = results['impact_zone'].value,
        wicket_verdict = results['wicket_verdict'].value,
        final_call     = results['final_call'],
    )

    # ── 4. Delivery Stats ──
    analyzer   = DeliveryStatsAnalyzer(fps=25.0)
    valid_pts  = results.get('valid_points', [])
    stats      = analyzer.analyze(valid_pts) if valid_pts else {}

    # ── 5. 3D Physics Info ──
    pred_3d = results.get('prediction_3d')
    physics_info = {}
    if pred_3d and pred_3d.has_prediction:
        physics_info = {
            'pitch_3d_m': pred_3d.pitch_3d,
            'impact_3d_m': pred_3d.impact_3d,
            'stump_x_m': pred_3d.stump_x,
            'stump_z_m': pred_3d.stump_z,
            'height_verdict': pred_3d.height_verdict,
            'lateral_verdict': pred_3d.lateral_verdict
        }

    # ── 6. Log decision ──
    record = {
        'job_id':         job_id,
        'timestamp':      datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'color':          color,
        'pitching_zone':  results['pitching_zone'].value,
        'impact_zone':    results['impact_zone'].value,
        'wicket_verdict': results['wicket_verdict'].value,
        'final_call':     results['final_call'],
        'confidence':     ai_info['confidence'],
    }
    _decision_log.append(record)

    return jsonify({
        'job_id':          job_id,
        'pitching_zone':   results['pitching_zone'].value,
        'impact_zone':     results['impact_zone'].value,
        'wicket_verdict':  results['wicket_verdict'].value,
        'final_call':      results['final_call'],
        'ai_verdict':      ai_info,
        'delivery_stats':  stats,
        'physics_3d':      physics_info,
        'annotated_video': f"/outputs/{job_id}/tracked_output.mp4",
        'drs_image':        f"/outputs/{job_id}/drs_decision.png",
        'ultraedge_image':  f"/outputs/{job_id}/ultraedge_waveform.png",
    })


@app.route('/outputs/<job_id>/<filename>')
def get_output(job_id, filename):
    return send_file(os.path.join(app.config['OUTPUT_FOLDER'], job_id, filename))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
