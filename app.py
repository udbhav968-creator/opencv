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
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        cv2.putText(frame, "LIVE ICC BROADCAST FEED -- HAWK-EYE 3D ACTIVE", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (56, 189, 248), 2, cv2.LINE_AA)
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


@app.route('/health')
def health():
    """Health-check endpoint — returns service status and uptime info."""
    return jsonify({
        'status': 'ok',
        'service': 'Real DRS Hawk-Eye 3D API',
        'timestamp': datetime.datetime.utcnow().isoformat() + 'Z',
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
        'timestamp':      datetime.datetime.utcnow().isoformat() + 'Z',
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
