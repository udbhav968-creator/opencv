"""
api/index.py — Vercel Serverless Function Entrypoint
Guaranteed top-level `app` export for @vercel/python.
"""
from flask import Flask, request, render_template, send_file, jsonify
import os
import uuid
import sys
import datetime
import traceback
import subprocess
import threading

# Always expose top-level app object for Vercel Serverless Runtime
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
drs_dir = os.path.join(root_dir, 'drs_opencv')

if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
if drs_dir not in sys.path:
    sys.path.insert(0, drs_dir)

app = Flask(
    __name__,
    template_folder=os.path.join(root_dir, 'templates'),
    static_folder=os.path.join(root_dir, 'static')
)

app.config['UPLOAD_FOLDER'] = '/tmp/uploads'
app.config['OUTPUT_FOLDER'] = '/tmp/outputs'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

_decision_log = []

# Try importing pipeline modules
pipeline_available = False
import_error_message = ""

try:
    from main import run_pipeline
    from generate_test_video import generate_hitting, generate_missing, generate_umpires_call
    from ai_verdict import generate_verdict_explanation
    from stats_analyzer import DeliveryStatsAnalyzer
    pipeline_available = True
except Exception as err:
    import_error_message = str(err)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analytics')
def analytics_page():
    return render_template('analytics.html')

from flask import Flask, request, render_template, send_file, jsonify, Response
import os
import uuid
import sys
import datetime
import cv2

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


@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'service': 'Real DRS Hawk-Eye 3D API',
        'timestamp': datetime.datetime.utcnow().isoformat() + 'Z',
        'pipeline_available': pipeline_available,
        'import_error': import_error_message,
        'total_decisions': len(_decision_log),
    })

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



@app.route('/api/history')
def history():
    return jsonify({'decisions': _decision_log[-20:], 'total': len(_decision_log)})


@app.route('/api/stats')
def session_stats():
    counts = {'OUT': 0, 'NOT OUT': 0, "UMPIRE'S CALL": 0, "NO BALL DETECTED": 0}
    for d in _decision_log:
        counts[d.get('final_call', 'NOT OUT')] = counts.get(d.get('final_call', 'NOT OUT'), 0) + 1
    return jsonify({'session_totals': counts, 'decisions': len(_decision_log)})


@app.route('/process', methods=['POST'])
def process():
    if not pipeline_available:
        return jsonify({'error': f'Pipeline unavailable: {import_error_message}'}), 500

    try:
        color  = request.form.get('color', 'auto')
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

        results = run_pipeline(input_path, job_dir, color_mode=color)

        pz_str = results['pitching_zone'].value if hasattr(results['pitching_zone'], 'value') else str(results['pitching_zone'])
        iz_str = results['impact_zone'].value if hasattr(results['impact_zone'], 'value') else str(results['impact_zone'])
        wv_str = results['wicket_verdict'].value if hasattr(results['wicket_verdict'], 'value') else str(results['wicket_verdict'])
        fc_str = str(results['final_call'])

        if results.get('no_ball_detected'):
            ai_info = {
                'summary': 'No Cricket Ball Detected in Video Feed',
                'reasoning': 'The computer vision ensemble scanned the video feed but did not detect a valid moving cricket ball delivery. Please upload a clear cricket delivery video clip.',
                'confidence': 100,
                'tips': ['Ensure video contains a cricket pitch and active delivery.']
            }
        else:
            ai_info = generate_verdict_explanation(
                pitching_zone  = pz_str,
                impact_zone    = iz_str,
                wicket_verdict = wv_str,
                final_call     = fc_str,
            )

        analyzer   = DeliveryStatsAnalyzer(fps=25.0)
        valid_pts  = results.get('valid_points', [])
        stats      = analyzer.analyze(valid_pts) if valid_pts else {}

        pred_3d = results.get('prediction_3d')
        physics_info = {}
        if pred_3d and getattr(pred_3d, 'has_prediction', False):
            physics_info = {
                'pitch_3d_m':     [float(v) for v in pred_3d.pitch_3d] if pred_3d.pitch_3d is not None else None,
                'impact_3d_m':    [float(v) for v in pred_3d.impact_3d] if pred_3d.impact_3d is not None else None,
                'stump_x_m':      float(pred_3d.stump_x) if pred_3d.stump_x is not None else 0.0,
                'stump_z_m':      float(pred_3d.stump_z) if pred_3d.stump_z is not None else 0.0,
                'lateral_verdict': str(pred_3d.lateral_verdict),
                'height_verdict':  str(pred_3d.height_verdict),
                'final_3d_verdict': str(pred_3d.final_3d_verdict),
            }

        record = {
            'job_id':         job_id,
            'timestamp':      datetime.datetime.utcnow().isoformat() + 'Z',
            'color':          color,
            'pitching_zone':  pz_str,
            'impact_zone':    iz_str,
            'wicket_verdict': wv_str,
            'final_call':     fc_str,
            'confidence':     ai_info.get('confidence', 0),
        }
        _decision_log.append(record)

        return jsonify({
            'job_id':          job_id,
            'pitching_zone':   pz_str,
            'impact_zone':     iz_str,
            'wicket_verdict':  wv_str,
            'final_call':      fc_str,
            'ai_verdict':      ai_info,
            'delivery_stats':  stats,
            'physics_3d':      physics_info,
            'timestamp':  datetime.datetime.utcnow().isoformat() + 'Z',
        })

        return jsonify(response_payload)

    except Exception as exc:
        traceback.print_exc()
        return jsonify({'error': str(exc)}), 500


@app.route('/outputs/<job_id>/<filename>')
def serve_output(job_id, filename):
    job_dir = os.path.join(app.config['OUTPUT_FOLDER'], job_id)
    file_path = os.path.join(job_dir, filename)
    if os.path.exists(file_path):
        return send_file(file_path)
    return jsonify({'error': 'File not found'}), 404


if __name__ == '__main__':
    app.run(debug=True, port=5000)
