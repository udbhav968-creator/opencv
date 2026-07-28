"""
api/index.py — Vercel Serverless Function Entrypoint
Guaranteed top-level `app` export for @vercel/python.
"""
from flask import Flask, request, render_template, send_file, jsonify
import os
import uuid
import sys
import datetime

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


@app.route('/api/history')
def history():
    return jsonify({'decisions': _decision_log[-20:], 'total': len(_decision_log)})


@app.route('/api/stats')
def session_stats():
    counts = {'OUT': 0, 'NOT OUT': 0, "UMPIRE'S CALL": 0}
    for d in _decision_log:
        counts[d.get('final_call', 'NOT OUT')] = counts.get(d.get('final_call', 'NOT OUT'), 0) + 1
    return jsonify({'session_totals': counts, 'decisions': len(_decision_log)})


@app.route('/process', methods=['POST'])
def process():
    if not pipeline_available:
        return jsonify({'error': f'Pipeline unavailable: {import_error_message}'}), 500

    color  = request.form.get('color', 'red')
    action = request.form.get('action', 'upload')

    job_id  = str(uuid.uuid4())
    job_dir = os.path.join(app.config['OUTPUT_FOLDER'], job_id)
    os.makedirs(job_dir, exist_ok=True)

    if action == 'upload':
        if 'video' not in request.files:
            return jsonify({'error': 'No video uploaded'}), 400
        file = request.files['video']
        if not file.filename:
            return jsonify({'error': 'No video selected'}), 400
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}.mp4")
        file.save(input_path)
    else:
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}.mp4")
        generators = {
            'synthetic_hitting':      generate_hitting,
            'synthetic_missing':      generate_missing,
            'synthetic_umpires_call': generate_umpires_call,
        }
        gen_fn = generators.get(action)
        if gen_fn is None:
            return jsonify({'error': f'Unknown action: {action}'}), 400
        gen_fn(input_path)

    try:
        results = run_pipeline(input_path, job_dir, color_mode=color)
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500

    if not results['success']:
        return jsonify({'error': 'Pipeline failed — not enough ball detections.'}), 500

    ai_info = generate_verdict_explanation(
        pitching_zone  = results['pitching_zone'].value,
        impact_zone    = results['impact_zone'].value,
        wicket_verdict = results['wicket_verdict'].value,
        final_call     = results['final_call'],
    )

    analyzer   = DeliveryStatsAnalyzer(fps=25.0)
    valid_pts  = results.get('valid_points', [])
    stats      = analyzer.analyze(valid_pts) if valid_pts else {}

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
        'job_id':         job_id,
        'pitching_zone':  results['pitching_zone'].value,
        'impact_zone':    results['impact_zone'].value,
        'wicket_verdict': results['wicket_verdict'].value,
        'final_call':     results['final_call'],
        'ai_verdict':     ai_info,
        'delivery_stats': stats,
        'physics_3d':     physics_info,
    })


@app.route('/outputs/<job_id>/<filename>')
def get_output(job_id, filename):
    return send_file(os.path.join(app.config['OUTPUT_FOLDER'], job_id, filename))
