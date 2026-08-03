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

        response_payload = {
            'job_id':          job_id,
            'pitching_zone':   pz_str,
            'impact_zone':     iz_str,
            'wicket_verdict':  wv_str,
            'final_call':      fc_str,
            'ai_verdict':      ai_info,
            'delivery_stats':  stats,
            'physics_3d':      physics_info,
            'detected_color':  results.get('detected_color', 'red'),
            'no_ball_detected': results.get('no_ball_detected', False),
        }

        _decision_log.append({
            'job_id':     job_id,
            'final_call': fc_str,
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
