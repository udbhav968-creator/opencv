"""
app.py — OpenCV DRS Web Application
Flask backend serving the DRS pipeline with AI verdict, stats, and health endpoints.
"""

from flask import Flask, request, render_template, send_file, jsonify
import os
import uuid
import sys
import datetime

# Add drs_opencv to path so imports work
sys.path.append(os.path.join(os.path.dirname(__file__), 'drs_opencv'))
from drs_opencv.main import run_pipeline
from drs_opencv.generate_test_video import generate_hitting, generate_missing, generate_umpires_call
from drs_opencv.ai_verdict import generate_verdict_explanation
from drs_opencv.stats_analyzer import DeliveryStatsAnalyzer

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/tmp/uploads'
app.config['OUTPUT_FOLDER'] = '/tmp/outputs'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# In-memory decision log (resets on cold start — use a DB for persistence)
_decision_log = []


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/health')
def health():
    """Health-check endpoint — returns service status and uptime info."""
    return jsonify({
        'status': 'ok',
        'service': 'OpenCV DRS API',
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
    color  = request.form.get('color', 'red')
    action = request.form.get('action', 'upload')

    job_id  = str(uuid.uuid4())
    job_dir = os.path.join(app.config['OUTPUT_FOLDER'], job_id)
    os.makedirs(job_dir, exist_ok=True)

    # ── 1. Obtain input video ──
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

    # ── 2. Run DRS pipeline ──
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

    # ── 5. Log decision ──
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
    })


@app.route('/outputs/<job_id>/<filename>')
def get_output(job_id, filename):
    return send_file(os.path.join(app.config['OUTPUT_FOLDER'], job_id, filename))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
