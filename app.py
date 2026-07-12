from flask import Flask, request, render_template, send_file, jsonify
import os
import uuid
import sys

# Add drs_opencv to path so imports work
sys.path.append(os.path.join(os.path.dirname(__file__), 'drs_opencv'))
from drs_opencv.main import run_pipeline
from drs_opencv.generate_test_video import generate_hitting, generate_missing, generate_umpires_call

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/tmp/uploads'
app.config['OUTPUT_FOLDER'] = '/tmp/outputs'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    color = request.form.get('color', 'red')
    action = request.form.get('action', 'upload')

    job_id = str(uuid.uuid4())
    job_dir = os.path.join(app.config['OUTPUT_FOLDER'], job_id)
    os.makedirs(job_dir, exist_ok=True)

    input_path = ""

    if action == 'upload':
        if 'video' not in request.files:
            return jsonify({'error': 'No video uploaded'}), 400
        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No video selected'}), 400
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}.mp4")
        file.save(input_path)
    else:
        # Generate synthetic video
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}.mp4")
        if action == 'synthetic_hitting':
            generate_hitting(input_path)
        elif action == 'synthetic_missing':
            generate_missing(input_path)
        elif action == 'synthetic_umpires_call':
            generate_umpires_call(input_path)
        else:
            return jsonify({'error': 'Invalid action'}), 400

    try:
        results = run_pipeline(input_path, job_dir, color_mode=color)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    if not results['success']:
        return jsonify({'error': 'Pipeline failed to process video'}), 500

    return jsonify({
        'job_id': job_id,
        'pitching_zone': results['pitching_zone'].value,
        'impact_zone': results['impact_zone'].value,
        'wicket_verdict': results['wicket_verdict'].value,
        'final_call': results['final_call']
    })

@app.route('/outputs/<job_id>/<filename>')
def get_output(job_id, filename):
    return send_file(os.path.join(app.config['OUTPUT_FOLDER'], job_id, filename))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
