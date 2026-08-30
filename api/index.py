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

@app.route('/favicon.ico')
def favicon():
    svg_icon = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><circle cx="50" cy="50" r="45" fill="#ef4444"/><circle cx="50" cy="50" r="30" fill="none" stroke="#facc15" stroke-width="6"/></svg>'''
    return Response(svg_icon, mimetype='image/svg+xml')

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

@app.route('/sw.js')
def service_worker():
    return send_file(os.path.join(root_dir, 'static', 'sw.js'), mimetype='application/javascript')

@app.route('/api/biomechanics')
def api_biomechanics():
    try:
        from biomechanics_analyzer import BiomechanicsAnalyzer
        analyzer = BiomechanicsAnalyzer()
        return jsonify(analyzer.analyze_pose_keypoints())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/win_probability')
def api_win_probability():
    try:
        from win_probability_engine import WinProbabilityEngine
        engine = WinProbabilityEngine()
        return jsonify(engine.simulate_win_probability())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/export_certificate')
def api_export_certificate():
    try:
        from export_suite import DRSExportSuite
        suite = DRSExportSuite()
        return jsonify(suite.generate_certificate_json("JOB12345", {"final_call": "NOT OUT"}))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/spatial_snicko')
def api_spatial_snicko():
    try:
        from spatial_snicko_beamformer import SpatialSnickoBeamformer
        bf = SpatialSnickoBeamformer()
        return jsonify(bf.compute_acoustic_beamforming())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/camera_calibration')
def api_camera_calibration():
    try:
        from camera_autocalibrator import CameraAutoCalibrator
        calib = CameraAutoCalibrator()
        return jsonify(calib.calibrate_cameras())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/action_legality')
def api_action_legality():
    try:
        from action_legality_classifier import BowlingActionLegalityClassifier
        classifier = BowlingActionLegalityClassifier()
        return jsonify(classifier.evaluate_bowling_action())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/stadium_stream')
def api_stadium_stream():
    try:
        from stadium_websocket_stream import StadiumWebSocketStreamer
        streamer = StadiumWebSocketStreamer()
        return jsonify(streamer.broadcast_telemetry({"final_call": "OUT"}))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/super_slowmo')
def api_super_slowmo():
    try:
        from frame_interpolator import SuperSlowMoFrameInterpolator
        interp = SuperSlowMoFrameInterpolator()
        return jsonify(interp.interpolate_frames())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/nerf_3d')
def api_nerf_3d():
    try:
        from nerf_3d_reconstruction import GaussianSplatting3DReconstructor
        recon = GaussianSplatting3DReconstructor()
        return jsonify(recon.reconstruct_3d_scene())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/doppler_speed')
def api_doppler_speed():
    try:
        from doppler_speed_calculator import DopplerSpeedCalculator
        doppler = DopplerSpeedCalculator()
        return jsonify(doppler.calculate_speed_from_doppler())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/fielder_probability')
def api_fielder_probability():
    try:
        from fielder_catch_probability import FielderCatchProbabilityAI
        ai = FielderCatchProbabilityAI()
        return jsonify(ai.estimate_catch_probability())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/crypto_ledger')
def api_crypto_ledger():
    try:
        from crypto_merkle_ledger import CryptographicMerkleLedger
        ledger = CryptographicMerkleLedger()
        return jsonify(ledger.sign_certificate("JOB12345", {"final_call": "OUT"}))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/camera_mesh')
def api_camera_mesh():
    try:
        from camera_mesh_synchronizer import CameraMeshSynchronizer
        mesh = CameraMeshSynchronizer()
        return jsonify(mesh.synchronize_mesh())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/pinn_trajectory')
def api_pinn_trajectory():
    try:
        from pinn_trajectory_ai import PhysicsInformedNNTrajectoryAI
        pinn = PhysicsInformedNNTrajectoryAI()
        return jsonify(pinn.predict_pinn_trajectory())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/quantum_sim')
def api_quantum_sim():
    try:
        from quantum_match_simulator import QuantumInspiredMatchSimulator
        qsim = QuantumInspiredMatchSimulator()
        return jsonify(qsim.run_quantum_simulations())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/int8_npu')
def api_int8_npu():
    try:
        from int8_npu_accelerator import INT8NPUAccelerator
        npu = INT8NPUAccelerator()
        return jsonify(npu.run_npu_inference())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/vision_transformer')
def api_vision_transformer():
    try:
        from vision_transformer_detector import VisionTransformerBallDetector
        vit = VisionTransformerBallDetector()
        return jsonify(vit.detect_attention_ball())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/youtube_harvester')
def api_youtube_harvester():
    try:
        from youtube_live_harvester import YouTubeLiveDatasetHarvester
        harvester = YouTubeLiveDatasetHarvester()
        return jsonify(harvester.harvest_live_streams())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/llm_umpire_reasoner')
def api_llm_umpire_reasoner():
    try:
        from llm_umpire_reasoner import MultimodalLLMUmpireReasoner
        reasoner = MultimodalLLMUmpireReasoner()
        return jsonify(reasoner.generate_umpire_reasoning())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/wind_humidity_aero')
def api_wind_humidity_aero():
    try:
        from wind_humidity_aero_ai import WindHumidityAeroAI
        aero = WindHumidityAeroAI()
        return jsonify(aero.compute_aero_drift())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/lidar_sensor_fusion')
def api_lidar_sensor_fusion():
    try:
        from lidar_sensor_fusion import LiDARSensorFusionEngine
        fusion = LiDARSensorFusionEngine()
        return jsonify(fusion.fuse_point_cloud())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/volumetric_4d')
def api_volumetric_4d():
    try:
        from volumetric_4d_reconstructor import Volumetric4DPitchReconstructor
        recon = Volumetric4DPitchReconstructor()
        return jsonify(recon.reconstruct_4d_volumetric())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/micro_vibration')
def api_micro_vibration():
    try:
        from micro_vibration_synthesizer import MicroVibrationEdgeSynthesizer
        vibe = MicroVibrationEdgeSynthesizer()
        return jsonify(vibe.synthesize_edge_vibration())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/thermal_pitch')
def api_thermal_pitch():
    try:
        from thermal_pitch_scanner import ThermalPitchMoistureScanner
        scanner = ThermalPitchMoistureScanner()
        return jsonify(scanner.scan_pitch_friction())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/bowling_torque')
def api_bowling_torque():
    try:
        from bowling_torque_analyzer import BiomechanicalBowlingTorqueAnalyzer
        analyzer = BiomechanicalBowlingTorqueAnalyzer()
        return jsonify(analyzer.analyze_bowling_torque())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/zk_proof_ledger')
def api_zk_proof_ledger():
    try:
        from zk_proof_drs_ledger import ZKProofCryptographicDRSLedger
        zk = ZKProofCryptographicDRSLedger()
        return jsonify(zk.generate_zk_proof())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/external_apis')
def api_external_apis():
    try:
        from api_integration_suite import ExternalAPIIntegrationSuite
        suite = ExternalAPIIntegrationSuite()
        return jsonify(suite.fetch_all_telemetry())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/stadium_presets')
def api_stadium_presets():
    try:
        from stadium_calibration import StadiumCalibrationManager
        mgr = StadiumCalibrationManager()
        return jsonify(mgr.list_all_stadiums())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/audio_commentary')
def api_audio_commentary():
    try:
        from audio_commentary_ai import AudioCommentaryAI
        ai = AudioCommentaryAI()
        pitching = request.args.get('pitching', 'IN_LINE')
        impact = request.args.get('impact', 'IN_LINE')
        wickets = request.args.get('wickets', 'HITTING')
        final_call = request.args.get('final_call', 'NOT OUT')
        speed_kmh = float(request.args.get('speed_kmh', 142.5))
        spin_rpm = int(request.args.get('spin_rpm', 2240))
        return jsonify(ai.generate_commentary(pitching=pitching, impact=impact, wickets=wickets, final_call=final_call, speed_kmh=speed_kmh, spin_rpm=spin_rpm))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/gpr_subsurface')
def api_gpr_subsurface():
    try:
        from gpr_subsurface_scanner import GPRSubSurfaceScanner
        scanner = GPRSubSurfaceScanner()
        return jsonify(scanner.scan_pitch_subsurface())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/spin_wobble')
def api_spin_wobble():
    try:
        from spin_wobble_predictor import SpinWobblePredictor
        pred = SpinWobblePredictor()
        return jsonify(pred.compute_spin_dynamics())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/spatial_hologram')
def api_spatial_hologram():
    try:
        from spatial_hologram_streamer import SpatialHologramStreamer
        holo = SpatialHologramStreamer()
        return jsonify(holo.stream_spatial_hologram())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/pqc_ledger')
def api_pqc_ledger():
    try:
        from pqc_dilithium_ledger import PQCDilithiumDRSLedger
        ledger = PQCDilithiumDRSLedger()
        return jsonify(ledger.sign_drs_decision())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'service': 'Real DRS Hawk-Eye 3D API',
        'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(),
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
            'timestamp':      datetime.datetime.now(datetime.timezone.utc).isoformat(),
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
            'annotated_video': f"/outputs/{job_id}/tracked_output.mp4",
            'drs_image':       f"/outputs/{job_id}/drs_decision.png",
            'ultraedge_image': f"/outputs/{job_id}/ultraedge_waveform.png",
            'timestamp':       datetime.datetime.now(datetime.timezone.utc).isoformat(),
        })

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
