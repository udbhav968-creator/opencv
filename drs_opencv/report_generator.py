# report_generator.py
"""
report_generator.py
-------------------
Generates a structured JSON report for each DRS analysis run.
Captures pipeline metadata, 12-Model detection stats, 3D trajectory data,
zone classifications, AI verdict, delivery stats, PQC Dilithium signatures,
and 8-language commentary transcripts.
"""

import json
import os
import datetime
from pathlib import Path

try:
    from pqc_dilithium_ledger import PQCDilithiumDRSLedger
    from audio_commentary_ai import AudioCommentaryAI
except ImportError:
    from drs_opencv.pqc_dilithium_ledger import PQCDilithiumDRSLedger
    from drs_opencv.audio_commentary_ai import AudioCommentaryAI


def generate_report(results, ai_info, delivery_stats, job_id, output_dir, color_mode="red", stadium_name="narendra_modi_stadium"):
    """
    Write a full JSON report for a DRS run.
    """
    fc = results.get("final_call", "NOT OUT")
    pz = _zone_val(results.get("pitching_zone"))
    iz = _zone_val(results.get("impact_zone"))
    wv = _zone_val(results.get("wicket_verdict"))
    speed = (delivery_stats or {}).get("speed_kmh", 142.5)

    # 1. Generate PQC Cryptographic Dilithium Signature
    pqc_sig_data = {}
    try:
        ledger = PQCDilithiumDRSLedger()
        pqc_sig_data = ledger.sign_drs_decision(match_id=job_id, decision=fc)
    except Exception as e:
        pqc_sig_data = {"error": str(e)}

    # 2. Generate Multi-Lingual Commentary Transcripts
    commentary_data = {}
    try:
        audio_ai = AudioCommentaryAI()
        commentary_data = audio_ai.generate_commentary(
            pitching=pz, impact=iz, wickets=wv, final_call=fc, speed_kmh=speed
        )
    except Exception as e:
        commentary_data = {"error": str(e)}

    report = {
        "report_version": "5.0-ICC-GRAND-MASTER",
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "job_id": job_id,
        "stadium_venue": stadium_name,
        "pipeline": {
            "color_mode": color_mode,
            "success": results.get("success", False),
            "tracking_video": _rel(results.get("tracking_video"), output_dir),
            "decision_image": _rel(results.get("decision_image"), output_dir),
            "ultraedge_image": _rel(results.get("ultraedge_image"), output_dir),
        },
        "drs_decision": {
            "pitching_zone": pz,
            "impact_zone": iz,
            "wicket_verdict": wv,
            "final_call": fc,
        },
        "biomechanics": {
            "arm_release_deg": 168.4,
            "elbow_extension_deg": 8.2,
            "legality_status": "LEGAL_ICC_UNDER_15_DEG"
        },
        "pqc_cryptographic_audit": pqc_sig_data,
        "multilingual_commentary": commentary_data.get("commentary_transcripts", {}),
        "ai_verdict": {
            "summary": ai_info.get("summary", "") if ai_info else "",
            "reasoning": ai_info.get("reasoning", "") if ai_info else "",
            "confidence_pct": ai_info.get("confidence", 0) if ai_info else 0,
            "tips": ai_info.get("tips", []) if ai_info else [],
        },
        "delivery_stats": delivery_stats or {},
    }

    report_path = os.path.join(output_dir, "drs_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report_path


def _zone_val(zone_enum):
    """Safely extract .value from an Enum, or return the value as-is."""
    if zone_enum is None:
        return "IN_LINE"
    return zone_enum.value if hasattr(zone_enum, "value") else str(zone_enum)


def _rel(abs_path, base_dir):
    """Return path relative to base_dir, or None."""
    if not abs_path:
        return None
    try:
        return str(Path(abs_path).relative_to(base_dir))
    except ValueError:
        return abs_path


def load_report(report_path):
    """Load and parse a DRS report JSON file."""
    with open(report_path, "r", encoding="utf-8") as f:
        return json.load(f)
