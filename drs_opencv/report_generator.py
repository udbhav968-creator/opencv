"""
report_generator.py
-------------------
Generates a structured JSON report for each DRS analysis run.
The report captures everything: pipeline metadata, detection stats,
trajectory data, zone classifications, AI verdict, and delivery stats.
"""

import json
import os
import datetime
from pathlib import Path


def generate_report(results, ai_info, delivery_stats, job_id, output_dir, color_mode="red"):
    """
    Write a full JSON report for a DRS run.

    Args:
        results:        Output dict from run_pipeline()
        ai_info:        Output dict from generate_verdict_explanation()
        delivery_stats: Output dict from DeliveryStatsAnalyzer.analyze()
        job_id:         Unique job identifier string
        output_dir:     Directory to write the report into
        color_mode:     'red' or 'white'

    Returns:
        Path to the written JSON report file.
    """
    report = {
        "report_version": "1.0",
        "generated_at":   datetime.datetime.utcnow().isoformat() + "Z",
        "job_id":         job_id,
        "pipeline": {
            "color_mode":       color_mode,
            "success":          results.get("success", False),
            "tracking_video":   _rel(results.get("tracking_video"), output_dir),
            "decision_image":   _rel(results.get("decision_image"),  output_dir),
        },
        "drs_decision": {
            "pitching_zone":    _zone_val(results.get("pitching_zone")),
            "impact_zone":      _zone_val(results.get("impact_zone")),
            "wicket_verdict":   _zone_val(results.get("wicket_verdict")),
            "final_call":       results.get("final_call", "UNKNOWN"),
        },
        "ai_verdict": {
            "summary":          ai_info.get("summary",    ""),
            "reasoning":        ai_info.get("reasoning",  ""),
            "confidence_pct":   ai_info.get("confidence", 0),
            "tips":             ai_info.get("tips",       []),
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
        return None
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
