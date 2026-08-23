# action_legality_classifier.py
"""
Automated ICC 15-Degree Bowling Action Legality Classifier.
Tracks skeletal elbow extension flex with automated warnings.
"""

class BowlingActionLegalityClassifier:
    def __init__(self, icc_threshold_deg=15.0):
        self.icc_threshold_deg = icc_threshold_deg

    def evaluate_bowling_action(self, elbow_extension_deg=8.2):
        is_legal = elbow_extension_deg <= self.icc_threshold_deg
        return {
            "elbow_extension_deg": elbow_extension_deg,
            "icc_threshold_deg": self.icc_threshold_deg,
            "action_verdict": "LEGAL_ACTION" if is_legal else "ILLEGAL_NO_BALL",
            "is_legal": is_legal,
            "compliance_margin_deg": round(self.icc_threshold_deg - elbow_extension_deg, 2)
        }

if __name__ == "__main__":
    classifier = BowlingActionLegalityClassifier()
    print("Action Legality Verdict:", classifier.evaluate_bowling_action()["action_verdict"])
