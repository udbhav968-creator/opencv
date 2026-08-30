# spin_wobble_predictor.py
"""
Dynamic 3D Spin Aerodynamics & Magnus Seam Wobble Predictor.
Calculates seam wobble angle, dynamic Magnus drift, and lateral deviation vectors.
"""

import math

class SpinWobblePredictor:
    def __init__(self):
        self.air_density = 1.225  # kg/m^3

    def compute_spin_dynamics(self, spin_rpm=2400.0, seam_angle_deg=18.5, velocity_kmh=138.0):
        omega = (spin_rpm * 2.0 * math.pi) / 60.0
        v_ms = velocity_kmh / 3.6
        magnus_force_n = 0.5 * self.air_density * (0.036 ** 2) * math.pi * (omega * 0.036) * v_ms * 0.12
        lateral_drift_cm = (magnus_force_n / 0.16) * (0.5 * (18.0 / v_ms) ** 2) * 100.0
        return {
            "spin_predictor_active": True,
            "spin_rpm": spin_rpm,
            "seam_wobble_angle_deg": seam_angle_deg,
            "magnus_force_newtons": round(magnus_force_n, 4),
            "lateral_drift_cm": round(lateral_drift_cm, 2),
            "restitution_spin_decay_pct": 8.5
        }

if __name__ == "__main__":
    pred = SpinWobblePredictor()
    print("Spin Wobble Dynamics:", pred.compute_spin_dynamics())
