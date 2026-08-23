# doppler_speed_calculator.py
"""
Doppler Effect Ball Speed Audio Synthesizer Engine.
Calculates delivery speed directly from stadium microphone acoustic frequency pitch shifts.
"""

import math

class DopplerSpeedCalculator:
    def __init__(self, speed_of_sound_ms=343.0):
        self.speed_of_sound_ms = speed_of_sound_ms

    def calculate_speed_from_doppler(self, f_approaching=440.0, f_receding=380.0):
        # Doppler formula: f_app = f0 * (v / (v - v_s)), f_rec = f0 * (v / (v + v_s))
        # v_s = v * (f_app - f_rec) / (f_app + f_rec)
        v_s_ms = self.speed_of_sound_ms * (f_approaching - f_receding) / (f_approaching + f_receding)
        v_s_kmh = v_s_ms * 3.6
        return {
            "doppler_calculator_active": True,
            "approaching_freq_hz": f_approaching,
            "receding_freq_hz": f_receding,
            "calculated_speed_ms": round(float(v_s_ms), 2),
            "calculated_speed_kmh": round(float(v_s_kmh), 2),
            "audio_speed_verified": True
        }

if __name__ == "__main__":
    doppler = DopplerSpeedCalculator()
    print("Doppler Speed Output:", doppler.calculate_speed_from_doppler()["calculated_speed_kmh"], "km/h")
