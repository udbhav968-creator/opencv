"""
ultraedge.py
------------
UltraEdge / Snickometer Audio-Visual Waveform Simulator.

Simulates synchronous audio frequency spectrum & acoustic spike detection:
  - Generates synchronized audio waveform data across video frames
  - Detects acoustic spikes: Bat Edge (sharp high frequency spike) vs Pad Impact (dull low frequency pulse)
  - Renders broadcast-style UltraEdge waveform panel graphic
"""

import numpy as np
import cv2


class UltraEdgeSimulator:
    """
    Simulates Snickometer / UltraEdge audio analysis & visual waveform generation.
    """

    def __init__(self, n_frames=30, sample_rate=1000):
        self.n_frames = n_frames
        self.sample_rate = sample_rate
        self.samples_per_frame = sample_rate // 25  # 40 audio samples per video frame

    def generate_waveform(self, impact_frame=18, edge_event=False):
        """
        Generates synthetic audio frequency waveform across frames.
        
        Args:
            impact_frame: Frame index where impact occurs
            edge_event: True if ball hit bat/glove, False if pure pad impact / clean
            
        Returns:
            dict containing:
              - 'waveform': 1D numpy array of audio amplitudes [-1.0, 1.0]
              - 'edge_detected': bool
              - 'event_type': 'BAT EDGE' / 'PAD IMPACT' / 'CLEAN'
              - 'max_frequency_hz': peak frequency detected
        """
        total_samples = self.n_frames * self.samples_per_frame
        time_axis = np.linspace(0, self.n_frames / 25.0, total_samples)

        # Baseline noise (ambient crowd / field sound)
        waveform = 0.05 * np.random.randn(total_samples)

        # Impact sample index
        impact_idx = min(impact_frame * self.samples_per_frame, total_samples - 1)

        if edge_event:
            # High-frequency sharp acoustic spike (Wood contact: 2500Hz - 4000Hz)
            end_idx = min(impact_idx + self.samples_per_frame * 2, total_samples)
            t_event = time_axis[impact_idx : end_idx]
            if len(t_event) > 0:
                t_rel = t_event - t_event[0]
                spike = 0.85 * np.exp(-t_rel * 40.0) * np.sin(2 * np.pi * 3200 * t_rel)
                waveform[impact_idx : impact_idx + len(spike)] += spike
            event_type = "BAT EDGE"
            peak_freq = 3200
        else:
            # Low-frequency dull pulse (Pad contact: 300Hz - 600Hz)
            end_idx = min(impact_idx + self.samples_per_frame * 3, total_samples)
            t_event = time_axis[impact_idx : end_idx]
            if len(t_event) > 0:
                t_rel = t_event - t_event[0]
                pulse = 0.4 * np.exp(-t_rel * 20.0) * np.sin(2 * np.pi * 450 * t_rel)
                waveform[impact_idx : impact_idx + len(pulse)] += pulse
            event_type = "PAD IMPACT"
            peak_freq = 450

        # Clip amplitude to range [-1.0, 1.0]
        waveform = np.clip(waveform, -1.0, 1.0)

        return {
            "waveform": waveform,
            "edge_detected": edge_event,
            "event_type": event_type,
            "max_frequency_hz": peak_freq,
            "impact_frame": impact_frame
        }

    def render_ultraedge_panel(self, waveform_data, current_frame_idx, width=720, height=180):
        """
        Renders a broadcast-style UltraEdge waveform graphic panel.
        """
        panel = np.full((height, width, 3), (15, 23, 42), dtype=np.uint8)

        # Draw Gridlines
        for y in range(30, height, 30):
            cv2.line(panel, (0, y), (width, y), (30, 41, 59), 1)
        center_y = height // 2
        cv2.line(panel, (0, center_y), (width, center_y), (51, 65, 85), 1)

        waveform = waveform_data["waveform"]
        n_samples = len(waveform)

        # Plot Waveform Green Oscilloscope Line
        pts = []
        for i, val in enumerate(waveform):
            px = int((i / n_samples) * width)
            py = int(center_y - val * (height * 0.4))
            pts.append((px, py))

        for i in range(1, len(pts)):
            cv2.line(panel, pts[i - 1], pts[i], (34, 197, 94), 2, cv2.LINE_AA)

        # Synchronized Frame Timeline Marker (Red vertical cursor)
        curr_px = int((current_frame_idx / max(1, self.n_frames)) * width)
        cv2.line(panel, (curr_px, 0), (curr_px, height), (239, 68, 68), 2)

        # Overlay Event Info Banner
        event_type = waveform_data["event_type"]
        color = (239, 68, 68) if event_type == "BAT EDGE" else (56, 189, 248)
        
        cv2.putText(panel, f"ULTRAEDGE - {event_type}", (15, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        cv2.putText(panel, f"Freq: {waveform_data['max_frequency_hz']} Hz", (width - 150, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (148, 163, 184), 1, cv2.LINE_AA)

        return panel
