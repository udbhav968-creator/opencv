# Contributing to OpenCV DRS

Thank you for your interest in contributing! 🏏

## Getting Started

```bash
git clone https://github.com/udbhav968-creator/opencv.git
cd opencv
pip install -r requirements.txt
```

## Development Workflow

1. **Fork** the repo and create your branch from `main`
2. Make your changes with clear, focused commits
3. Run the synthetic pipeline to verify nothing is broken:
   ```bash
   cd drs_opencv
   python generate_test_video.py --out test.mp4 --outcome hitting
   python main.py --input test.mp4 --output_dir output
   ```
4. Open a **Pull Request** with a clear description

## Commit Convention

Use conventional commit prefixes:
- `feat:` — new feature
- `fix:` — bug fix
- `docs:` — documentation only
- `refactor:` — code change, no feature/fix
- `ci:` — CI/deployment changes
- `chore:` — maintenance

## Areas to Contribute

| Area | File | Description |
|------|------|-------------|
| Detection | `drs_opencv/ball_detector.py` | Improve HSV thresholds or add ML model |
| Tracking | `drs_opencv/tracker.py` | Tune Kalman parameters |
| Trajectory | `drs_opencv/trajectory_predictor.py` | Add parabolic/height modelling |
| AI Verdict | `drs_opencv/ai_verdict.py` | Improve natural language generation |
| Stats | `drs_opencv/stats_analyzer.py` | Add spin rate, release point analysis |
| UI | `templates/index.html` | Improve the web interface |
| API | `app.py` | Add new endpoints |

## Code Style

- Follow PEP 8
- Add docstrings to all public functions/classes
- Keep functions focused and single-purpose

## License

By contributing you agree that your contributions will be licensed under the MIT License.
