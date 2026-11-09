# CoffeeGuard AI

An explainable computer-vision system for detecting coffee leaf diseases. CoffeeGuard combines EfficientNetV2, FastAPI, Streamlit, ONNX Runtime, robustness analysis, and uncertainty-aware decisions for practical model-assisted screening.

## Capabilities

- Classifies healthy leaves, Cercospora, Leaf Rust, and Phoma
- Reports confidence and uncertainty
- Provides explainability heat maps and out-of-distribution checks
- Includes evaluation, robustness testing, and calibration
- Exposes FastAPI inference and Streamlit interfaces
- Supports Docker deployment

## Technology

| Area | Stack |
|---|---|
| Model | PyTorch, EfficientNetV2, ONNX Runtime |
| API | FastAPI, Uvicorn, Pydantic |
| Interface | Streamlit, Altair |
| Engineering | Python, uv, pytest, Ruff, Docker |

## Quick start

```bash
git clone https://github.com/Ab-xo/coffee-guard-ai.git
cd coffee-guard-ai
docker compose up --build
```

- Web interface: `http://localhost:8501`
- API documentation: `http://localhost:8000/docs`

## Limitations

This is a decision-support prototype, not an agricultural diagnosis. Performance varies with dataset source, image quality, disease stage, and field conditions.

## License

MIT
