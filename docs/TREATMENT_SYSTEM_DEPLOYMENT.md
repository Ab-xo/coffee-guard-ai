# Treatment Recommendation System - Deployment Guide

**Version**: 1.0  
**Date**: September 2026

---

## Overview

The CoffeeGuard treatment recommendation system provides actionable advice to farmers based on disease predictions. It integrates:

- **Treatment knowledge base** (`ml/knowledge/treatments.json`)
- **Recommendation engine** (`src/coffeeguard/treatment/`)
- **API integration** (automatic recommendations in `/predict` and `/analyze` endpoints)
- **Field feedback system** (`src/coffeeguard/field/`) for tracking accuracy

---

## Quick Start

### 1. Check Treatment Database

The treatment knowledge base is at `ml/knowledge/treatments.json`:

```bash
# View treatments
cat ml/knowledge/treatments.json | jq '.'

# Validate JSON
python -c "import json; json.load(open('ml/knowledge/treatments.json'))"
```

### 2. Test Recommendation Engine

```python
from coffeeguard.treatment import TreatmentRecommender

# Initialize
recommender = TreatmentRecommender()

# Get recommendation for Leaf Rust (class_id=2)
recommendation = recommender.get_recommendation(
    class_id=2,          # Leaf Rust
    confidence=0.92,     # High confidence
    severity="mild"      # Mild infection
)

print(recommendation)

# Format for display
markdown_output = recommender.format_for_display(
    recommendation,
    format_type="markdown"
)
print(markdown_output)
```

### 3. Test API with Treatments

```bash
# Start API
uv run uvicorn app.main:app --app-dir apps/api --reload

# Test prediction (includes treatments)
curl -X POST "http://localhost:8000/predict" \
  -F "file=@apps/web/samples/leaf_rust.jpg" \
  | jq '.treatment'

# Expected output:
# {
#   "disease": "Coffee Leaf Rust",
#   "urgency": "HIGH - Act within 5-7 days",
#   "treatments": [...],
#   "prevention": [...]
# }
```

---

## Architecture

```
┌──────────────────────────────────────────────────┐
│          User uploads coffee leaf image          │
└────────────────────┬─────────────────────────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │  CoffeeGuard Pipeline │
         │  (Disease Detection)  │
         └───────────┬───────────┘
                     │
           ┌─────────┴─────────┐
           │                   │
           ▼                   ▼
   Disease Detected?      Healthy/Rejected
           │                   │
           │ YES               │ → Basic advice
           ▼                   │
┌──────────────────────┐       │
│ TreatmentRecommender │       │
│  - Load knowledge DB │       │
│  - Estimate severity │       │
│  - Select treatments │       │
└──────────┬───────────┘       │
           │                   │
           ▼                   ▼
    ┌──────────────────────────────┐
    │   Return to User:            │
    │   - Disease name             │
    │   - Treatment options        │
    │   - Prevention advice        │
    │   - Urgency level            │
    │   - Safety notes             │
    └──────────────────────────────┘
```

---

## Configuration

### Severity Estimation

Currently estimates severity from confidence:
- **High confidence (≥0.95)**: moderate severity
- **Medium confidence (0.80-0.95)**: mild severity
- **Low confidence (<0.80)**: mild severity

**Future enhancement**: Use image analysis to estimate actual lesion coverage.

### Treatment Levels

Each disease has treatments for:
- **Mild**: Few spots, early intervention
- **Moderate**: Multiple spots, intensive treatment
- **Severe**: Heavy infection, emergency response

### Language Support

Current: English  
Planned: Amharic translation

To add Amharic:
1. Create `ml/knowledge/treatments_am.json`
2. Update `TreatmentRecommender` to accept language parameter
3. API accepts `Accept-Language: am` header

---

## Field Feedback System

### Setting Up Field Data Collection

```python
from coffeeguard.field import FieldDataCollector

# Initialize collector
collector = FieldDataCollector()
# Data stored in: data/field/

# When making prediction:
pred_id = collector.record_prediction(
    image_path="farmer_photo.jpg",
    predicted_class=2,
    predicted_label="Leaf Rust",
    confidence=0.89,
    metadata={
        "location": "Sidama Region, Farm A",
        "farmer_id": "F001",
        "gps_coords": {"lat": 6.5, "lon": 38.5}
    }
)

# Later, collect feedback:
collector.add_feedback(
    entry_id=pred_id,
    was_correct=True,
    severity="moderate",
    treatment_applied="Copper fungicide",
    notes="Farmer followed organic treatment, will check in 2 weeks"
)

# Generate statistics
stats = collector.get_statistics()
print(f"Field accuracy: {stats['accuracy']:.1%}")

# Generate report
report = collector.generate_report()
print(report)

# Export for retraining
collector.export_for_retraining()
```

### Tracking Treatment Outcomes

```python
# Initial prediction
pred_id = collector.record_prediction(...)

# Follow-up visit (2-4 weeks later)
collector.add_feedback(
    entry_id=pred_id,
    treatment_outcome="improved",  # improved/same/worse
    notes="Disease spread stopped, new leaves healthy"
)
```

---

## Updating Treatment Database

### Process

1. **Consult agronomists** - Work with local experts
2. **Review literature** - Latest coffee disease management research
3. **Update JSON** - Edit `ml/knowledge/treatments.json`
4. **Validate** - Ensure valid JSON syntax
5. **Test** - Run through recommender engine
6. **Deploy** - Restart API to load new treatments

### Example: Adding New Treatment

```json
{
  "diseases": {
    "leaf_rust": {
      "treatments": {
        "mild": {
          "organic": [
            "Apply copper hydroxide every 21 days (2.5-3 g/L)",
            "NEW: Biofungicide X-Protocol (experimental)",
            "Remove and burn infected leaves"
          ]
        }
      }
    }
  }
}
```

### Validation Script

```python
import json
from pathlib import Path

def validate_treatments():
    """Validate treatment database structure."""
    path = Path("ml/knowledge/treatments.json")
    
    with open(path) as f:
        data = json.load(f)
    
    # Check required fields
    assert "version" in data
    assert "diseases" in data
    
    # Check each disease
    for disease_key, disease_data in data["diseases"].items():
        assert "name" in disease_data
        print(f"✓ {disease_data['name']}")
        
        if "treatments" in disease_data:
            for severity in disease_data["treatments"]:
                print(f"  ✓ {severity} severity defined")
    
    print("\n✅ Treatment database valid")

validate_treatments()
```

---

## API Response Examples

### Healthy Leaf

```json
{
  "status": "accepted",
  "label": "Healthy",
  "confidence": 0.96,
  "treatment": {
    "disease": "Healthy Leaf",
    "advice": "Your coffee plant looks healthy! Continue good practices.",
    "prevention": [
      "Monitor plants weekly for early disease detection",
      "Maintain 2-2.5m spacing between plants",
      "Apply organic mulch and compost annually"
    ]
  }
}
```

### Leaf Rust - Mild

```json
{
  "status": "accepted",
  "label": "Leaf Rust",
  "confidence": 0.89,
  "treatment": {
    "disease": "Coffee Leaf Rust",
    "description": "Most damaging coffee disease - orange powdery pustules",
    "severity": "mild",
    "urgency": "HIGH - Act within 5-7 days to prevent rapid spread",
    "organic_treatments": [
      "Copper hydroxide spray every 21 days (2.5-3 g/L)",
      "Remove and burn infected leaves weekly",
      "Apply potassium sulfate (50g per plant)"
    ],
    "conventional_treatments": [
      "Systemic fungicide (Cyproconazole) immediately",
      "Repeat after 21 days",
      "Most effective treatment option"
    ],
    "prevention": [
      "Preventive fungicide program (2,000-4,000 Birr/ha/year)",
      "Weekly scouting during rainy season",
      "Spray before rainy season starts"
    ],
    "safety_note": "Wear gloves, mask, and protective clothing"
  }
}
```

### Cercospora - Severe

```json
{
  "status": "accepted",
  "label": "Cercospora",
  "confidence": 0.95,
  "treatment": {
    "disease": "Brown Eye Spot (Cercospora)",
    "severity": "severe",
    "urgency": "critical",
    "urgent_message": "Contact agricultural extension officer immediately",
    "treatments": [
      "Apply fungicide within 24-48 hours",
      "Remove 50-60% of infected plant material",
      "Intensive spray program: every 7 days for 8 weeks"
    ],
    "cost_warning": "Treatment cost high but necessary to prevent complete crop loss"
  }
}
```

---

## Monitoring & Analytics

### System Health

```bash
# Check if treatment system is working
curl http://localhost:8000/predict \
  -F "file=@test_diseased_leaf.jpg" \
  | jq '.treatment != null'
# Should return: true (if disease detected)
```

### Field Performance Metrics

```python
from coffeeguard.field import AccuracyTracker

tracker = AccuracyTracker()

# Record daily/weekly metrics
tracker.record_batch_metrics(
    date="2026-09-25",
    total_predictions=50,
    correct_predictions=42,
    per_class_accuracy={
        "Healthy": 0.95,
        "Cercospora": 0.70,
        "Leaf Rust": 0.90,
        "Phoma": 0.85
    }
)

# Analyze trends
trends = tracker.get_trends(days=30)
print(f"30-day mean accuracy: {trends['mean_accuracy']:.1%}")
print(f"Trend: {trends['trend']}")
```

---

## Troubleshooting

### Treatment not appearing in API response

```python
# Check 1: Is treatment DB loaded?
from coffeeguard.treatment import TreatmentRecommender
try:
    rec = TreatmentRecommender()
    print("✓ Treatment DB loaded")
except FileNotFoundError as e:
    print(f"✗ Treatment DB not found: {e}")

# Check 2: Is disease being detected?
# Treatments only appear for accepted disease predictions (not healthy/rejected)
```

### Invalid JSON in treatment database

```bash
# Validate JSON syntax
python -m json.tool ml/knowledge/treatments.json > /dev/null
echo $?  # Should output: 0
```

### Field data not saving

```python
from pathlib import Path

# Check data directory exists and is writable
data_dir = Path("data/field")
print(f"Exists: {data_dir.exists()}")
print(f"Writable: {os.access(data_dir, os.W_OK)}")

# Check file permissions
feedback_file = data_dir / "feedback.jsonl"
if feedback_file.exists():
    print(f"File size: {feedback_file.stat().st_size} bytes")
```

---

## Security Considerations

### Treatment Database Integrity

- **Read-only in production**: Prevent accidental modifications
- **Version control**: Track all changes in git
- **Expert review**: All treatment updates reviewed by agronomists
- **Backup**: Daily backups of treatment DB

### Field Data Privacy

- **Anonymization**: No personally identifiable farmer information
- **Encryption**: Encrypt data at rest and in transit
- **Access control**: Limit access to authorized personnel only
- **GDPR compliance**: Follow data protection regulations

---

## Future Enhancements

### Planned Features

1. **Severity estimation from images** - Analyze lesion coverage automatically
2. **Amharic translation** - Full localization for Ethiopian farmers
3. **Treatment outcome tracking** - Measure which treatments work best
4. **Cost estimation** - Provide treatment cost estimates in local currency
5. **Weather integration** - Adjust spray timing based on forecast
6. **SMS delivery** - Send treatment advice via SMS for offline farmers
7. **Voice interface** - Audio recommendations for low-literacy users
8. **Treatment database versioning** - A/B test different recommendation strategies

### Research Opportunities

- Analyze which treatments have best outcomes in field data
- Identify factors affecting treatment effectiveness
- Regional variations in disease management
- Economic impact assessment of recommendations

---

## Support

**Technical Issues**:
- GitHub: [repository]/issues
- Email: coffeeguard-tech@example.org

**Content Updates** (Treatments):
- Lead Agronomist: [Name]
- Email: [email]

**Field Testing Support**:
- See: `docs/FIELD_TESTING_GUIDE.md`

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-09-25 | Initial deployment guide |
