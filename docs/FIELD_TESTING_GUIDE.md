# CoffeeGuard AI - Field Testing Guide

**Version**: 1.0  
**For**: Agricultural Extension Officers, Researchers, Field Teams  
**Date**: September 2026

---

## Table of Contents

1. [Overview](#overview)
2. [Objectives](#objectives)
3. [Equipment Needed](#equipment-needed)
4. [Data Collection Protocol](#data-collection-protocol)
5. [Photo Guidelines](#photo-guidelines)
6. [Using the System](#using-the-system)
7. [Feedback Collection](#feedback-collection)
8. [Data Management](#data-management)
9. [Safety & Ethics](#safety--ethics)
10. [Troubleshooting](#troubleshooting)

---

## Overview

Field testing validates CoffeeGuard AI's performance with real farm photos under actual conditions. Current model accuracy is **97.4% on laboratory data** but drops to **63% on external datasets** (especially Cercospora). Field testing helps us:

- Measure real-world accuracy
- Identify failure modes
- Collect training data for improvements
- Build farmer trust through transparent validation

---

## Objectives

### Primary Goals

1. **Validate accuracy** on 200+ farm photos (50 per class minimum)
2. **Collect diverse conditions**: different phones, lighting, angles, disease severity
3. **Track treatment outcomes**: Did recommended actions work?
4. **Identify misclassifications**: What does the model get wrong?

### Success Criteria

- Field accuracy ≥85% (lower than lab is expected)
- All four disease classes represented
- Metadata complete for ≥90% of photos
- Feedback received for ≥60% of predictions

---

## Equipment Needed

### Required

- **Smartphone** with camera (any modern phone with 5+ MP camera)
- **Internet connection** for uploading (mobile data or WiFi)
- **Paper forms** for recording metadata (provided in appendix)
- **GPS device** or phone GPS for location tagging

### Optional but Recommended

- **Portable power bank** for extended field work
- **Color card** for white balance reference
- **Ruler or scale reference** for size documentation
- **Protective cases** for phones in field conditions

---

## Data Collection Protocol

### 1. Site Selection

- **Geographic diversity**: Multiple woredas/kebeles
- **Altitude range**: Cover 1,400-2,200 masl (Ethiopian coffee zones)
- **Farm diversity**: Small/large farms, organic/conventional
- **Seasonal coverage**: Collect during rainy and dry seasons

### 2. Plant Selection

For each disease class, select:
- **10 plants with mild symptoms** (<10% leaf area affected)
- **10 plants with moderate symptoms** (10-30% affected)
- **10 plants with severe symptoms** (>30% affected)

Record plant characteristics:
- Age (years since planting)
- Variety (if known)
- Previous disease history
- Recent treatments applied

### 3. Photo Capture

**Per plant, take 5 photos:**

1. **Close-up of affected leaf** (primary diagnostic photo)
2. **Whole leaf in context** (shows surrounding leaves)
3. **Plant overview** (overall health status)
4. **Symptom detail** (zoomed into lesions/pustules)
5. **Scale reference** (leaf next to ruler or known object)

### 4. Metadata Recording

For each photo, record (use form in Appendix A):

**Required**:
- Date and time
- Location (GPS coordinates or kebele name)
- Disease diagnosis (by expert agronomist - this is ground truth)
- Severity (mild/moderate/severe)
- Weather conditions
- Time of day

**Recommended**:
- Farmer ID (anonymized: F001, F002, etc.)
- Plant age
- Variety
- Recent treatments
- Camera phone model
- Photo quality issues (blur, glare, etc.)

---

## Photo Guidelines

### DO:

✅ **Take photos in natural daylight** (early morning or late afternoon best)  
✅ **Fill frame with leaf** (leaf should be 50-80% of image)  
✅ **Focus clearly** (tap screen to focus before shooting)  
✅ **Include whole leaf** when possible  
✅ **Photograph both upper and lower leaf surfaces** (especially for rust)  
✅ **Keep leaf flat** (avoid heavy curl or fold)  
✅ **Neutral background** (hold leaf against paper/shirt if needed)  
✅ **Multiple angles** if symptoms unclear  

### DON'T:

❌ Don't use flash (causes glare and washes out symptoms)  
❌ Don't photograph in deep shade or twilight  
❌ Don't include multiple leaves overlapping  
❌ Don't photograph wet leaves (wait for dew to dry)  
❌ Don't zoom digitally (move closer instead)  
❌ Don't heavily edit/filter photos before upload  

### Quality Checklist

Before moving to next plant, verify photo is:
- [ ] In focus (zoom in to check)
- [ ] Well-lit (details visible, not too dark/bright)
- [ ] Correct orientation (upright)
- [ ] Symptoms visible
- [ ] Metadata recorded

---

## Using the System

### Web Interface (Recommended for Field Testing)

1. **Navigate to**: `http://coffeeguard-server.local:8501` (local deployment)
2. **Upload photo** via drag-and-drop or file browser
3. **Wait for result** (typically 2-5 seconds)
4. **Record prediction**:
   - Predicted class
   - Confidence percentage
   - Any warnings/rejections
5. **Provide feedback** (see next section)

### API (For Automated Collection)

```bash
# Predict with treatments
curl -X POST "http://server:8000/predict" \
  -F "file=@photo.jpg" \
  -H "X-Request-ID: field-test-001"

# Analyze with CAM and details
curl -X POST "http://server:8000/analyze" \
  -F "file=@photo.jpg" \
  > result.json
```

---

## Feedback Collection

### Immediate Feedback (In-Field)

After each prediction, record:

1. **Was the prediction correct?** (Yes/No)
2. **If NO, what is the correct disease?**
3. **Severity match?** (Does predicted severity match observed?)
4. **Confidence appropriate?** (High confidence for obvious cases?)
5. **Treatment recommendations helpful?** (Clear and actionable?)

### Follow-Up (2-4 Weeks Later)

Return to same plants and assess:

1. **Treatment applied?** (Yes/No, which treatment)
2. **Disease progression**: Better/Same/Worse
3. **Yield impact** (if harvest season)
4. **Farmer feedback**: Ease of use, trust in system

### Recording Feedback

**Using Field Data Collector**:

```python
from coffeeguard.field import FieldDataCollector

collector = FieldDataCollector()

# Record prediction
pred_id = collector.record_prediction(
    image_path="field_photos/F001_plant1_leaf1.jpg",
    predicted_class=2,  # Leaf Rust
    predicted_label="Leaf Rust",
    confidence=0.92,
    metadata={
        "location": "Limu Woreda, Field Site A",
        "gps_coords": {"lat": 8.1234, "lon": 36.5678},
        "farmer_id": "F001",
        "severity": "moderate"
    }
)

# Add feedback after expert verification
collector.add_feedback(
    entry_id=pred_id,
    was_correct=True,
    notes="Clear rust pustules on underside, confident diagnosis"
)

# Generate report
report = collector.generate_report()
print(report)
```

---

## Data Management

### File Naming Convention

```
[FarmerID]_[PlantID]_[LeafID]_[ViewType]_[YYYYMMDD].jpg

Examples:
F001_P01_L01_closeup_20260925.jpg
F001_P01_L01_overview_20260925.jpg
F002_P05_L03_detail_20260926.jpg
```

### Directory Structure

```
field_data/
├── photos/
│   ├── farmer_001/
│   ├── farmer_002/
│   └── ...
├── metadata/
│   ├── site_a_metadata.csv
│   └── site_b_metadata.csv
├── predictions/
│   └── predictions.jsonl
└── feedback/
    └── feedback.jsonl
```

### Backup Protocol

- **Daily**: Copy all photos to external drive
- **Weekly**: Upload to cloud storage (Google Drive/Dropbox)
- **Monthly**: Send full dataset to central coordination team

### Privacy & GDPR

- **Anonymize farmers**: Use codes (F001), not names
- **Consent forms**: Get written permission before photographing
- **No faces**: Don't include farmers or workers in photos
- **Secure storage**: Encrypt data, restricted access
- **Data retention**: State how long data will be kept

---

## Safety & Ethics

### Field Safety

- Work in pairs when visiting remote farms
- Inform local authorities of field visit schedule
- Carry first aid kit and emergency contacts
- Be aware of wildlife/terrain hazards
- Respect farmers' property and crops

### Ethical Considerations

1. **Informed Consent**: Explain study purpose clearly
2. **No harm**: Don't delay urgent treatment for data collection
3. **Fair compensation**: Provide small payment/gift for participation
4. **Feedback to farmers**: Share results, don't just extract data
5. **Data ownership**: Clarify who owns collected data
6. **Benefit sharing**: How will farmers benefit from improved system?

### Agronomist Verification

- **All ground truth labels must be verified by certified agronomist**
- When uncertain, collect sample for lab confirmation
- Document confidence in ground truth (certain/probable/unsure)

---

## Troubleshooting

### Common Issues

**"Image quality too low" rejection**
- Solution: Retake in better light, ensure focus
- Check: Brightness, blur, file size

**"Not recognized as coffee leaf" rejection**
- Solution: Ensure whole leaf visible, neutral background
- Check: Leaf fills 50%+ of frame, not obscured

**App/API not responding**
- Check: Internet connection
- Check: Server status (`curl http://server:8000/health`)
- Restart: If local server, restart service

**Photos not uploading**
- Check: File size <10MB
- Check: Format is JPEG or PNG
- Try: Compress/resize if too large

**GPS not working**
- Enable location services on phone
- Alternative: Manually record location name
- Use: What3words app for precise location

### Contact Support

**Technical Issues**:
- Email: coffeeguard-support@example.org
- Phone: +251-XX-XXX-XXXX
- WhatsApp: +251-XX-XXX-XXXX

**Scientific Questions**:
- Lead Agronomist: Dr. [Name]
- Email: [email]

---

## Appendix A: Field Data Collection Form

```
COFFEEGUARD FIELD TESTING - DATA COLLECTION FORM

Date: ___/___/_____ Time: _____:_____
Location: __________________________ GPS: ___.___, ___. ___
Collector: _________________________ Form ID: __________

PLANT INFORMATION
Farmer ID: F____    Plant ID: P____    Leaf ID: L____
Plant Age: ____ years    Variety: ______________

DIAGNOSIS (by certified agronomist)
Disease: ☐ Healthy ☐ Cercospora ☐ Leaf Rust ☐ Phoma
Severity: ☐ Mild ☐ Moderate ☐ Severe
Confidence: ☐ Certain ☐ Probable ☐ Unsure

PHOTO INFORMATION
Photo IDs: _________________________________
Weather: ☐ Sunny ☐ Partly Cloudy ☐ Overcast ☐ Rainy
Time: ☐ Morning (6-10am) ☐ Midday (10am-2pm) ☐ Afternoon (2-6pm)
Camera: _________________

SYSTEM PREDICTION
Predicted: ________________ Confidence: _____%
Status: ☐ Accepted ☐ Uncertain ☐ Rejected
Correct?: ☐ Yes ☐ No    If No, why?: _______________

NOTES:
_______________________________________________________
_______________________________________________________
_______________________________________________________
```

---

## Appendix B: Safety Checklist

Before field deployment:

- [ ] Team briefed on objectives and protocol
- [ ] Consent forms prepared and translated
- [ ] Equipment charged and tested
- [ ] Backup power available
- [ ] First aid kit packed
- [ ] Emergency contacts listed
- [ ] Transportation arranged
- [ ] Farmer appointments confirmed
- [ ] Weather forecast checked
- [ ] Data backup system ready

---

## Revision History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | 2026-09-25 | Initial release | CoffeeGuard Team |

---

**Questions or feedback on this guide?**  
Contact: coffeeguard-fieldtest@example.org
