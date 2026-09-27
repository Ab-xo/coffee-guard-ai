"""Field data collection system for real-world validation."""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict


@dataclass
class FeedbackEntry:
    """Represents a single field feedback entry."""
    
    id: str
    timestamp: str
    image_path: str
    predicted_class: int
    predicted_label: str
    confidence: float
    
    # User feedback
    was_correct: Optional[bool] = None
    actual_label: Optional[str] = None
    severity: Optional[str] = None
    
    # Context metadata
    location: Optional[str] = None
    gps_coords: Optional[Dict[str, float]] = None
    farmer_id: Optional[str] = None
    plant_age_years: Optional[int] = None
    treatment_applied: Optional[str] = None
    treatment_outcome: Optional[str] = None
    
    # Image metadata
    device_model: Optional[str] = None
    weather_conditions: Optional[str] = None
    time_of_day: Optional[str] = None
    
    # Quality metrics
    image_quality_score: Optional[float] = None
    ood_distance: Optional[float] = None
    
    # Notes
    notes: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeedbackEntry":
        """Create from dictionary."""
        return cls(**data)


class FieldDataCollector:
    """Collects and manages field testing data and feedback."""
    
    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize field data collector.
        
        Args:
            data_dir: Directory for storing field data. Defaults to data/field/
        """
        if data_dir is None:
            repo_root = Path(__file__).parents[3]
            data_dir = repo_root / "data" / "field"
        
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.feedback_file = self.data_dir / "feedback.jsonl"
        self.images_dir = self.data_dir / "images"
        self.images_dir.mkdir(exist_ok=True)
    
    def record_prediction(
        self,
        image_path: str,
        predicted_class: int,
        predicted_label: str,
        confidence: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Record a prediction for later feedback collection.
        
        Args:
            image_path: Path to the input image
            predicted_class: Model's predicted class ID
            predicted_label: Human-readable class label
            confidence: Prediction confidence
            metadata: Optional additional metadata
        
        Returns:
            Unique ID for this prediction
        """
        entry_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        entry = FeedbackEntry(
            id=entry_id,
            timestamp=timestamp,
            image_path=image_path,
            predicted_class=predicted_class,
            predicted_label=predicted_label,
            confidence=confidence
        )
        
        # Add metadata if provided
        if metadata:
            for key, value in metadata.items():
                if hasattr(entry, key):
                    setattr(entry, key, value)
        
        # Save to JSONL file
        with open(self.feedback_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry.to_dict()) + '\n')
        
        return entry_id
    
    def add_feedback(
        self,
        entry_id: str,
        was_correct: bool,
        actual_label: Optional[str] = None,
        **kwargs
    ) -> bool:
        """Add feedback for a previous prediction.
        
        Args:
            entry_id: ID from record_prediction()
            was_correct: Whether the prediction was correct
            actual_label: Correct label if prediction was wrong
            **kwargs: Additional feedback fields (severity, notes, etc.)
        
        Returns:
            True if feedback was added successfully
        """
        # Load all entries
        entries = self.load_all_feedback()
        
        # Find and update the entry
        found = False
        for entry in entries:
            if entry['id'] == entry_id:
                entry['was_correct'] = was_correct
                if actual_label:
                    entry['actual_label'] = actual_label
                entry.update(kwargs)
                found = True
                break
        
        if not found:
            return False
        
        # Rewrite the file
        with open(self.feedback_file, 'w', encoding='utf-8') as f:
            for entry in entries:
                f.write(json.dumps(entry) + '\n')
        
        return True
    
    def load_all_feedback(self) -> List[Dict[str, Any]]:
        """Load all feedback entries.
        
        Returns:
            List of feedback entries as dictionaries
        """
        if not self.feedback_file.exists():
            return []
        
        entries = []
        with open(self.feedback_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        
        return entries
    
    def get_statistics(self) -> Dict[str, Any]:
        """Calculate statistics from collected feedback.
        
        Returns:
            Dictionary with accuracy metrics and statistics
        """
        entries = self.load_all_feedback()
        
        if not entries:
            return {
                "total_predictions": 0,
                "feedback_received": 0,
                "accuracy": None
            }
        
        total = len(entries)
        with_feedback = [e for e in entries if e.get('was_correct') is not None]
        correct = [e for e in with_feedback if e['was_correct']]
        
        stats = {
            "total_predictions": total,
            "feedback_received": len(with_feedback),
            "feedback_rate": len(with_feedback) / total if total > 0 else 0,
            "correct_predictions": len(correct),
            "accuracy": len(correct) / len(with_feedback) if with_feedback else None
        }
        
        # Per-class accuracy
        if with_feedback:
            per_class = {}
            for entry in with_feedback:
                label = entry['predicted_label']
                if label not in per_class:
                    per_class[label] = {"total": 0, "correct": 0}
                per_class[label]["total"] += 1
                if entry['was_correct']:
                    per_class[label]["correct"] += 1
            
            for label in per_class:
                per_class[label]["accuracy"] = (
                    per_class[label]["correct"] / per_class[label]["total"]
                )
            
            stats["per_class"] = per_class
        
        # Confidence analysis
        if with_feedback:
            high_conf = [e for e in with_feedback if e['confidence'] >= 0.8]
            if high_conf:
                high_conf_correct = [e for e in high_conf if e['was_correct']]
                stats["high_confidence_accuracy"] = (
                    len(high_conf_correct) / len(high_conf)
                )
        
        return stats
    
    def export_for_retraining(self, output_dir: Optional[Path] = None) -> Path:
        """Export validated feedback as training data.
        
        Args:
            output_dir: Where to save the export. Defaults to data/field/export/
        
        Returns:
            Path to the export directory
        """
        if output_dir is None:
            output_dir = self.data_dir / "export"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load validated entries (those with feedback)
        entries = self.load_all_feedback()
        validated = [e for e in entries if e.get('was_correct') is not None]
        
        # Create manifest for retraining
        manifest = []
        for entry in validated:
            # Use actual label if prediction was wrong, otherwise predicted label
            true_label = (
                entry['actual_label'] 
                if not entry['was_correct'] and entry.get('actual_label')
                else entry['predicted_label']
            )
            
            manifest.append({
                "image_path": entry['image_path'],
                "label": true_label,
                "confidence": entry['confidence'],
                "metadata": {
                    "source": "field_feedback",
                    "original_prediction": entry['predicted_label'],
                    "was_correct": entry['was_correct'],
                    "timestamp": entry['timestamp']
                }
            })
        
        # Save manifest
        manifest_path = output_dir / "field_data_manifest.json"
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"Exported {len(manifest)} validated entries to {manifest_path}")
        return output_dir
    
    def generate_report(self, output_path: Optional[Path] = None) -> str:
        """Generate a human-readable report of field testing results.
        
        Args:
            output_path: Where to save the report. If None, returns as string.
        
        Returns:
            Report content as string
        """
        stats = self.get_statistics()
        
        lines = []
        lines.append("=" * 60)
        lines.append("COFFEEGUARD FIELD TESTING REPORT")
        lines.append("=" * 60)
        lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"\nTotal predictions: {stats['total_predictions']}")
        lines.append(f"Feedback received: {stats['feedback_received']} "
                    f"({stats['feedback_rate']:.1%})")
        
        if stats['accuracy'] is not None:
            lines.append(f"\nOverall accuracy: {stats['accuracy']:.1%}")
            lines.append(f"Correct predictions: {stats['correct_predictions']}")
        
        if 'per_class' in stats:
            lines.append("\n" + "-" * 60)
            lines.append("PER-CLASS ACCURACY")
            lines.append("-" * 60)
            for label, metrics in stats['per_class'].items():
                lines.append(f"\n{label}:")
                lines.append(f"  Total: {metrics['total']}")
                lines.append(f"  Correct: {metrics['correct']}")
                lines.append(f"  Accuracy: {metrics['accuracy']:.1%}")
        
        if 'high_confidence_accuracy' in stats:
            lines.append("\n" + "-" * 60)
            lines.append(f"High confidence (≥80%) accuracy: "
                        f"{stats['high_confidence_accuracy']:.1%}")
        
        lines.append("\n" + "=" * 60)
        
        report = "\n".join(lines)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"Report saved to {output_path}")
        
        return report
