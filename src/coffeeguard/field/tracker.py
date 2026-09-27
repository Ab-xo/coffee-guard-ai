"""Track model accuracy in field deployment."""

import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta


class AccuracyTracker:
    """Track and analyze model performance over time in field deployment."""
    
    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize accuracy tracker.
        
        Args:
            data_dir: Directory for tracking data. Defaults to artifacts/field_tracking/
        """
        if data_dir is None:
            repo_root = Path(__file__).parents[3]
            data_dir = repo_root / "artifacts" / "field_tracking"
        
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_file = self.data_dir / "accuracy_metrics.jsonl"
    
    def record_batch_metrics(
        self,
        date: str,
        total_predictions: int,
        correct_predictions: int,
        per_class_accuracy: Dict[str, float],
        metadata: Optional[Dict] = None
    ):
        """Record accuracy metrics for a batch/period.
        
        Args:
            date: Date string (YYYY-MM-DD)
            total_predictions: Total predictions made
            correct_predictions: Number of correct predictions
            per_class_accuracy: Accuracy per disease class
            metadata: Optional additional information
        """
        entry = {
            "date": date,
            "timestamp": datetime.now().isoformat(),
            "total_predictions": total_predictions,
            "correct_predictions": correct_predictions,
            "accuracy": correct_predictions / total_predictions if total_predictions > 0 else 0,
            "per_class_accuracy": per_class_accuracy,
            "metadata": metadata or {}
        }
        
        with open(self.metrics_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry) + '\n')
    
    def get_trends(self, days: int = 30) -> Dict:
        """Analyze accuracy trends over time.
        
        Args:
            days: Number of days to analyze
        
        Returns:
            Dictionary with trend analysis
        """
        if not self.metrics_file.exists():
            return {"error": "No metrics data available"}
        
        # Load recent metrics
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_metrics = []
        
        with open(self.metrics_file, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                entry_date = datetime.fromisoformat(entry['timestamp'])
                if entry_date >= cutoff_date:
                    recent_metrics.append(entry)
        
        if not recent_metrics:
            return {"error": f"No metrics in last {days} days"}
        
        # Calculate trends
        accuracies = [m['accuracy'] for m in recent_metrics]
        
        return {
            "period_days": days,
            "total_entries": len(recent_metrics),
            "mean_accuracy": sum(accuracies) / len(accuracies),
            "min_accuracy": min(accuracies),
            "max_accuracy": max(accuracies),
            "latest_accuracy": accuracies[-1] if accuracies else None,
            "trend": "improving" if len(accuracies) > 1 and accuracies[-1] > accuracies[0] else "stable"
        }
