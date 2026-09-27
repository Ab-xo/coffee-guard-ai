"""Treatment recommendation engine."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any

# Class ID to disease name mapping
CLASS_NAMES = {
    0: "healthy",
    1: "cercospora",
    2: "leaf_rust",
    3: "phoma"
}


class TreatmentRecommender:
    """Provides treatment recommendations based on disease predictions."""
    
    def __init__(self, treatments_path: Optional[Path] = None):
        """Initialize recommender with treatment database.
        
        Args:
            treatments_path: Path to treatments.json. If None, uses default location.
        """
        if treatments_path is None:
            # Default to ml/knowledge/treatments.json relative to repo root
            repo_root = Path(__file__).parents[3]
            treatments_path = repo_root / "ml" / "knowledge" / "treatments.json"
        
        self.treatments_path = Path(treatments_path)
        self.treatments_db = self._load_treatments()
    
    def _load_treatments(self) -> Dict[str, Any]:
        """Load treatment database from JSON file."""
        if not self.treatments_path.exists():
            raise FileNotFoundError(
                f"Treatment database not found at {self.treatments_path}"
            )
        
        with open(self.treatments_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_recommendation(
        self,
        class_id: int,
        confidence: float,
        severity: str = "mild"
    ) -> Dict[str, Any]:
        """Get treatment recommendation for a predicted disease.
        
        Args:
            class_id: Predicted disease class (0-3)
            confidence: Prediction confidence (0-1)
            severity: Disease severity level ('mild', 'moderate', 'severe')
        
        Returns:
            Dictionary with treatment recommendations
        """
        disease_key = CLASS_NAMES.get(class_id, "healthy")
        disease_data = self.treatments_db["diseases"].get(disease_key, {})
        
        if not disease_data:
            return self._default_recommendation()
        
        # Build recommendation
        recommendation = {
            "disease": disease_data.get("name", disease_key.title()),
            "description": disease_data.get("description", ""),
            "confidence": confidence,
            "severity": severity,
            "urgency": disease_data.get("urgency", "moderate"),
            "treatments": [],
            "prevention": disease_data.get("prevention", []),
            "advice": disease_data.get("advice", "")
        }
        
        # Get severity-specific treatments
        if "treatments" in disease_data:
            treatments = disease_data["treatments"]
            if isinstance(treatments, dict) and severity in treatments:
                severity_treatments = treatments[severity]
                
                # Handle different formats
                if isinstance(severity_treatments, dict):
                    # Has 'organic', 'conventional', 'urgent', etc.
                    if "urgent" in severity_treatments:
                        recommendation["urgent_message"] = severity_treatments["urgent"]
                    if "actions" in severity_treatments:
                        recommendation["treatments"] = severity_treatments["actions"]
                    if "organic" in severity_treatments:
                        recommendation["organic_treatments"] = severity_treatments["organic"]
                    if "conventional" in severity_treatments:
                        recommendation["conventional_treatments"] = severity_treatments["conventional"]
                    if "critical_actions" in severity_treatments:
                        recommendation["treatments"] = severity_treatments["critical_actions"]
                    if "cost_warning" in severity_treatments:
                        recommendation["cost_warning"] = severity_treatments["cost_warning"]
                elif isinstance(severity_treatments, list):
                    recommendation["treatments"] = severity_treatments
        
        # Add general safety and support info
        general = self.treatments_db.get("general_advice", {})
        recommendation["safety_note"] = general.get("safety", "")
        recommendation["support_resources"] = general.get("resources", {})
        
        return recommendation
    
    def _default_recommendation(self) -> Dict[str, Any]:
        """Return default recommendation when disease not found."""
        return {
            "disease": "Unknown",
            "description": "Disease information not available",
            "treatments": [],
            "prevention": [],
            "advice": "Please consult with an agricultural extension officer."
        }
    
    def estimate_severity(
        self,
        confidence: float,
        image_quality_metrics: Optional[Dict[str, float]] = None
    ) -> str:
        """Estimate disease severity from confidence and image metrics.
        
        This is a rough proxy since we don't have ground truth severity labels.
        
        Args:
            confidence: Model confidence (0-1)
            image_quality_metrics: Optional quality metrics from preprocessing
        
        Returns:
            Severity level: 'mild', 'moderate', or 'severe'
        """
        # High confidence suggests clear, possibly advanced disease
        # Low confidence might indicate early/mild disease or uncertainty
        
        if confidence >= 0.95:
            # Very confident prediction - likely clear symptoms
            return "moderate"  # Conservative: don't assume severe without expert
        elif confidence >= 0.80:
            return "mild"
        else:
            # Low confidence - either very mild or uncertain
            return "mild"
    
    def format_for_display(
        self,
        recommendation: Dict[str, Any],
        format_type: str = "markdown"
    ) -> str:
        """Format recommendation for display.
        
        Args:
            recommendation: Recommendation dictionary from get_recommendation()
            format_type: Output format ('markdown', 'text', or 'html')
        
        Returns:
            Formatted string
        """
        if format_type == "markdown":
            return self._format_markdown(recommendation)
        elif format_type == "html":
            return self._format_html(recommendation)
        else:
            return self._format_text(recommendation)
    
    def _format_markdown(self, rec: Dict[str, Any]) -> str:
        """Format as markdown."""
        lines = []
        
        # Header
        lines.append(f"## {rec['disease']}")
        if rec.get('description'):
            lines.append(f"\n{rec['description']}")
        
        # Urgency warning
        if rec.get('urgency') and rec['urgency'] != 'moderate':
            lines.append(f"\n**⚠️ Urgency: {rec['urgency'].upper()}**")
        
        if rec.get('urgent_message'):
            lines.append(f"\n🚨 **{rec['urgent_message']}**")
        
        # Treatments
        if rec.get('treatments'):
            lines.append("\n### Recommended Actions")
            for i, treatment in enumerate(rec['treatments'], 1):
                lines.append(f"{i}. {treatment}")
        
        if rec.get('organic_treatments'):
            lines.append("\n### Organic Treatment Options")
            for i, treatment in enumerate(rec['organic_treatments'], 1):
                lines.append(f"{i}. {treatment}")
        
        if rec.get('conventional_treatments'):
            lines.append("\n### Conventional Treatment Options")
            for i, treatment in enumerate(rec['conventional_treatments'], 1):
                lines.append(f"{i}. {treatment}")
        
        # Prevention
        if rec.get('prevention'):
            lines.append("\n### Prevention")
            for item in rec['prevention']:
                lines.append(f"- {item}")
        
        # Advice
        if rec.get('advice'):
            lines.append(f"\n### Advice\n{rec['advice']}")
        
        # Safety
        if rec.get('safety_note'):
            lines.append(f"\n### Safety\n⚠️ {rec['safety_note']}")
        
        # Cost warning
        if rec.get('cost_warning'):
            lines.append(f"\n💰 **{rec['cost_warning']}**")
        
        return "\n".join(lines)
    
    def _format_text(self, rec: Dict[str, Any]) -> str:
        """Format as plain text."""
        lines = []
        lines.append(f"Disease: {rec['disease']}")
        if rec.get('description'):
            lines.append(f"Description: {rec['description']}")
        if rec.get('treatments'):
            lines.append("\nTreatments:")
            for t in rec['treatments']:
                lines.append(f"  - {t}")
        if rec.get('prevention'):
            lines.append("\nPrevention:")
            for p in rec['prevention']:
                lines.append(f"  - {p}")
        return "\n".join(lines)
    
    def _format_html(self, rec: Dict[str, Any]) -> str:
        """Format as HTML."""
        html = f"<h2>{rec['disease']}</h2>"
        if rec.get('description'):
            html += f"<p>{rec['description']}</p>"
        if rec.get('treatments'):
            html += "<h3>Treatments</h3><ul>"
            for t in rec['treatments']:
                html += f"<li>{t}</li>"
            html += "</ul>"
        if rec.get('prevention'):
            html += "<h3>Prevention</h3><ul>"
            for p in rec['prevention']:
                html += f"<li>{p}</li>"
            html += "</ul>"
        return html


# Convenience function
def get_treatment_advice(
    class_id: int,
    confidence: float,
    severity: Optional[str] = None,
    format_type: str = "markdown"
) -> str:
    """Quick function to get formatted treatment advice.
    
    Args:
        class_id: Predicted disease class (0-3)
        confidence: Prediction confidence (0-1)
        severity: Disease severity ('mild', 'moderate', 'severe') or None to auto-estimate
        format_type: Output format ('markdown', 'text', or 'html')
    
    Returns:
        Formatted treatment recommendation string
    """
    recommender = TreatmentRecommender()
    
    if severity is None:
        severity = recommender.estimate_severity(confidence)
    
    recommendation = recommender.get_recommendation(class_id, confidence, severity)
    return recommender.format_for_display(recommendation, format_type)
