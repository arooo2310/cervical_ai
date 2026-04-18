from django import template
register = template.Library()

@register.filter
def risk_label(score, threshold=0.005):
    """'High' if score >= threshold else 'Low'."""
    try:
        return "High" if float(score) >= float(threshold) else "Low"
    except (TypeError, ValueError):
        return "Low"

@register.filter
def risk_class(score, threshold=0.5):
    """Lowercase version for CSS class."""
    return risk_label(score, threshold).lower()
