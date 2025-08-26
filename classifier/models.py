from django.db import models
from django.contrib.auth.models import User


class Prediction(models.Model):
    user = models.ForeignKey(
        User, on_delete=models.CASCADE, null=True, blank=True
    )
    anon_id = models.CharField(
        max_length=50, null=True, blank=True, db_index=True
    )  # for anonymous session tracking

    canvas_image = models.ImageField(upload_to="predictions/canvas/")
    heatmap_image = models.ImageField(upload_to="predictions/heatmaps/")
    predicted_class = models.IntegerField()
    confidence = models.FloatField()
    probabilities = models.JSONField()

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        who = self.user.username if self.user else f"anon:{self.anon_id or 'unknown'}"
        return f"Prediction {self.id} by {who} → {self.predicted_class} ({self.confidence:.2f})"


class Report(models.Model):
    prediction = models.ForeignKey(
        Prediction, on_delete=models.CASCADE, related_name="reports"
    )
    user = models.ForeignKey(
        User, on_delete=models.CASCADE, null=True, blank=True
    )
    anon_id = models.CharField(
        max_length=50, null=True, blank=True, db_index=True
    )  # for anonymous reports

    correct_label = models.IntegerField(null=True, blank=True)  # optional: true digit
    reason = models.TextField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        who = self.user.username if self.user else f"anon:{self.anon_id or 'unknown'}"
        return f"Report {self.id} on Prediction {self.prediction_id} by {who}"
