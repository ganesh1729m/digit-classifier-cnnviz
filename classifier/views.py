# classifier/views.py

import base64
import io
import json
from PIL import Image
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.base import ContentFile
from django.utils.crypto import get_random_string
from django.contrib.auth.forms import UserCreationForm
from django.contrib import messages
from .models import Prediction, Report
from classifier.ml.predict import predict
from django.shortcuts import get_object_or_404
from django.contrib.auth.decorators import login_required
import cv2
import numpy as np


# --------------------
# User Registration
# --------------------
def register(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            messages.success(request, f"Account created for {user.username}! You can now log in.")
            return redirect('login')
    else:
        form = UserCreationForm()
    return render(request, 'classifier/register.html', {'form': form})


# --------------------
# Helper: User or Anonymous
# --------------------
def get_user_or_session(request):
    """Return (user, anon_id) to track anonymous usage."""
    if request.user.is_authenticated:
        return request.user, None
    if "anon_id" not in request.session:
        request.session["anon_id"] = get_random_string(16)
    return None, request.session["anon_id"]


# --------------------
# Home Page
# --------------------
def home(request):
    return render(request, "classifier/home.html")



# --------------------
# Dashboard
# --------------------
@login_required
def dashboard(request):
    user = request.user
    anon_id = request.session.get("anon_id")

    if user.is_superuser:
        # Admin sees all predictions and reports
        predictions = Prediction.objects.all().order_by('-created_at')
        reports = Report.objects.all().order_by('-created_at')
    else:
        # Regular users see their own + anonymous predictions/reports
        predictions = Prediction.objects.filter(user=user).order_by('-created_at')
        if anon_id:
            anon_preds = Prediction.objects.filter(user__isnull=True, anon_id=anon_id)
            predictions = predictions | anon_preds
            predictions = predictions.distinct().order_by('-created_at')

        reports = Report.objects.filter(user=user).order_by('-created_at')
        if anon_id:
            anon_reports = Report.objects.filter(user__isnull=True, prediction__anon_id=anon_id)
            reports = reports | anon_reports
            reports = reports.distinct().order_by('-created_at')

    return render(request, 'classifier/dashboard.html', {
        'predictions': predictions,
        'reports': reports,
        'is_admin': user.is_superuser  # pass this flag to template for admin-only features
    })


# --------------------
# Array to Base64 (Utils)
# --------------------
def array_to_base64(arr):
    """
    Convert a HxW (gray) or HxWxC (BGR/RGB) ndarray to base64 PNG.
    Accepts None and returns None.
    Always outputs as RGB PNG.
    """
    if arr is None:
        return None
    arr = np.asarray(arr)
    if arr.ndim == 2:
        # grayscale -> RGB
        arr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    elif arr.ndim == 3:
        # assume BGR if OpenCV produced it; convert to RGB for PIL
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        # Heuristic: if last dim is 3, treat as BGR->RGB
        if arr.shape[2] == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        elif arr.shape[2] == 4:
            # If ever RGBA/BGRA, drop alpha for simplicity
            arr = arr[:, :, :3]
    else:
        # Unexpected shape; make it safe (return None)
        return None

    img_pil = Image.fromarray(arr)
    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# --------------------
# Prediction API
# --------------------
@csrf_exempt
def predict_digit(request):
    if request.method != "POST":
        return JsonResponse({"success": False, "message": "POST required"}, status=405)

    try:
        data_url = request.POST.get("image")
        if not data_url:
            return JsonResponse({"success": False, "message": "No image received"}, status=400)

        # Decode canvas → PIL
        _, encoded = data_url.split(",", 1)
        image_bytes = base64.b64decode(encoded)
        image_pil = Image.open(io.BytesIO(image_bytes)).convert("L")

        # Run prediction
        result = predict(image_pil, from_canvas=True, Need_Features=True)

        # 🔹 Use array_to_base64 here instead of manual PIL conversion
        heatmap_dataurl = None
        heatmap_arr = result.get("heatmap")
        if heatmap_arr is not None:
            heatmap_b64 = array_to_base64(heatmap_arr)
            heatmap_dataurl = f"data:image/png;base64,{heatmap_b64}"


        return JsonResponse({
            "success": True,
            "predicted_class": int(result["predicted_class"]),
            "confidence": float(result["confidence"]),
            "probabilities": result["probabilities"],
            "heatmap": heatmap_dataurl,
            "feature_maps": result["feature_maps"],
            # "occlusion": occlusion_dataurl,
        })

    except Exception as e:
        return JsonResponse({"success": False, "message": str(e)}, status=500)


# --------------------
# Save Prediction
# --------------------
@csrf_exempt
def save_prediction(request):
    """
    Save a prediction with canvas + heatmap + metadata.
    """
    if request.method != "POST":
        return JsonResponse({"success": False, "message": "POST required"}, status=405)

    try:
        data = json.loads(request.body.decode("utf-8"))

        canvas_data = data.get("canvas_image")
        predicted_class = data.get("predicted_class")
        if not canvas_data or predicted_class is None:
            return JsonResponse({"success": False, "message": "canvas_image and predicted_class required"}, status=400)

        # Decode canvas
        _, canvas_b64 = canvas_data.split(",", 1)
        canvas_file = ContentFile(base64.b64decode(canvas_b64), name="canvas.png")

        # Decode heatmap if provided
        heatmap_file = None
        heatmap_data = data.get("heatmap_image")
        if heatmap_data:
            try:
                _, heatmap_b64 = heatmap_data.split(",", 1)
                heatmap_file = ContentFile(base64.b64decode(heatmap_b64), name="heatmap.png")
            except Exception:
                heatmap_file = None

        user, anon_id = get_user_or_session(request)

        pred = Prediction.objects.create(
            user=user,
            canvas_image=canvas_file,
            heatmap_image=heatmap_file,
            predicted_class=int(predicted_class),
            confidence=float(data.get("confidence") or 0.0),
            probabilities=data.get("probabilities") or []
        )

        return JsonResponse({"success": True, "prediction_id": pred.id, "message": "Prediction saved"})

    except Exception as e:
        return JsonResponse({"success": False, "message": str(e)}, status=500)


# --------------------
# Re-run Prediction
# --------------------
@login_required
@csrf_exempt
def rerun_prediction(request, prediction_id):
    if request.method != "POST":
        return JsonResponse({"success": False, "message": "POST required"}, status=405)
    try:
        pred = get_object_or_404(Prediction, id=prediction_id)
        img = Image.open(pred.canvas_image.path).convert("L")

        result = predict(img, from_canvas=True)

        # Heatmap → base64
        heatmap_arr = result.get("heatmap")
        heatmap_dataurl = None
        if heatmap_arr is not None:
            heatmap_b64 = array_to_base64(heatmap_arr)
            if heatmap_b64:
                heatmap_dataurl = f"data:image/png;base64,{heatmap_b64}"
                # save to model
                heatmap_file = ContentFile(base64.b64decode(heatmap_b64), name="heatmap.png")
                pred.heatmap_image.save("heatmap.png", heatmap_file, save=False)

        # Update prediction
        pred.predicted_class = int(result["predicted_class"])
        pred.confidence = float(result["confidence"])
        pred.probabilities = result.get("probabilities", [])
        pred.save()

        return JsonResponse({
            "success": True,
            "predicted_class": pred.predicted_class,
            "confidence": pred.confidence,
            "probabilities": pred.probabilities,
            "heatmap_url": heatmap_dataurl
        })
    except Exception as e:
        return JsonResponse({"success": False, "message": str(e)}, status=500)


# --------------------
# Re-run Report
# --------------------
@login_required
@csrf_exempt
def rerun_report(request, report_id):
    if request.method != "POST":
        return JsonResponse({"success": False, "message": "POST required"}, status=405)
    try:
        report = get_object_or_404(Report, id=report_id)
        pred = report.prediction

        # Open saved canvas image
        img = Image.open(pred.canvas_image.path).convert("L")

        result = predict(img, from_canvas=False)

        # Update prediction
        pred.predicted_class = int(result["predicted_class"])
        pred.confidence = float(result["confidence"])
        pred.probabilities = result.get("probabilities", [])
        pred.save()

        # Heatmap → base64
        heatmap_arr = result.get("heatmap")
        heatmap_dataurl = None
        if heatmap_arr is not None:
            heatmap_b64 = array_to_base64(heatmap_arr)
            if heatmap_b64:
                heatmap_dataurl = f"data:image/png;base64,{heatmap_b64}"
                # save to model
                heatmap_file = ContentFile(base64.b64decode(heatmap_b64), name="heatmap.png")
                pred.heatmap_image.save("heatmap.png", heatmap_file, save=True)

        return JsonResponse({
            "success": True,
            "predicted_class": pred.predicted_class,
            "confidence": pred.confidence,
            "probabilities": pred.probabilities,
            "heatmap": heatmap_dataurl,
            "reason": report.reason
        })

    except Exception as e:
        return JsonResponse({"success": False, "message": str(e)}, status=500)


#--------------------
# Report Misclassification (Fixed)
#--------------------
@csrf_exempt
def report_misclassification(request):
    """
    Handles misclassification reports.

    If the prediction already exists, attach a report to it.
    Only create a new prediction if it's truly a new image not yet predicted.
    """
    if request.method != "POST":
        return JsonResponse({"success": False, "message": "POST required"}, status=405)

    try:
        data = json.loads(request.body.decode("utf-8"))
        user, anon_id = get_user_or_session(request)

        # --------------------
        # Case 1: Existing prediction
        # --------------------
        if "prediction_id" in data:
            pred = get_object_or_404(Prediction, id=data["prediction_id"])
            reason = data.get("reason", "")
            correct_label = data.get("correct_label")

            if correct_label is not None:
                # Save the correct label in report
                reason = f"Correct label: {correct_label}. {reason}"

            Report.objects.create(
                prediction=pred,
                user=user,
                reason=reason
            )

            return JsonResponse({"success": True, "message": "Report attached to existing prediction"})

        # --------------------
        # Case 2: New prediction (optional, only if reporting a brand-new image)
        # --------------------
        canvas_data = data.get("canvas_image")
        if not canvas_data:
            return JsonResponse({"success": False, "message": "canvas_image required"}, status=400)

        _, canvas_b64 = canvas_data.split(",", 1)
        canvas_file = ContentFile(base64.b64decode(canvas_b64), name="canvas.png")
        img = Image.open(io.BytesIO(base64.b64decode(canvas_b64))).convert("L")

        # Run ML prediction
        result = predict(img, from_canvas=True)
        predicted_class = int(result["predicted_class"])
        confidence = float(result["confidence"])
        probabilities = result.get("probabilities", [])

        # Heatmap
        heatmap_file = None
        heatmap_arr = result.get("heatmap")
        heatmap_dataurl = None
        if heatmap_arr is not None:
            heatmap_b64 = array_to_base64(heatmap_arr)
            if heatmap_b64:
                heatmap_dataurl = f"data:image/png;base64,{heatmap_b64}"
                heatmap_file = ContentFile(base64.b64decode(heatmap_b64), name="heatmap.png")

        # Save new prediction
        pred = Prediction.objects.create(
            user=user,
            canvas_image=canvas_file,
            heatmap_image=heatmap_file,
            predicted_class=predicted_class,
            confidence=confidence,
            probabilities=probabilities
        )

        reason = data.get("reason", "")
        correct_label = data.get("correct_label")
        if correct_label is not None:
            reason = f"Correct label: {correct_label}. {reason}"

        # Save report linked to this prediction
        Report.objects.create(
            prediction=pred,
            user=user,
            reason=reason
        )

        return JsonResponse({"success": True, "message": "Report submitted", "prediction_id": pred.id})

    except Exception as e:
        return JsonResponse({"success": False, "message": str(e)}, status=500)
