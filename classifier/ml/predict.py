import base64
import io
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter, ImageOps
from torchvision import transforms

from .cnn_model import CNNModel

# -----------------------
# Device setup
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# Input Transformation (MNIST-like)
# -----------------------
transform = transforms.Compose([
    transforms.Grayscale(),         # Ensure single channel
    transforms.Resize((28, 28)),    # Match MNIST input size
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# -----------------------
# Model Loader (singleton)
# -----------------------
_model_cache = None

def load_model():
    global _model_cache
    if _model_cache is None:
        model = CNNModel()
        model.load_state_dict(torch.load("classifier/ml/mnist_cnn_aug.pth", map_location=device))
        model.to(device)
        model.eval()
        _model_cache = model
    return _model_cache

# -----------------------
# Utilities
# -----------------------
def tensor_to_base64(tensor):
    """
    Convert a single-channel tensor to base64 PNG (grayscale), upscaled to 56x56.
    Returns a base64 string (no data URL prefix).
    """
    arr = tensor.detach().cpu().numpy()
    arr -= arr.min()
    arr /= (arr.max() + 1e-8)
    arr = np.uint8(arr * 255)
    img_pil = Image.fromarray(arr)
    buf = io.BytesIO()
    img_pil = img_pil.resize((56, 56), Image.NEAREST)  # upscale for visibility
    img_pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

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

# -----------------------
# Occlusion Sensitivity
# -----------------------
def occlusion_sensitivity(model, input_tensor, target_class, patch_size=7, stride=3):
    """
    Generate occlusion sensitivity map.
    Args:
        model: the trained CNN
        input_tensor: shape [1, 1, 28, 28]
        target_class: int, predicted class index
        patch_size: size of occlusion patch
        stride: stride to slide patch
    Returns:
        np.array of shape (28,28) with values in [0,1]
    """
    model.eval()
    with torch.no_grad():
        _, _, H, W = input_tensor.shape
        base_prob = F.softmax(model(input_tensor), dim=1)[0, target_class].item()

        sensitivity_map = np.zeros((H, W), dtype=np.float32)

        for y in range(0, H - patch_size + 1, stride):
            for x in range(0, W - patch_size + 1, stride):
                occluded = input_tensor.clone()
                # Occlude with zero (black square)
                occluded[:, :, y:y+patch_size, x:x+patch_size] = 0

                prob = F.softmax(model(occluded), dim=1)[0, target_class].item()
                drop = base_prob - prob
                sensitivity_map[y:y+patch_size, x:x+patch_size] = drop

        # Normalize to [0,1]
        mn, mx = sensitivity_map.min(), sensitivity_map.max()
        if mx > mn:
            sensitivity_map = (sensitivity_map - mn) / (mx + 1e-8)
        else:
            sensitivity_map[:] = 0.0

    return sensitivity_map

# -----------------------
# Grad-CAM Utility
# -----------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.gradients = None
        self.activations = None
        self.fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self.bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        # grad_output[0] is dL/dActivation
        self.gradients = grad_output[0].detach()

    def generate(self):
        """
        Produce a normalized CAM in [0,1] with shape (H, W) of the target layer,
        later resized to 28x28 by caller.
        """
        # Global-average-pool the gradients to get channel weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # [B,C,1,1]
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [B,1,H,W]
        cam = F.relu(cam)

        cam = cam.squeeze().cpu().numpy()
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)
        return cam

    def close(self):
        self.fwd_hook.remove()
        self.bwd_hook.remove()

# -----------------------
# Prediction Function
# -----------------------
def predict(image_pil, from_canvas=True, Need_Features=False, Occlusion=False):
    """
    Args:
        image_pil: PIL.Image
        from_canvas: bool, True if coming from user canvas (black bg, white stroke)
    Returns:
        dict with strictly JSON-serializable fields:
          - predicted_class: int
          - confidence: float
          - probabilities: list[float] length 10
          - heatmap: str (base64 PNG)  # Grad-CAM colormap
          - feature_maps: {conv1: [b64...], conv2: [b64...]}  (only if Need_Features)
          - occlusion: str (base64 PNG) or None               (only if Occlusion)
    """
    model = load_model()

    # ---- Preprocess ----
    image_pil = image_pil.convert("L")
    if from_canvas:
        # Invert (canvas is black bg, white stroke) -> MNIST style (white on black)
        image_pil = ImageOps.invert(image_pil)
        # Slight blur to smooth strokes
        image_pil = image_pil.filter(ImageFilter.GaussianBlur(radius=1))

    input_tensor = transform(image_pil).unsqueeze(0).to(device)  # [1,1,28,28]

    # ---- Forward pass ----
    target_layer = model.conv1  # use last conv layer for CAM
    gradcam = GradCAM(model, target_layer)

    output = model(input_tensor)                   # [1,10]
    probs_t = F.softmax(output, dim=1)[0]          # [10]
    probs = probs_t.detach().cpu().numpy()
    pred_class = int(np.argmax(probs))
    confidence = float(np.max(probs))

    # ---- Grad-CAM ----
    model.zero_grad()
    output[0, pred_class].backward()
    cam = gradcam.generate()                       # (H,W) in [0,1]
    gradcam.close()

    # Prepare overlay image
    img = input_tensor.squeeze().detach().cpu().numpy()
    img = (img * 0.5) + 0.5   # denormalize
    img = np.clip(img, 0, 1)
    img = np.uint8(img * 255)

    # Resize CAM to 28x28 and colorize with JET
    cam_28 = cv2.resize(cam, (28, 28))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * cam_28), cv2.COLORMAP_JET)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    overlay = cv2.addWeighted(img_rgb, 0, heatmap_color, 0.5, 0)
    # heatmap_b64 = array_to_base64(overlay)

    # ---- Feature maps (optional) ----
    feature_maps = {}
    if Need_Features:
        with torch.no_grad():
            act1 = F.relu(model.conv1(input_tensor))   # [1,32,28,28]
            p1   = model.pool1(act1)                   # [1,32,14,14]
            act2 = F.relu(model.conv2(p1))             # [1,64,14,14]
        # Encode first 32 from each (you already used 32)
        feature_maps["conv1"] = [tensor_to_base64(act1[0, i]) for i in range(min(32, act1.shape[1]))]
        feature_maps["conv2"] = [tensor_to_base64(act2[0, i]) for i in range(min(32, act2.shape[1]))]

    # ---- Occlusion Sensitivity (optional) ----
    # occlusion_b64 = None
    # if Occlusion:
    #     occ_map = occlusion_sensitivity(model, input_tensor, pred_class)  # (28,28) in [0,1]
    #     occ_color = cv2.applyColorMap(np.uint8(255 * occ_map), cv2.COLORMAP_JET)
    #     overlay = cv2.addWeighted(img_rgb, 1, occ_color, 0.5, 0)
    #     occlusion_b64 = array_to_base64(overlay)

    # ---- Build JSON-safe response ----
    return {
        "predicted_class": pred_class,
        "confidence": float(confidence),
        "probabilities": [float(x) for x in probs.tolist()],  # ensure pure Python floats
        "heatmap": overlay,         
        "feature_maps": feature_maps,   # dict of lists of base64 strings
        # "occlusion": occlusion_b64,     # base64 PNG or None
    }
