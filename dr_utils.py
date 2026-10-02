"""
dr_utils.py - shared code for the Diabetic Retinopathy project.

The SAME file is used by the training notebook and by the web prototype, so the
image preprocessing at inference time is identical to training (no train/serve skew).

Contents
  1. Preprocessing  : crop black borders -> pad to square -> resize -> CLAHE -> unsharp mask
  2. Models         : EfficientNet-B3 / ResNet50 with a new 5-class head (transfer learning)
  3. Grad-CAM       : heat-map showing which retinal regions drove the prediction
  4. Checkpoints    : save / load a self-describing checkpoint
"""
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

CLASS_NAMES = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]   # ImageNet statistics
EVAL_TF = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


# --------------------------------------------------------------------------- #
# 1. PREPROCESSING  (all functions take / return RGB uint8 numpy arrays)
# --------------------------------------------------------------------------- #
def crop_black_borders(img, tol=7):
    """Remove the black background around the circular fundus region."""
    mask = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) > tol
    if not mask.any():                       # image too dark - return unchanged
        return img
    return img[mask.any(1)][:, mask.any(0)]


def pad_to_square(img):
    """Pad with black so resizing keeps the true aspect ratio (no distortion)."""
    h, w = img.shape[:2]
    s = max(h, w)
    out = np.zeros((s, s, 3), dtype=img.dtype)
    y, x = (s - h) // 2, (s - w) // 2
    out[y:y + h, x:x + w] = img
    return out


def apply_clahe(img, clip=2.0, tile=8):
    """Contrast Limited Adaptive Histogram Equalisation on the LAB lightness channel."""
    l, a, b = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2LAB))
    l = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)


def unsharp_mask(img, sigma=2.0, amount=0.5):
    """Edge enhancement: img + amount * (img - blurred). Sharpens vessels / lesions."""
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 1 + amount, blur, -amount, 0)


def ben_graham(img, sigma=10):
    """Local-average colour subtraction (Ben Graham, Kaggle DR 2015). Used for comparison only."""
    return cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), sigma), -4, 128)


def preprocess_fundus(img_rgb, size=300):
    """Full pipeline used for training AND inference."""
    img = crop_black_borders(img_rgb)
    img = pad_to_square(img)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    img = apply_clahe(img)
    return unsharp_mask(img)


# --------------------------------------------------------------------------- #
# 2. MODELS (transfer learning)
# --------------------------------------------------------------------------- #
def build_model(arch='efficientnet_b3', num_classes=5, pretrained=True):
    """ImageNet-pretrained backbone + new classification head."""
    if arch == 'efficientnet_b3':
        m = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT if pretrained else None)
        m.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(m.classifier[1].in_features, num_classes))
    elif arch == 'resnet50':
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        m.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(m.fc.in_features, num_classes))
    else:
        raise ValueError(f'Unknown architecture: {arch}')
    return m


def get_head(model, arch):
    return model.classifier if arch.startswith('efficientnet') else model.fc


def set_backbone_trainable(model, arch, trainable):
    """trainable=False -> only the new head learns (feature extraction).
       trainable=True  -> whole network learns (fine-tuning)."""
    head_ids = {id(p) for p in get_head(model, arch).parameters()}
    for p in model.parameters():
        p.requires_grad = True if trainable else (id(p) in head_ids)


# --------------------------------------------------------------------------- #
# 3. GRAD-CAM
# --------------------------------------------------------------------------- #
def _target_layer(model, arch):
    return model.features[-1] if arch.startswith('efficientnet') else model.layer4[-1]


def gradcam(model, arch, x, class_idx=None):
    """x: (1,3,H,W) tensor. Returns (cam in [0,1] at feature resolution, class index, softmax probs)."""
    model.eval()
    acts, grads = [], []

    def fwd_hook(_, __, out):
        acts.append(out)
        out.register_hook(lambda g: grads.append(g))

    handle = _target_layer(model, arch).register_forward_hook(fwd_hook)
    try:
        with torch.enable_grad():
            logits = model(x)
            idx = int(logits.argmax(1)) if class_idx is None else int(class_idx)
            model.zero_grad()
            logits[0, idx].backward()
    finally:
        handle.remove()
    weights = grads[0].mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * acts[0].detach()).sum(1))[0].cpu().numpy()
    cam = cam / (cam.max() + 1e-8)
    probs = torch.softmax(logits.detach()[0], 0).cpu().numpy()
    return cam, idx, probs


def overlay_cam(img_rgb, cam, alpha=0.45):
    """Blend a Grad-CAM map (jet colours) on top of an RGB uint8 image."""
    h, w = img_rgb.shape[:2]
    heat = cv2.applyColorMap(np.uint8(255 * cv2.resize(cam, (w, h))), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img_rgb, 1 - alpha, heat, alpha, 0)


# --------------------------------------------------------------------------- #
# 4. CHECKPOINTS
# --------------------------------------------------------------------------- #
def save_checkpoint(path, model, arch, img_size, extra=None):
    ck = {'arch': arch, 'img_size': int(img_size), 'class_names': CLASS_NAMES,
          'state_dict': {k: v.cpu() for k, v in model.state_dict().items()}}
    ck.update(extra or {})
    torch.save(ck, path)


def load_checkpoint(path, device='cpu'):
    ck = torch.load(path, map_location=device)
    model = build_model(ck['arch'], len(ck['class_names']), pretrained=False)
    model.load_state_dict(ck['state_dict'])
    return model.to(device).eval(), ck
