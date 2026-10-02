import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._fh = target_layer.register_forward_hook(self._forward_hook)
        self._bh = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, target_index=None):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)
        if target_index is None:
            target_index = int(probs.argmax(dim=1).item())
        logits[:, target_index].sum().backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        denom = cam.max().clamp_min(1e-8)
        cam = cam / denom
        return cam.detach(), target_index, probs.detach()[0]

    def remove(self):
        self._fh.remove()
        self._bh.remove()


def perturb_raw(x, idx, noise_std=0.02, brightness_delta=0.05, contrast_delta=0.05):
    """Perturb a raw image tensor in [0,1] before normalization."""
    y = x.clone()
    mode = idx % 4
    if mode == 0:
        y = y + torch.randn_like(y) * noise_std
    elif mode == 1:
        y = y + brightness_delta
    elif mode == 2:
        mean = y.mean(dim=(2, 3), keepdim=True)
        y = (y - mean) * (1.0 + contrast_delta) + mean
    else:
        y = torch.roll(y, shifts=(2, -2), dims=(2, 3))
    return y.clamp(0.0, 1.0)


def cosine_map_similarity(a, b):
    av = a.reshape(-1).float()
    bv = b.reshape(-1).float()
    if float(av.norm()) < 1e-10 or float(bv.norm()) < 1e-10:
        return 0.0
    return float(F.cosine_similarity(av.unsqueeze(0), bv.unsqueeze(0)).item())


def saliency_stability(model, cam_engine, raw_x, base_cam, preprocess_fn, n_perturb=4,
                       noise_std=0.02, brightness_delta=0.05, contrast_delta=0.05):
    with torch.no_grad():
        target = int(model(preprocess_fn(raw_x)).argmax(1).item())
    sims = []
    for i in range(int(n_perturb)):
        xp_raw = perturb_raw(raw_x, i, noise_std, brightness_delta, contrast_delta)
        xp = preprocess_fn(xp_raw)
        cam_p, _, _ = cam_engine(xp, target_index=target)
        sims.append(cosine_map_similarity(base_cam, cam_p))
    if not sims:
        return 1.0
    return float(np.clip(np.mean(sims), 0.0, 1.0))
