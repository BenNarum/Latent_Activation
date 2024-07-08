import torch
import torch.nn.functional as F
from .latent_blender import LatentBlender

class ReLUActivation:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "samples": ("LATENT",),
            "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            "add_to_original": ("BOOLEAN", {"default": True}),
            "normalize": ("BOOLEAN", {"default": False}),
            "clamp": ("BOOLEAN", {"default": False}),
            "clamp_min": ("FLOAT", {"default": -3.0, "min": -10.0, "max": 0.0, "step": 0.1}),
            "clamp_max": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 10.0, "step": 0.1}),
            "composite": ("BOOLEAN", {"default": False}),
            "blend_amount": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
        }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "apply_activation"
    CATEGORY = "latent/experimental"

    def apply_activation(self, samples, strength, add_to_original, normalize, clamp, clamp_min, clamp_max, composite, blend_amount):
        latent = samples["samples"]
        transformed_latent = torch.relu(latent)
        return LatentBlender.blend_latent(samples, latent, transformed_latent, strength, add_to_original, composite, blend_amount, normalize, clamp, clamp_min, clamp_max)

class SigmoidActivation:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "samples": ("LATENT",),
            "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            "add_to_original": ("BOOLEAN", {"default": True}),
            "normalize": ("BOOLEAN", {"default": False}),
            "clamp": ("BOOLEAN", {"default": False}),
            "clamp_min": ("FLOAT", {"default": -3.0, "min": -10.0, "max": 0.0, "step": 0.1}),
            "clamp_max": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 10.0, "step": 0.1}),
            "composite": ("BOOLEAN", {"default": False}),
            "blend_amount": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
        }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "apply_activation"
    CATEGORY = "latent/experimental"

    def apply_activation(self, samples, strength, add_to_original, normalize, clamp, clamp_min, clamp_max, composite, blend_amount):
        latent = samples["samples"]
        transformed_latent = torch.sigmoid(latent)
        return LatentBlender.blend_latent(samples, latent, transformed_latent, strength, add_to_original, composite, blend_amount, normalize, clamp, clamp_min, clamp_max)

class TanhActivation:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "samples": ("LATENT",),
            "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            "add_to_original": ("BOOLEAN", {"default": True}),
            "normalize": ("BOOLEAN", {"default": False}),
            "clamp": ("BOOLEAN", {"default": False}),
            "clamp_min": ("FLOAT", {"default": -3.0, "min": -10.0, "max": 0.0, "step": 0.1}),
            "clamp_max": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 10.0, "step": 0.1}),
            "composite": ("BOOLEAN", {"default": False}),
            "blend_amount": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
        }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "apply_activation"
    CATEGORY = "latent/experimental"

    def apply_activation(self, samples, strength, add_to_original, normalize, clamp, clamp_min, clamp_max, composite, blend_amount):
        latent = samples["samples"]
        transformed_latent = torch.tanh(latent)
        return LatentBlender.blend_latent(samples, latent, transformed_latent, strength, add_to_original, composite, blend_amount, normalize, clamp, clamp_min, clamp_max)

class LeakyReLUActivation:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "samples": ("LATENT",),
            "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            "negative_slope": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 10.0, "step": 0.01}),
            "add_to_original": ("BOOLEAN", {"default": True}),
            "normalize": ("BOOLEAN", {"default": False}),
            "clamp": ("BOOLEAN", {"default": False}),
            "clamp_min": ("FLOAT", {"default": -3.0, "min": -10.0, "max": 0.0, "step": 0.1}),
            "clamp_max": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 10.0, "step": 0.1}),
            "composite": ("BOOLEAN", {"default": False}),
            "blend_amount": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
        }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "apply_activation"
    CATEGORY = "latent/experimental"

    def apply_activation(self, samples, strength, negative_slope, add_to_original, normalize, clamp, clamp_min, clamp_max, composite, blend_amount):
        latent = samples["samples"]
        transformed_latent = F.leaky_relu(latent, negative_slope=negative_slope)
        return LatentBlender.blend_latent(samples, latent, transformed_latent, strength, add_to_original, composite, blend_amount, normalize, clamp, clamp_min, clamp_max)

class ELUActivation:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "samples": ("LATENT",),
            "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            "alpha": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            "add_to_original": ("BOOLEAN", {"default": True}),
            "normalize": ("BOOLEAN", {"default": False}),
            "clamp": ("BOOLEAN", {"default": False}),
            "clamp_min": ("FLOAT", {"default": -3.0, "min": -10.0, "max": 0.0, "step": 0.1}),
            "clamp_max": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 10.0, "step": 0.1}),
            "composite": ("BOOLEAN", {"default": False}),
            "blend_amount": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
        }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "apply_activation"
    CATEGORY = "latent/experimental"

    def apply_activation(self, samples, strength, alpha, add_to_original, normalize, clamp, clamp_min, clamp_max, composite, blend_amount):
        latent = samples["samples"]
        transformed_latent = F.elu(latent, alpha=alpha)
        return LatentBlender.blend_latent(samples, latent, transformed_latent, strength, add_to_original, composite, blend_amount, normalize, clamp, clamp_min, clamp_max)

class SoftplusActivation:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "samples": ("LATENT",),
            "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            "beta": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.01}),
            "threshold": ("FLOAT", {"default": 20.0, "min": 0.1, "max": 100.0, "step": 0.1}),
            "add_to_original": ("BOOLEAN", {"default": True}),
            "normalize": ("BOOLEAN", {"default": False}),
            "clamp": ("BOOLEAN", {"default": False}),
            "clamp_min": ("FLOAT", {"default": -3.0, "min": -10.0, "max": 0.0, "step": 0.1}),
            "clamp_max": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 10.0, "step": 0.1}),
            "composite": ("BOOLEAN", {"default": False}),
            "blend_amount": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
        }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "apply_activation"
    CATEGORY = "latent/experimental"

    def apply_activation(self, samples, strength, beta, threshold, add_to_original, normalize, clamp, clamp_min, clamp_max, composite, blend_amount):
        latent = samples["samples"]
        transformed_latent = F.softplus(latent, beta=beta, threshold=threshold)
        return LatentBlender.blend_latent(samples, latent, transformed_latent, strength, add_to_original, composite, blend_amount, normalize, clamp, clamp_min, clamp_max)

class SwishActivation:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "samples": ("LATENT",),
            "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            "beta": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.01}),
            "add_to_original": ("BOOLEAN", {"default": True}),
            "normalize": ("BOOLEAN", {"default": False}),
            "clamp": ("BOOLEAN", {"default": False}),
            "clamp_min": ("FLOAT", {"default": -3.0, "min": -10.0, "max": 0.0, "step": 0.1}),
            "clamp_max": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 10.0, "step": 0.1}),
            "composite": ("BOOLEAN", {"default": False}),
            "blend_amount": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
        }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "apply_activation"
    CATEGORY = "latent/experimental"

    def apply_activation(self, samples, strength, beta, add_to_original, normalize, clamp, clamp_min, clamp_max, composite, blend_amount):
        latent = samples["samples"]
        transformed_latent = latent * torch.sigmoid(beta * latent)
        return LatentBlender.blend_latent(samples, latent, transformed_latent, strength, add_to_original, composite, blend_amount, normalize, clamp, clamp_min, clamp_max)

class PReLUActivation:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "samples": ("LATENT",),
            "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            "init": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
            "add_to_original": ("BOOLEAN", {"default": True}),
            "normalize": ("BOOLEAN", {"default": False}),
            "clamp": ("BOOLEAN", {"default": False}),
            "clamp_min": ("FLOAT", {"default": -3.0, "min": -10.0, "max": 0.0, "step": 0.1}),
            "clamp_max": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 10.0, "step": 0.1}),
            "composite": ("BOOLEAN", {"default": False}),
            "blend_amount": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
        }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "apply_activation"
    CATEGORY = "latent/experimental"

    def apply_activation(self, samples, strength, init, add_to_original, normalize, clamp, clamp_min, clamp_max, composite, blend_amount):
        latent = samples["samples"]
        prelu = torch.nn.PReLU(init=init)
        transformed_latent = prelu(latent)
        return LatentBlender.blend_latent(samples, latent, transformed_latent, strength, add_to_original, composite, blend_amount, normalize, clamp, clamp_min, clamp_max)

class GELUActivation:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "samples": ("LATENT",),
            "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            "add_to_original": ("BOOLEAN", {"default": True}),
            "normalize": ("BOOLEAN", {"default": False}),
            "clamp": ("BOOLEAN", {"default": False}),
            "clamp_min": ("FLOAT", {"default": -3.0, "min": -10.0, "max": 0.0, "step": 0.1}),
            "clamp_max": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 10.0, "step": 0.1}),
            "composite": ("BOOLEAN", {"default": False}),
            "blend_amount": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
        }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "apply_activation"
    CATEGORY = "latent/experimental"

    def apply_activation(self, samples, strength, add_to_original, normalize, clamp, clamp_min, clamp_max, composite, blend_amount):
        latent = samples["samples"]
        transformed_latent = F.gelu(latent)
        return LatentBlender.blend_latent(samples, latent, transformed_latent, strength, add_to_original, composite, blend_amount, normalize, clamp, clamp_min, clamp_max)

class SELUActivation:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "samples": ("LATENT",),
            "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            "add_to_original": ("BOOLEAN", {"default": True}),
            "normalize": ("BOOLEAN", {"default": False}),
            "clamp": ("BOOLEAN", {"default": False}),
            "clamp_min": ("FLOAT", {"default": -3.0, "min": -10.0, "max": 0.0, "step": 0.1}),
            "clamp_max": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 10.0, "step": 0.1}),
            "composite": ("BOOLEAN", {"default": False}),
            "blend_amount": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
        }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "apply_activation"
    CATEGORY = "latent/experimental"

    def apply_activation(self, samples, strength, add_to_original, normalize, clamp, clamp_min, clamp_max, composite, blend_amount):
        latent = samples["samples"]
        transformed_latent = F.selu(latent)
        return LatentBlender.blend_latent(samples, latent, transformed_latent, strength, add_to_original, composite, blend_amount, normalize, clamp, clamp_min, clamp_max)

class MishActivation:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "samples": ("LATENT",),
            "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            "softplus_beta": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.01}),
            "softplus_threshold": ("FLOAT", {"default": 20.0, "min": 0.1, "max": 100.0, "step": 0.1}),
            "add_to_original": ("BOOLEAN", {"default": True}),
            "normalize": ("BOOLEAN", {"default": False}),
            "clamp": ("BOOLEAN", {"default": False}),
            "clamp_min": ("FLOAT", {"default": -3.0, "min": -10.0, "max": 0.0, "step": 0.1}),
            "clamp_max": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 10.0, "step": 0.1}),
            "composite": ("BOOLEAN", {"default": False}),
            "blend_amount": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
        }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "apply_activation"
    CATEGORY = "latent/experimental"

    def apply_activation(self, samples, strength, softplus_beta, softplus_threshold, add_to_original, normalize, clamp, clamp_min, clamp_max, composite, blend_amount):
        latent = samples["samples"]
        transformed_latent = latent * torch.tanh(F.softplus(latent, beta=softplus_beta, threshold=softplus_threshold))
        return LatentBlender.blend_latent(samples, latent, transformed_latent, strength, add_to_original, composite, blend_amount, normalize, clamp, clamp_min, clamp_max)
