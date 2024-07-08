from .latent_blender import LatentBlender  # Import the LatentBlender class from latent_blender.py
from . import activators  # Import the activators.py file from this directory

# Define NODE_CLASS_MAPPINGS with the activation nodes
NODE_CLASS_MAPPINGS = {
    'ReLUActivation': activators.ReLUActivation,
    'SigmoidActivation': activators.SigmoidActivation,
    'TanhActivation': activators.TanhActivation,
    'LeakyReLUActivation': activators.LeakyReLUActivation,
    'ELUActivation': activators.ELUActivation,
    'SoftplusActivation': activators.SoftplusActivation,
    'SwishActivation': activators.SwishActivation,
    'PReLUActivation': activators.PReLUActivation,
    'GELUActivation': activators.GELUActivation,
    'SELUActivation': activators.SELUActivation,
    'MishActivation': activators.MishActivation,
}

# Define display name mappings for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    'ReLUActivation': 'ReLU Activation',
    'SigmoidActivation': 'Sigmoid Activation',
    'TanhActivation': 'Tanh Activation',
    'LeakyReLUActivation': 'Leaky ReLU Activation',
    'ELUActivation': 'ELU Activation',
    'SoftplusActivation': 'Softplus Activation',
    'SwishActivation': 'Swish Activation',
    'PReLUActivation': 'PReLU Activation',
    'GELUActivation': 'GELU Activation',
    'SELUActivation': 'SELU Activation',
    'MishActivation': 'Mish Activation',
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
