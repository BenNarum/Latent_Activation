# Latent Activation

This set of nodes will apply various activation functions directly to latent tensors. Have fun and experiment with different non-linear transformations of the latent space!

## Activation Functions

The following activation functions are included in this pack:

- **ReLU:**  
  <p align="center"><code>ReLU(x) = max(0, x)</code></p>

- **Sigmoid:**  
  <p align="center"><code>Sigmoid(x) = 1 / (1 + e<sup>-x</sup>)</code></p>

- **Tanh:**  
  <p align="center"><code>Tanh(x) = (e<sup>x</sup> - e<sup>-x</sup>) / (e<sup>x</sup> + e<sup>-x</sup>)</code></p>

- **Leaky ReLU:**  
  <p align="center"><code>Leaky ReLU(x) = max(0.01x, x)</code></p>

- **ELU:**  
  <p align="center"><code>ELU(x) = x</code> if <code>x > 0</code>, else <code>ELU(x) = α (e<sup>x</sup> - 1)</code></p>

- **Softplus:**  
  <p align="center"><code>Softplus(x) = log(1 + e<sup>x</sup>)</code></p>

- **Swish:**  
  <p align="center"><code>Swish(x) = x * Sigmoid(x)</code></p>

- **GELU:**  
  <p align="center"><code>GELU(x) = 0.5x(1 + tanh(√(2/π) (x + 0.044715x<sup>3</sup>)))</code></p>

- **SELU:**  
  <p align="center"><code>SELU(x) = λ x</code> if <code>x > 0</code>, else <code>SELU(x) = λ α (e<sup>x</sup> - 1)</code></p>
  <p align="center">where <code>λ ≈ 1.0507</code> and <code>α ≈ 1.67326</code>.</p>

- **Mish:**  
  <p align="center"><code>Mish(x) = x * tanh(Softplus(x))</code></p>

- **PReLU:**  
  <p align="center"><code>PReLU(x) = x</code> if <code>x > 0</code>, else <code>PReLU(x) = ax</code></p>

## Installation

1. Clone or download this repository to your `ComfyUI/custom_nodes` directory.

## Usage

1. In your ComfyUI workflow, add one of the activation nodes (e.g., `ReLU Activation`) after a node that outputs a latent tensor (such as `KSampler` or `LoadLatent`).
2. Connect the output of the activation node to a `VAEDecode` node to generate an image from the transformed latent.
3. Adjust the following parameters to control the effect of the activation:
    - **Strength:**  Determines the intensity of the activation function (how much of the transformed latent is mixed with the original).
    - **Add to Original:** If enabled, the activated latent will be added to the original latent. If disabled, the original latent will be replaced.
    - **Normalize:** If enabled, the transformed latent will be normalized to have zero mean and unit variance.
    - **Clamp:** If enabled, the values of the transformed latent will be clamped within a specified range (`Clamp Min` and `Clamp Max`).
    - **Composite:** If enabled, the activated latent will be composited (blended) with the upscaled original latent.
    - **Blend Amount:**  Controls the blending ratio during compositing (0.0 to 1.0).
    - **Additional Parameters:** Certain activation functions have additional parameters (e.g., `alpha` for ELU, `beta` and `threshold` for Softplus, `negative_slope` for Leaky ReLU, etc.).

## Contributing

Contributions are welcome! Feel free to submit issues, fork the repository, and create pull requests.

## License

This project is licensed under the [MIT License](LICENSE).
