# Latent Activation

This set of nodes will apply various activation functions directly to latent tensors. Have fun and experiment with different non-linear transformations of the latent space!

## Activation Functions

The following activation functions are included in this pack:

- **ReLU:**  
  \[
  \text{ReLU}(x) = \max(0, x)
  \]

- **Sigmoid:**  
  \[
  \text{Sigmoid}(x) = \frac{1}{1 + e^{-x}}
  \]

- **Tanh:**  
  \[
  \text{Tanh}(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
  \]

- **Leaky ReLU:**  
  \[
  \text{Leaky ReLU}(x) = \max(0.01x, x)
  \]

- **ELU:**  
  \[
  \text{ELU}(x) = 
  \begin{cases} 
  x & \text{if } x > 0 \\
  \alpha (e^x - 1) & \text{if } x \leq 0 
  \end{cases}
  \]

- **Softplus:**  
  \[
  \text{Softplus}(x) = \log(1 + e^x)
  \]

- **Swish:**  
  \[
  \text{Swish}(x) = x \cdot \text{Sigmoid}(x)
  \]

- **GELU:**  
  \[
  \text{GELU}(x) = 0.5x(1 + \text{tanh}(\sqrt{\frac{2}{\pi}} (x + 0.044715x^3)))
  \]

- **SELU:**  
  \[
  \text{SELU}(x) = \lambda 
  \begin{cases} 
  x & \text{if } x > 0 \\
  \alpha (e^x - 1) & \text{if } x \leq 0 
  \end{cases}
  \]
  where \(\lambda \approx 1.0507\) and \(\alpha \approx 1.67326\).

- **Mish:**  
  \[
  \text{Mish}(x) = x \cdot \text{tanh}(\text{Softplus}(x))
  \]

- **PReLU:**  
  \[
  \text{PReLU}(x) = 
  \begin{cases} 
  x & \text{if } x > 0 \\
  ax & \text{if } x \leq 0 
  \end{cases}
  \]

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

Contributions are welcome! Feel free to submit issues, fork the repository and create pull requests.

## License

This project is licensed under the [MIT License](LICENSE).
