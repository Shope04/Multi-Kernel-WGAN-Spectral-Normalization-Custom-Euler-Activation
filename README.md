# Multi-Kernel-WGAN-Spectral-Normalization-Custom-Euler-Activation

Generative Adversarial Network with Custom Activations
Overview

This repository contains an implementation of a Generative Adversarial Network (GAN) featuring custom activation functions. The custom activation functions are designed to provide more flexibility and control over the model's learning behavior. This codebase is suitable for image generation tasks and uses PyTorch as the primary deep learning framework.
Features

    Custom Activation Functions for both Generator and Discriminator
    Advanced training loop with Wasserstein Loss
    Support for both GPU and CPU training
    Modular design for easy experimentation

bash

    Clone the Repository

    bash

git clone https://github.com/your_username/your_repository_name.git

Set Data Path

Update the dataroot in the config dictionary to point to the directory where your training dataset resides.

Train the Model

Simply run the script:

bash

    python your_script_name.py

    This will start the training process. The generated images and model checkpoints will be saved in the specified directories.

    Monitor Training

    The script will periodically output the Wasserstein loss for both the generator and discriminator. Generated images will be saved to the trainpics directory.

License

This project is licensed under the MIT License - see the LICENSE file for details.
