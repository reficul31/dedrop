# DeDrop - Raindrop Removal and Generation using Generative Adversarial Network
Rain has a variety of visual impacts. It often causes dramatic fluctuations in the intensity of images and videos, which can seriously impede the performance of outdoor vision systems. Rain affects the general viewing field of all lenses adversely, thus increasing the complexity of modelling exigent Computer Vision focused tasks such as autonomous driving, object tracking, scene segmentation.Additionally, rain obfuscates vision systems in a variety of ways such as rain streaks and droplets possessing varying spatial-luminosity properties. Therefore, it is vital to tackle the problem of rain obstruction in images. The referenced paper provides a method for Single Image Rain Removal (SIRR) using a Generative Adversarial Network (GAN), but the paper only focuses on rainy images with obstructed backgrounds and not on images clicked by obfuscated camera lenses. Through this project, we attempt to explore the removal of raindrops from said images, extending the work of the original paper to de-noise images obstructed by droplets on camera lenses.

# Methods Used
In this project, we have 2 methods for Raindrop removal.
## Proposed Method
![Proposed](./assets/Proposed_new.png)
## VRGNet for Raindrop Removal
![VRGNet](./assets/VRGNet_new.png)

# Training
For plotting the loss we have used the Structural Similarity Index Metric (SSIM).
| Proposed | VRGNet |
| -------- | ------ |
| ![Proposed](./assets/vae_gan_loss.png)| ![Baseline](./assets/baseline_loss.png)|

# Results
## Abalation Study
![Abalation](./assets/abalation.png)

## Quantative Results
### PRNet Outputs
![PRNet](./assets/PRNet_output.png)

### VAE-GAN Output (Proposed)
![VAE-GAN](./assets/vae_gan_output.png)

### VRGNet Output
![VRGNet](./assets/VRGNet_output.png)