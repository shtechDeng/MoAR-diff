# MoAR-diff :  Controlled Diffusion Model for Infant Brain MR Motion Artifact Reduction
An unconditional diffusion pre-training model with an optional fine-tuning ControlNet model

Brain Magnetic Resonance Imaging (MRI) provides high-resolution structural details for neuroscience studies but is often affected by motion artifacts, especially in children. These artifacts substantially compromise the imaging quality and the wider utility of MRI. The efficacy of existing deep learning-based head motion artifacts eliminating approaches is often curtailed by the scarcity of paired data. To address this, we propose a novel Motion Artifact Reduction Diffusion model (MoAR-Diff). Our model allows to be pre-trained on images without head motion artifacts, and fine-tuned using a smaller set of paired data. In the inference phase, we use a smartly designed denoising strategy and a frequency-domain consistency constraint to reduce artifacts while preserving the structural integrity and fidelity of the denoised images. Additionally, by incorporating age features from a pre-trained text encoder, the model has the flexibility to handle rapidly developing brain for infants from 0-6 years old. This approach ensures the accuracy and robustness of the model across different age groups, surpassing the performance of both popular unsupervised and supervised methods.

![The overall architecture of the MoAR-Diff model ](https://github.com/shtechDeng/MoAR-diff/blob/main/UnDPM_finetune/models/model_final.png?raw=true)

Fig 1. (a) In the training phase, A U-Net is trained on images without artifacts to predict the Gaussian noise epsilon added at each forward stage, based on different time steps N. To enable supervised fine-tuning, we lock the original U-Net block and create a trainable copy of the encoder, connecting it to zero-convolution layers. (b) In the inference phase, We apply forward diffusion to the image x_wA, adding Gaussian noise to obscure the motion artifacts. We predict the corresponding noise epsilon for each time step to obtain the denoised result x_0|N at the current time step. We then perform a data consistency step in the frequency domain using the fourier transform (fft) of the image x_0|N and the original image x_wA, followed by adding the Gaussian noise for the next time step.


