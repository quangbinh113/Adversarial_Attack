# Adversarial Face Detection and Defense

This Git repository contains code implementations related to Adversarial Face Detection and Defense. The repository is organized into three main folders:

## 1. Face Detection (YOLOv5 on Wider Face Dataset)

The 'face_detection' folder contains the implementation of YOLOv5 for face detection on the Wider Face dataset. The folder includes the necessary scripts, configuration files, and dataset loading code to train and evaluate the YOLOv5 model on the Wider Face dataset.

### Usage
- Detailed usage instructions and setup guide can be found in the 'face_detection' folder.

## 2. Adversarial Attack

The 'AdversarialAttack' folder contains code related to adversarial attacks on deep learning models, particularly face detection models. Various attack techniques like FGSM, PGD, and CW are implemented to generate adversarial examples for the face detection model.

### Usage
- Detailed usage instructions and setup guide can be found in the 'AdversarialAttack' folder.

## 3. Adversarial Defense

The 'AdversarialDefense' folder contains code implementations of adversarial defense techniques for robust face detection.

### 3.1 Adversarial Training

In the 'AdversarialDefense/AdversarialTraining' subfolder, you'll find the implementation of adversarial training methods to enhance the robustness of the face detection model against adversarial attacks.

### 3.2 Denoising

The 'AdversarialDefense/Denoising' subfolder contains implementations of denoising techniques to remove noise from adversarial examples before passing them to the face detection model.

#### 3.2.1 Denoising using Filters

In the 'AdversarialDefense/Denoising/DenoisingUsingFilters' subfolder, you'll find implementations of various filtering techniques (e.g., Gaussian, Median) to denoise adversarial examples.

#### 3.2.2 Denoising using SRGAN

In the 'AdversarialDefense/Denoising/DenoisingUsingSRGAN' subfolder, you'll find the implementation of denoising adversarial examples using SRGAN (Super-Resolution Generative Adversarial Network).

### Usage
- Detailed usage instructions and setup guides for each defense technique can be found in the respective subfolders.

## License

[MIT License](LICENSE)

## Contributors

- [Truong Quang Binh](https://github.com/quangbinh113)
- [Hoang Xuan Viet](https://github.com/collaborator1)
- [Tran Cat Khanh](https://github.com/collaborator2)
- [Le Tran Thang](https://github.com/thang662)
- [Nguyen Thien Hoan](https://github.com/collaborator2)

