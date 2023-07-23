# Adversarial Face Detection and Defense

This Git repository contains code implementations related to Adversarial Face Detection and Defense. The repository is organized into three main folders:

## 1. Face Detection (YOLOv5 on Human Face Dataset)

The 'face_detection' folder contains the implementation of YOLOv5 for face detection on the Human Face dataset. The folder includes the necessary scripts, configuration files, and dataset loading code to train and evaluate the YOLOv5 model on the Wider Face dataset.

### Usage
- Detailed usage instructions and setup guide can be found in the 'face_detection' folder.

## 2. Adversarial Attack

The 'AdversarialAttack' folder contains code related to adversarial attacks on deep learning models, particularly face detection models. Various attack techniques like FGSM,  Adding noise to the face after face detection are implemented to generate adversarial examples for the face detection model.
### 2.1  Adding noise to the face after face detection
In the folder AttacksAdversarial,  put the image in the folder data/inference_image and run !python3 inference.py 
### 2.2  FGSM 
In the folder AttacksAdversarial , run the file fgsm-attack.ipynb

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

![alt text](https://github.com/quangbinh113/CV20222_Adversarial_Attack/blob/main/AdversarialDefense/models/visualize.png?raw=true)
## 4. Graphical User Interface
You can run the GUI.ipynb for GUI 


### Usage
- Detailed usage instructions and setup guides for each defense technique can be found in the respective subfolders.

## License

[GPL-3.0 License](LICENSE)

## Contributors

- [Truong Quang Binh](https://github.com/quangbinh113)
- [Hoang Xuan Viet](https://github.com/collaborator1)
- [Tran Cat Khanh](https://github.com/khanhha1005)
- [Le Tran Thang](https://github.com/thang662)
- [Nguyen Thien Hoan](https://github.com/collaborator2)

