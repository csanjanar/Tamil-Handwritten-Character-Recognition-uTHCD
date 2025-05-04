# 📝 Tamil Handwritten Character Recognition (THCR) Optimization using CNNs and uTHCD

    Final-Year Undergraduate Project
`CNNs · Data Augmentation · Multi-Class Classification · Model Evaluation · Benchmarking`

This project investigates the optimization of Convolutional Neural Networks (CNNs) for **Tamil handwritten character recognition**, focusing on the **Unconstrained Tamil Handwritten Character Database (uTHCD)**.

[faizalhajamohideen/uthcdtamil-handwritten-database](https://www.kaggle.com/datasets/faizalhajamohideen/uthcdtamil-handwritten-database)
> ![Kaggle](https://img.shields.io/badge/Kaggle-035a7d?style=for-the-badge&logo=kaggle&logoColor=white)

![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?logo=TensorFlow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?logo=Keras&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?logo=scikit-learn&logoColor=white) 
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?logo=opencv&logoColor=white)

---

## 📚 Table of Contents

1. [🔍 Overview](#-overview)  
2. [📂 Project Structure](#-project-structure)  
3. [📌 Project Motivation](#-project-motivation)  
4. [🗃️ Dataset: uTHCD](#️-dataset-uthcd)  
5. [🗝️ Key Project Features](#️-key-project-features)  
6. [💠 Model Architecture](#-model-architecture)  
7. [📊 Experimental Results](#-experimental-results)  
    - [🔁 Augmentation Experiments](#-key-augmentation-experiments)
    - [🏗️ Architectural Experiments](#-key-architectural-experiments)
    - [🕸️ CNN Configuration Variants](#-cnn-configuration-variants)
    - [📉 LR Scheduling & Early Stopping](#-learning-rate-scheduling--early-stopping)
8. [📈 Final Results](#-final-results)  
9. [📥 Installation](#-installation)  
10. [⚙️ Running the Notebook](#-running-the-notebook)  
11. [📘 Final Report](#-final-report)  
12. [✍️ Author](#-author)  
13. [🙏 Acknowledgements](#-acknowledgements)

---

## 🔍 Overview  

While Tamil OCR research has historically lacked standardized datasets and reproducible benchmarks, this project addresses the gap by:
- Building upon the **benchmark CNN model proposed in the uTHCD paper**
- Applying architectural and regularization enhancements
- Leveraging **data augmentation** to increase generalizability
- Benchmarking model performance across 156 classes using structured experimentation

---

## 📂 Project Structure

```
THCR_Project/
├── notebooks/
│   └── THCR_CNN.ipynb         # Main model training & evaluation notebook
├── utils/
│   └── data_utils_4.py        # Data loading, preprocessing, augmentation functions
├── data/
│   └── tuning_results.csv     # Experiment tracking table
├── report/
│   └── final_report.pdf       # Methodology, results, analysis
├── requirements.txt
└── README.md
```

---

## 📌 Project Motivation

Tamil's script, consisting of vowels, consonants, and compound characters, presents unique challenges in handwritten character recognition (HCR). This project aims to:

- Improve benchmark performance on the **uTHCD dataset**, addressing the lack of research on diverse, standardized handwriting datasets in Tamil OCR
- Contribute structured experiments and insights for reproducible research

---

## 🗃️ Dataset: uTHCD
- Source: [Kaggle | uTHCD-Unconstrained Tamil Handwritten Database](https://www.kaggle.com/datasets/faizalhajamohideen/uthcdtamil-handwritten-database)
  
| Format    | Grayscale image size | Classes | Train-Test Split | Train-Val Split | Advantages over prior datasets |
| -------- | ------- | ------- | ------- | ------- | ------- |
| HDF5 (`.h5`)  | 64×64    | 156 Tamil characters    | 70:30    | 7:1    | <ul><li>Higher sample diversity</li><li>Balanced classes</li><li>Clean annotations</li></ul>|

- **Samples**: As originally extracted from dataset  
```
Train Set| X: (62870, 64, 64), Y: (62870,)
Test Set | X: (28080, 64, 64),  Y: (28080,)
Train-Validation Split ===================
Training data    | X: (55000, 64, 64), Y: (55000,)
Validation data  | X: (7870, 64, 64),   Y: (7870,)
```

### *️⃣ Unconstrained data
<mark>The Advantage of uTHCD</mark>: **Offline vs. Online Samples**

> *uTHCD includes both **offline** and **online** samples, offering richer handwriting diversity compared to the HPL dataset, which contains only online input.*
>
> ![Offline vs. Online Samples](https://github.com/user-attachments/assets/a216867a-6193-4721-a71d-d6baed39eec7)
> - **Offline samples** (left): Capture natural variations like stroke thickness, pressure, and handwriting artifacts — essential for realistic OCR training.
> - **Online samples** (right): Stylus-based inputs that are uniform and sparse, limiting feature richness for CNNs.


---

## 🗝️ Key Project Features

#### 🔸 CNN Model Development
  - Optimization of the uTHCD benchmark architecture  
  - Hyperparameter tuning (dropout, batch normalization, early stopping, LR scheduling)

#### 🔸 Data Augmentation and Preprocessing for <mark>Enhanced Training Data</mark>
- **Diversity**: by augmenting images with random scaling, rotation, affine transformations, morphological operations, noise and blur effects, and brightness/contrast adjustments. 
  - Augmentation techniques were carefully selected and tuned to preserve Tamil character integrity while increasing variability.
  
> *Custom augmentation pipeline developed to retain character legibility while simulating natural handwriting distortions.*

<p align="center">
  <img src="https://github.com/user-attachments/assets/ab677a30-cf25-40f4-8a50-6e219141a526" alt="Augmented sample comparison 1" width="500"/>
</p>

> *Identifying techniques that preserve visual integrity of Tamil characters*

<p align="center">
  <img src="https://github.com/user-attachments/assets/0618b0e1-12aa-4ac4-80d5-a80402eb7129" alt="Augmented sample comparison 2" width="500"/>
</p>

- **Size**: *Offline augmentation* was applied to expand the training set. Full and random augmentation strategies were used to reach the desired sample diversity.
> *With an augmentation factor of 0.1:* 
> - *Refer to the Report for details on determining sample size based on the augmentation factor*
> >    
  ```
  Full Augmentation:     100%|██████████| 5500/5500 
  Random Augmentation:   100%|██████████| 787/787 
  ==> Total augmented samples (Factor 0.1): 6287
  Total augmented data shapes | X: (6287, 64, 64), Y: (6287,)
  ```
    
#### 🔸 Efficient Training
  - Executed on Colab with A100 / T4 GPUs for high throughput
#### 🔸 Evaluation & Experiment Tracking
  - Precision, Recall, F1-Score for multi-class classification  
  - Tracking via `tuning_results.csv` + TensorBoard
#### 🔸 Benchmarking
  - Compared model performance to uTHCD and HPL benchmarks  
  - Tracked variations across architectural changes
#### 🔸 Reproducible Results
  - All code, configurations, and metrics are documented for repeatable benchmarking

---

## 💠 Model Architecture

Final architecture used in benchmarking:

```
Input (64x64x1)
→ Conv2D(64) → BatchNorm → ReLU → Dropout(0.1) → MaxPooling
→ Conv2D(64) → ReLU → Dropout(0.05) → MaxPooling
→ Flatten
→ Dense(1024) → Dropout(0.5)
→ Dense(512) → Dropout(0.5)
→ Output(156) → Softmax
```

> 💡 *Batch normalization was applied selectively after the first Conv layer, based on tuning outcomes.*

---

## 📊 Experimental Results

Performance across all experiments was tracked using [`tuning_results.csv`](./data/tuning_results.csv) & [`model_comparison.csv`](./data/model_comparison.csv) covering variations in:

- **Data Augmentation Strategies**
- **CNN Architecture & Dropout**
- **Batch Normalization Placement**
- **Learning Rate Scheduling**
- **Early Stopping Criteria**
  
> 📎 *The following experiment sets collectively showcase how targeted adjustments in augmentation, normalization, and training policies led to a highly generalized and accurate model for Tamil handwritten character recognition.*

<details>
<summary>🔁 <strong>Key Augmentation Experiments</strong></summary>

| **Model** | **Test Accuracy** | **Strategy** | **Remarks** |
|-----------|-------------------|--------------|-------------|
| Model1    | 90.59%            | Reduced augmentation                  | Baseline experiment |
| Model2    | 91.55%            | + Shear, Brightness/Contrast          | Noticeable improvement |
| Model6    | 92.98%            | + Morphological transforms (erosion/dilation) | Best result pre-architecture tuning |
| Model7    | 92.27%            | + Black/White noise                   | Performance declined due to feature confusion |

> ✳️ *Morphological transformations introduced meaningful variability, while noise-based augmentations degraded accuracy by resembling diacritics.*

</details>

---

<details>
<summary>🏗️ <strong>Key Architectural Experiments</strong></summary>

| **Model** | **Test Accuracy** | **Changes Made** | **Training Accuracy** |
|-----------|-------------------|------------------|------------------------|
| Model9    | 93.53%            | Added BatchNorm after conv layer 1 & 2 | 99.88% |
| Model10   | 82.94%            | Introduced Global Average Pooling       | 94.57% |
| Model14   | 93.92%            | BatchNorm after first conv + LR scheduling | 99.94% |
| Model17   | 93.17%            | BatchNorm after both conv layers        | 99.83% |

> 🧠 *Batch normalization placement significantly impacted generalization. Global Average Pooling degraded performance due to early spatial compression.*

</details>

---

<details>
<summary>🕸️ <strong>CNN Configuration Variants</strong></summary>

| **Configuration** | **Test Accuracy** | **Architecture** | **Dropout Settings** |
|-------------------|-------------------|------------------|----------------------|
| Basic             | 87.00%            | 64C2-MP2-64C2-MP2-1024N-512N | None |
| Basic + Dropout   | 91.10%            | Same as above     | Conv(0.1, 0.05), Dense(0.5) |
| Benchmark (uTHCD) | 93.16%            | Same              | Dropout + Aug ×3 |
| 2.22_Benchmark    | 89.06%            | 64C3-MP2-64C3-MP2-1024N-512N | Dropout + Aug ×3 |

> 📍 *The official uTHCD benchmark architecture with dropout and augmentation proved highly effective. Slight changes (e.g., kernel size from 2 to 3) affected generalization.*

</details>

---

<details>
<summary>📉 <strong>Learning Rate Scheduling & Early Stopping</strong></summary>

| **Model** | **Test Accuracy** | **EarlyStopping Patience** | **LR Reduction Patience** | **LR Factor** |
|-----------|-------------------|-----------------------------|----------------------------|---------------|
| Model7    | 92.27%            | 20                          | 0                          | —             |
| Model12   | 93.78%            | 15                          | 10                         | 0.1           |
| Model13   | 92.90%            | 10                          | 5                          | 0.1           |
| **Model14** | **93.92%**        | **15**                        | **8**                        | **0.1**         |

> ✅ *Model14 achieved the highest performance by combining tuned LR scheduling with optimal batch normalization and augmentation.*

</details>

---

### 📈 Final Results 

A comparison between the official uTHCD benchmark and the enhanced model developed in this project, using refined augmentation, configured early stopping, batch normalization, and learning rate scheduling.

#### 📋 Summary Table
> Results highlighting our improved model performance compared to the benchmark.
>
| Source | Model | Test Accuracy (%) | Train Accuracy (%) | Validation Accuracy (%) | Sensitivity | F1-Score |
|:------:|:------|:-----------------:|:------------------:|:------------------------:|:-----------:|:--------:|
| uTHCD | Final Benchmark | 93.16 | 95.76 | 98.35 | 0.9315 | 0.931 |
| Reproduced | Baseline Reproduction | 90.18 | 97.75 | 93.86 | - | - |
| Ours | Optimized (Augmentation only) | 92.98 | 99.09 | 95.69 | 0.9298 | 0.9296 |
| Ours | **Optimzized (BatchNorm and LR Schedule)** | **93.92** | **99.94** | **96.91** | **0.9392** | **0.9392** |

#### 📖 Highlights & Evaluation
> 💡 **Insights:**   
- Reproducing the uTHCD benchmark did **not fully match** the reported performance, likely due to augmentation and training parameters not being explicitly stated in the original paper.

> ⭐ **Impact of Augmentation Techniques**
- **Morphological transformations** (e.g., erosion, dilation) introduced realistic variability, improving generalization and reducing overfitting.
- In contrast, **noise-based augmentations** (e.g., Gaussian noise, blur) often produced artifacts resembling diacritical marks or stroke distortions, which misled the classifier and degraded performance.

> ⭐ **Best Performing Hyperparameters**
- **Batch Normalization**: Applied **after the first convolutional layer** helped stabilize training and improve generalization.
- **Early Stopping**: A patience of **15 epochs** provided the best trade-off between training time and validation stability.
- **Learning Rate Schedule**: `ReduceLROnPlateau` with a **factor of 0.1** and **patience of 8 epochs**, down to a **min LR of 1e-6**, significantly improved model convergence.

> ⭐ **Optimized Model Performance**
- Achieved an **F1-Score of 0.9392**, outperforming both the reproduced and original benchmark.
- Indicates a strong **precision-recall balance**, and better handling of false positives/negatives across all 156 classes.

> 📎 See [tuning_results.csv](./data/tuning_results.csv) for full experiment comparisons.

#### 🖼️ Learning Curves 

![Training curve](https://github.com/user-attachments/assets/8c327246-76f8-4c92-b81f-ecf6fb39cee1)
> Model Accuracy and Loss Over Epochs with Batch Norm in first convolutional layer only and Learning Rate Schedule

---

## 📥 Installation

```bash
git clone https://github.com/yourusername/tamil-handwritten-cnn.git
cd tamil-handwritten-cnn
pip install -r requirements.txt
```

---

## ⚙️ Running the Notebook

> Developed in **Google Colab** with GPU support

1. Open `notebooks/THCR_CNN.ipynb` in Colab  
2. Mount Google Drive (as shown in notebook)  
3. Ensure the following files are accessible:  
   - `hdf5_uTHCD_compressed.h5`  
   - `data_utils_4.py`  
   - `tuning_results.csv`

Drive:
> 📎 Ensure dataset and utility files are placed in the same Colab directory or mounted via Google Drive as shown in the notebook.

---

## 📘 Final Report

See [`report/final_report.pdf`](./report/final_report.pdf) for a full description of:
- Problem statement
- Dataset analysis
- Model design
- Experiments & results
- Conclusion and future work

---

## ✍️ Author

**C Sanjana Rajendran**  
B.Sc. Computer Science, *University of London (SIM GE)*  
[GitHub](https://github.com/csanjanar) • [LinkedIn](https://linkedin.com/in/c-sanjana-rajendran)

---

## 🙏 Acknowledgements

- Mr. Don Wee (Project Supervisor)  
- N. Shaffi and F. Hajamohideen for the uTHCD Dataset  
  > N. Shaffi and F. Hajamohideen, "uTHCD: A New Benchmarking for Tamil Handwritten OCR," *IEEE Access*, vol. 9, pp. 101469–101493, 2021.  
  > DOI: [10.1109/ACCESS.2021.3096823](https://doi.org/10.1109/ACCESS.2021.3096823)

---


