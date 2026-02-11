# Cephalometric Landmark Detection - Geographic Bias Mitigation

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Research-green.svg)]()

> **Addressing geographic bias in AI-powered dental diagnostics through population-specific deep learning**

Research collaboration with the University of the Western Cape Faculty of Dentistry to develop cephalometric landmark detection models optimized for South African patient populations.

---

## 📋 Project Overview

Existing commercial AI tools for cephalometric analysis (BoneFinder, DentaliQ, WebCeph) are predominantly trained on European and Asian datasets, achieving only **57-66% accuracy** for South African patients—a critical healthcare equity issue. This project addresses that gap.

**Key Achievement:** Developed DeepFuse model achieving **85.8% clinical deployment threshold** (first system to exceed clinical standards for SA patients) with **1.20mm Mean Radial Error**.

---

## 👥 Team & Collaboration

**My Role:** Analytics Translator  
**Research Partner:** University of the Western Cape Faculty of Dentistry  
**Clinical Supervisor:** Dr. Suvarna Indermun  
**Project Timeline:** September 2024 - January 2025

---

## 🎯 My Role: Analytics Translator

As Analytics Translator, I bridge the gap between technical ML development and clinical stakeholder needs:

- **Data Quality Leadership:** Led validation of 1,714 medical imaging samples achieving 99.8% Data Quality KPI
- **Model Development:** Built and trained VGG-16 baseline and DeepFuse architectures
- **Preprocessing Pipeline:** Designed TMJ detection, intelligent cropping, and histogram matching systems
- **Stakeholder Communication:** Translated clinical requirements into technical specs and model performance into clinical insights
- **Documentation:** Created Business Requirements Documents and technical documentation for healthcare AI

---

## 📊 Dataset Composition

**1,714 Total Radiographs from Four International Sources:**

| Dataset | Geographic Origin | Samples | Purpose |
|---------|------------------|---------|---------|
| Aariz | Unknown | 1,000 | Diversity baseline |
| PKU | Peking, China | 102 | International benchmark |
| ISBI 2015 | Taiwan | 400 | Kaggle challenge data |
| **UWC** | **South Africa** | **212** | **Target population** |

### Data Quality Metrics

| Quality Dimension | Score | Details |
|-------------------|-------|---------|
| **Completeness** | 100% | All 1,714 image-annotation pairs with 19 landmarks |
| **Consistency** | 100% | All coordinates within boundaries, anatomically plausible |
| **Validity** | 100% | All annotations within image dimensions |
| **Uniqueness** | 98.4% | 11 duplicates identified and removed |
| **Overall KPI** | **99.8%** | Production-ready dataset |

---

## 🧠 Model Architecture & Performance

### Evolution of Approaches

#### 1. Baseline: VGG-16 (External Data Only)
- **Training:** 1,502 external images (Aariz, PKU, ISBI 2015)
- **Performance on SA test:** 2.04mm MRE, 62.0% SDR@2mm
- **Finding:** Population bias evident—below 85% clinical threshold

#### 2. Failed Experiment: Fine-Tuning
- **Approach:** Transfer learning on 170 SA images
- **Result:** 2.41mm MRE, 37.8% SDR@2mm ❌ (39% degradation)
- **Root Cause:** Compression ratio mismatch (SA images 50% higher resolution)
- **Key Learning:** Transfer learning fails with fundamental domain differences

#### 3. SA-Only Model (From Scratch)
- **Training:** 212 South African radiographs only
- **Performance:** 1.46mm MRE, 81.58% SDR@2mm
- **Gap:** 3.42pp below clinical threshold (limited training data)

#### 4. **Production Model: DeepFuse Combined Dataset** ✅
- **Architecture:** ResNet-50 backbone + U-Net decoder with heatmap regression
- **Training:** All 1,714 images from scratch
- **Performance:**
  - **1.20mm MRE** (41.2% improvement over baseline)
  - **85.8% SDR@2mm** ✓ (Exceeds 85% clinical threshold)
  - **93.6% SDR@3mm**
  - **96.4% SDR@4mm**

### Model Performance Comparison

| Metric | Baseline | SA-Only | **Combined** | Clinical Threshold |
|--------|----------|---------|--------------|-------------------|
| MRE | 2.04mm | 1.46mm | **1.20mm** | <2mm ✓ |
| SDR@2mm | 62.0% | 81.58% | **85.8%** | ≥85% ✓ |
| SDR@3mm | 80.9% | 91.63% | **93.6%** | - |
| SDR@4mm | 87.6% | 95.45% | **96.4%** | - |

---

## 🔧 Technical Implementation

### Data Preprocessing Pipeline

1. **File Standardization**
   - Converted all datasets to JSON annotations and PNG images
   - Standardized naming: `{dataset}_{id}_cephalogram.png`

2. **Histogram Contrast Matching**
   - Normalized pixel distributions to reference (PKU '54.bmp')

3. **Automated TMJ Detection**
   - Binary thresholding (value=200) for anatomical anchor point
   - Center-based object selection with noise filtering

4. **Intelligent Cropping**
   - TMJ-based with 200px left margin
   - 40% width safety limit preventing over-cropping
   - 200px top removal preserving anatomical ROI

5. **Heatmap Generation**
   - 128×128 spatial heatmaps for model explainability
   - Images resized to 512×512 for training efficiency

### DeepFuse Architecture
```
Input: 512×512 grayscale radiograph
         ↓
ResNet-50 Encoder (pre-trained backbone)
         ↓
Multi-Scale Feature Fusion Module
         ↓
U-Net Decoder with Skip Connections
         ↓
19 Landmark Heatmaps (128×128)
         ↓
Soft-argmax Coordinate Extraction
         ↓
Output: 19 (x,y) landmark coordinates
```

**Key Innovations:**
- **Multi-scale fusion:** Combines local details + global structure
- **Heatmap regression:** Spatial reasoning vs. direct coordinates
- **Adaptive Wing Loss:** Robust to outliers

---

## 📈 Results & Clinical Significance

### Landmark-Specific Performance

**Best Performing:**
- Upper lip: 0.34mm
- Anterior nasal spine: 0.37mm
- Porion: 0.38mm

**Most Challenging:**
- Soft tissue pogonion: 1.50mm
- Orbitale: 1.12mm
- Sella: 0.89mm

### Clinical Deployment Readiness

✅ **Exceeds 85% SDR@2mm threshold**  
✅ **93.3% of images meet diagnostic standards**  
✅ **High model confidence (0.922 mean) with reliable calibration**  
⚠️ **Requires clinician oversight for final validation**

---

## 🎓 Analytics Translation Examples

### Stakeholder Communication

**Clinical Requirement → Technical Specification:**
> "We need reliable landmark detection for South African patients"  
→ "Dataset with 1,714 samples, 99.8% Data Quality KPI, <2mm MRE, ≥85% SDR@2mm, anatomical validation gates, demographic diversity"

**Technical Finding → Clinical Decision:**
> "Fine-tuning degraded 39% due to compression mismatch"  
→ "Recommendation: Train from scratch on combined data avoiding transfer learning assumptions—clinical impact: achieves 85.8% threshold vs. 37.8% with fine-tuning"

**Model Performance → Business Value:**
> "DeepFuse achieves 1.20mm MRE with 85.8% SDR@2mm"  
→ "First system meeting clinical standards for SA patients—ready for faculty validation study with oversight protocols"

---

## ⚠️ Current Limitations

**Model Constraints:**
- SA-only model 3.42pp below threshold due to limited data (212 vs. 1,714 images)
- Compression ratio heterogeneity across sources
- Higher variability in posterior soft tissue landmarks

**Data Constraints:**
- Limited SA training samples (212 images)
- Potential underrepresentation of full population diversity
- Pixel spacing variations (±20-30%) across datasets

---

## 🚀 Future Development

**Immediate:**
- Expand SA dataset to 400-500 images for SA-only model improvement
- Validate pixel spacing through calibration markers
- Multi-site data collection across SA hospitals

**Clinical Validation:**
- Validation studies with UWC dental faculty
- Real-world deployment in educational settings
- Integration with practice management systems

**Technical:**
- Ensemble methods for robustness
- Attention mechanisms for challenging landmarks
- Population-specific fine-tuning strategies

---

## 💡 Skills Demonstrated for AI Product Management

This project showcases critical AI Product Manager capabilities:

✓ **Model Development** - Led systematic architecture iterations identifying optimal approaches  
✓ **Data Quality Leadership** - 99.8% Data Quality KPI through comprehensive frameworks  
✓ **Root Cause Analysis** - Identified compression mismatch causing 39% degradation  
✓ **Healthcare Domain** - Clinical thresholds, anatomical validation, medical imaging  
✓ **Stakeholder Translation** - Technical metrics → clinical deployment criteria  
✓ **Strategic Decisions** - From-scratch training based on failure analysis  
✓ **Ethical AI** - Population bias mitigation for healthcare equity

---

## 🛠️ Technologies

**Deep Learning:** PyTorch, ResNet-50, U-Net, Heatmap Regression, Adaptive Wing Loss  
**Data Processing:** Python, Pandas, NumPy, OpenCV, PIL, JSON validation  
**Medical Imaging:** TMJ detection, Histogram matching, Intelligent cropping  
**Quality Assurance:** MD5 hashing, Anatomical validation gates, Statistical analysis

---

## 📞 Contact

**Analytics Translator:** Carmen-Nicola Ioniță  
**Email:** carmennikola@gmail.com  
**LinkedIn:** [linkedin.com/in/carmen-nicola-ionita](https://www.linkedin.com/in/carmen-nicola-ioni%C8%9B%C4%83-415b822a0/)  
**Portfolio:** [ionita-carmen-nicola-portofolio.lovable.app](https://ionita-carmen-nicola-portofolio.lovable.app/)

---

## 🏥 Ethics & Compliance

- ✅ GDPR-compliant anonymized data processing
- ✅ Designed to support, not replace, clinical judgment
- ✅ Requires qualified professional review
- ✅ 5-year secure data retention per research standards

---

## 🙏 Acknowledgments

**Clinical Partner:** University of the Western Cape Faculty of Dentistry  
**Supervisor:** Dr. Suvarna Indermun  
**Team:** Victor, Arnout, Yorbe, Jason  
**Dataset Contributors:** Aariz, PKU, ISBI 2015 Grand Challenge

---

**Status:** ✅ Completed (January 2025)  
**Clinical Impact:** First model exceeding deployment threshold for South African patients  
**Research Significance:** Demonstrated population-specific training mitigates AI bias in healthcare
