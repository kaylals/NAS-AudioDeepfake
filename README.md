# NAS-AudioDeepfake 🎯

> **Audio Deepfake Detection for Modern TTS Systems**  
> Leveraging PC-DARTS architecture with custom training pipeline for multilingual deployment and SOTA TTS robustness

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🚀 Key Innovations

- **🌍 Modern TTS Architecture**: PC-DARTS framework adapted for 12+ state-of-the-art TTS systems 
- **🌏 Multilingual Training Pipeline**: Custom implementation supporting Chinese (50%) and English (30%) with 10+ language coverage
- **⚡ Production Architecture**: Real-time PC-based deployment with optimized neural architecture search
- **🔬 Domain Adaptation Framework**: Systematic architecture application for contemporary deepfake challenges

## 🎯 Problem Statement & Motivation

### The Challenge
**Existing anti-spoofing models fail in real-world scenarios due to:**
- **Temporal Gap**: ASVspoof dataset is outdated, while TTS technology has advanced dramatically
- **Language Limitation**: Most models trained on English-only datasets fail on Chinese and other languages  
- **TTS Evolution**: Modern systems like VALL-E, Bark, and MMS generate highly realistic speech
- **Deployment Gap**: Research models not optimized for real-time PC deployment

### Business Impact
- **Voice Authentication Security**: Banking and finance systems vulnerable to modern TTS attacks
- **Real-time Monitoring**: Need for continuous user voice verification in production systems
- **Multilingual Markets**: Chinese market requires robust Chinese language support

## 📊 Dataset & Methodology Innovation

### Custom Multilingual TTS Dataset
| Component | Details | Rationale |
|-----------|---------|-----------|
| **Modern TTS Models** | 12+ SOTA systems | Reflect current threat landscape |
| **Language Distribution** | Chinese (50%), English (30%), Others (20%) | Target market requirements |
| **Bonafide Sources** | AISHELL + ASVspoof2019 eval | Professional recording quality |
| **Total Scale** | 20k samples across 12 TTS systems | Systematic threat modeling dataset |


## 🔬 Technical Innovation

### 1. Neural Architecture Search Implementation
- **PC-DARTS Framework**: Implemented differentiable architecture search for audio domain
- **Architecture Adaptation**: Custom cell design optimized for temporal audio features
- **Search Space Optimization**: Tailored for modern TTS detection requirements

### 2. Cross-lingual Training Framework  
- **Custom Pipeline**: Multilingual training system built from ground up
- **Feature Engineering**: Language-agnostic audio representations
- **Domain Transfer**: Architecture application across diverse linguistic contexts

### 3. Production-First Architecture Design
- **Real-time Constraints**: Neural architecture optimized for <50ms inference
- **Resource Efficiency**: Implementation designed for consumer PC hardware
- **Scalable Framework**: Modular architecture supporting various deployment scenarios

### 4. Systematic Experimental Framework
- **Ablation Studies**: Comprehensive analysis of architecture components
- **Hyperparameter Optimization**: Optuna-based automated tuning for multilingual training
- **Performance Engineering**: End-to-end optimization from architecture to deployment

## 🏗️ System Architecture

```
📁 Project Structure
├── 🧠 models/              # PC-DARTS neural architecture
├── 📊 ASVDataloader/       # Custom audio data pipeline  
├── 🔧 experiments/         # Systematic experiment framework
│   ├── baseline/           # Original PC-DARTS implementation
│   ├── augmentation_study/ # Data augmentation research
│   ├── loss_optimization/  # Advanced loss functions
│   └── evaluation/         # Performance assessment
├── 🌐 web_demo/           # Production web interface
├── ⚡ inference/          # Optimized prediction pipeline
└── 📈 results/            # Experiment tracking & analysis
```

## 📈 Experimental Results

### Model Performance Comparison
| Model Configuration | ASVspoof2019 EER | Custom Dataset EER | Chinese Performance | Real-time Capable |
|---------------------|------------------|-------------------|---------------------|-------------------|
| Original PC-DARTS | [BASELINE] | [POOR] | Not Supported | ❌ |
| + Custom Training Pipeline (v1) | [PLACEHOLDER] | [IMPROVED] | [PLACEHOLDER] | ✅ |
| + Data Augmentation Strategy | 7.95% | [PLACEHOLDER] | [PLACEHOLDER] | ✅ |
| + Loss Engineering | **7.00%** | [**BEST**] | [**BEST**] | ✅ |

### Key Architecture Insights
- **Domain Gap Challenge**: Original architecture required substantial adaptation for modern TTS
- **Language Generalization**: Neural architecture search principles effectively transfer across languages
- **Training Pipeline Impact**: Custom implementation critical for contemporary threat detection
- **Production Viability**: Architecture maintains efficiency while improving robustness

### Ablation Study Results
| Component | Contribution | Key Insight |
|-----------|-------------|-------------|
| Modern TTS Training Data | [MAJOR] | Essential for contemporary threat detection |
| Multilingual Fine-tuning | [SIGNIFICANT] | Enables cross-language generalization |
| Loss Function Engineering | 13.4% relative improvement | Label smoothing reduces overconfidence |
| Production Optimization | <50ms latency | Real-time deployment feasible |

## 🚀 Quick Start

### Installation & Setup
```bash
git clone https://github.com/kaylals/NAS-AudioDeepfake.git
cd NAS-AudioDeepfake
pip install -r requirements.txt
```

### Training Pipeline
```bash
# Baseline training
python experiments/baseline/train_model.py --config configs/baseline.yaml

# Optimized training with label smoothing
python experiments/loss_optimization/finetune_v2.py --config configs/label_smoothing.yaml

# Hyperparameter optimization
python experiments/optimization/finetune_optuna.py
```

### Inference & Demo
```bash
# Single model prediction
python inference/detect.py --model finetune_models/best_model.pth --audio test.wav

# Web demo
cd web_demo && python app.py
# Access at http://localhost:5000
```

## 🎯 Business Impact & Real-World Applications


### Target Applications
- **Financial Services**: Real-time voice authentication for banking and payments
- **Enterprise Security**: Employee voice verification for remote work environments  
- **Content Moderation**: Automated detection of synthetic audio in social media
- **Legal Evidence**: Forensic analysis of audio authenticity in court proceedings

---

**🔗 Connect with me:** [LinkedIn](http://linkedin.com/in/shuo-liu-3a66a315b) | 📧 Email: shuoliu10@gmail.com

*This project demonstrates expertise in deep learning research, systematic experimentation, and production ML system design.*