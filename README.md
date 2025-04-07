# 🎯 Iterative Degradation Correction with a Mamba-Based Deep Unfolding Network for Spectral Compressive Imaging (IDC-Mamba)

<div align="center">

<!-- [![Paper Status](https://img.shields.io/badge/Paper-Published%20in%20IEEE%20TMM-success?style=for-the-badge)](https://ieeexplore.ieee.org/document/10214675)-->
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue?style=for-the-badge)](LICENSE)
<!-- [![GitHub stars](https://img.shields.io/github/stars/liu-lei98/DADFNet?style=for-the-badge)](https://github.com/liu-lei98/DADFNet)-->

</div>

## 📌 Overview
This repository contains the official PyTorch implementation of our paper:

**"Iterative Degradation Correction with a Mamba-Based Deep Unfolding Network for Spectral Compressive Imaging"**


## 🏗️ Network Architecture
<div align="center">
  <img src="https://github.com/liu-lei98/IDC-Mamba/blob/main/Figures/overall.png"   alt="">
  <p><em>Figure 1: The overall architecture of IDC-Mamba </em></p>
  <img src="https://github.com/liu-lei98/IDC-Mamba/blob/main/Figures/denoiser.png" alt="">
  <p><em>Figure 2: The overall architecture of denoiser </em></p>
  <img src="https://github.com/liu-lei98/IDC-Mamba/blob/main/Figures/vis1.png"  alt="">
  <p><em>Figure 3: The Visual Comparison </em></p>
</div>

## 🚀 Quick Start

### Prerequisites
- Linux or macOS
- NVIDIA GPU + CUDA cuDNN
- Python 3.8+
- PyTorch 1.10+

### Installation

# Create and activate conda environment
--conda create -n IDC-Mamba python=3.8
--conda activate IDC-Mamba

# Install dependencies
--pip install -r requirements.txt

