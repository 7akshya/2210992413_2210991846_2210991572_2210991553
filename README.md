# Generative Adversarial Networks for Synthetic Data Generation: A Comprehensive Survey of Architectures, Applications and Challenges

## Project Information

**Authors:** Lakshya Yadav, Gaurav Singh, Eklavya Gehlawat, Sudhanshi Garg
**Roll Numbers:** 2210991846, 2210991572, 2210991553, 2210992413
**Project Type:** Research Paper / Survey Study
**Team:** Group Research Project

## Paper Overview

This research paper presents a comprehensive survey of Generative Adversarial Networks (GANs) for synthetic data generation. The study analyzes the evolution of GAN architectures from the original Vanilla GAN to more advanced and domain-specific models such as DCGAN, WGAN, WGAN-GP, SN-GAN, BigGAN, StyleGAN2, CTGAN, and TimeGAN.

The paper organizes GANs into a structured taxonomy based on their purpose:

- **Foundational Frameworks**
- **Stability-Oriented Models**
- **Domain-Specific Applications**

It also compares GAN applications across image, tabular, and time-series data generation, while discussing the most widely used evaluation metrics such as FID, Inception Score (IS), TSTR, discriminative score, and predictive score.

## Abstract

The Generative Adversarial Network (GAN), introduced by Goodfellow et al. in 2014, transformed the field of generative modeling by enabling highly realistic synthetic data generation. Unlike earlier approaches such as Variational Autoencoders, GANs rely on an adversarial framework involving a generator and a discriminator, trained simultaneously in a minimax game.

This paper reviews more than **23 GAN architectures** and categorizes them according to their purpose and application domain. It highlights how training strategies, divergence measures, regularization, and architectural innovations have shaped the development of GANs over the last decade. The study finds that models such as WGAN-GP improve training stability and mode coverage, while StyleGAN2 achieves superior perceptual quality in image synthesis. Beyond image generation, the paper also surveys CTGAN for tabular data and TimeGAN for sequential data, emphasizing that evaluation in non-image domains remains an open challenge.

The survey concludes that although GANs have reached practical maturity for image synthesis, further work is required in benchmarking, scalability, privacy, and governance for broader synthetic data applications.

## Key Contributions

- **Comprehensive Survey:** Reviews 23 influential GAN architectures from 2014–2024
- **Three-Tier Taxonomy:** Categorizes GANs into foundational, stability-oriented, and domain-specific models
- **Cross-Domain Analysis:** Covers image, tabular, and time-series data synthesis
- **Evaluation Framework:** Summarizes standard GAN evaluation metrics including FID, IS, TSTR, discriminative score, and predictive score
- **Critical Discussion:** Highlights open challenges in benchmarking, privacy, governance, and scalability
- **Research-Oriented Perspective:** Connects GAN evolution to practical applications such as healthcare, synthetic data sharing, and sequential modeling

## GAN Architectures Covered

### Foundational Frameworks
- Vanilla GAN
- DCGAN

### Stability-Oriented Models
- WGAN
- WGAN-GP
- SN-GAN

### Domain-Specific Applications
- BigGAN
- StyleGAN2
- CTGAN
- TimeGAN

## Methodology

The study follows a structured survey methodology to identify and evaluate the most influential GAN architectures.

### 1. Literature Curation

The literature review focused on major machine learning and computer vision venues, including:

- NeurIPS
- ICML
- ICLR
- CVPR
- ICCV

A total of 23 architectures were selected based on:

- citation impact
- novelty of contribution
- long-term influence on GAN research

### 2. Taxonomy Design

The reviewed architectures were grouped into three categories:

**Foundational Frameworks**
Establish baseline adversarial training principles

**Stability-Oriented Models**
Address gradient instability, convergence, and mode collapse

**Domain-Specific Applications**
Adapt GANs for specific data modalities such as high-fidelity images, tabular data, and time series

### 3. Comparative Evaluation

The survey compares models using domain-appropriate metrics:

**Image Synthesis**
- FID (Fréchet Inception Distance)
- IS (Inception Score)

**Tabular Synthesis**
- TSTR (Train on Synthetic, Test on Real)

**Time-Series Synthesis**
- Discriminative Score
- Predictive Score

## Repository Structure

```
GAN-Survey-Research-Paper/
│
├── paper/
│   └── GAN_RP.pdf
│
├── figures/
│   ├── figure1_gan_distribution.png
│   ├── figure2_taxonomy.png
│   └── figure3_evaluation_protocol.png
│
├── source-code/
│   └── gan_architectures_evaluation_suite.py
│
├── report-and-ppt/
│   └── [Project report and presentation]
│
└── README.md
```

## Key Findings

The survey shows that:

- GAN research has evolved from simple adversarial training to highly specialized architectures
- DCGAN improved image synthesis by introducing convolutional design principles
- WGAN and WGAN-GP significantly improved training stability through Wasserstein-based objectives
- SN-GAN reduced instability by controlling the discriminator's spectral norm
- BigGAN demonstrated the power of large-scale conditional image synthesis
- StyleGAN2 achieved state-of-the-art perceptual realism in image generation
- CTGAN adapted GANs for mixed-type tabular data generation
- TimeGAN showed that combining adversarial learning with temporal supervision improves time-series synthesis
- Evaluation remains inconsistent outside image domains, especially for tabular and time-series generation



## Quantitative Highlights

Some representative results discussed in the paper include:

| Model | Dataset / Task | FID ↓ | IS ↑ | Other Metric | Notes |
|-------|----------------|-------|------|--------------|-------|
| Vanilla GAN | MNIST | ~120 | ~2.0 | — | Original MLP architecture; severe mode collapse |
| DCGAN | CIFAR-10 (32×32) | 37.11 | 6.40 | — | Convolutional GAN with batch normalization |
| WGAN | CIFAR-10 | ~41 | ~3.8 | Weight clipping | Stable gradients via Wasserstein loss |
| WGAN-GP | CIFAR-10 | 29.3 | 7.86 | Gradient penalty λ=10 | Better stability and diversity |
| BigGAN | ImageNet 128×128 | 7.4 | 166.5 | Class-conditional | Large-scale conditional synthesis |
| StyleGAN2 | FFHQ 1024×1024 | 2.84 | — | PPL: 145.0 | High-fidelity face synthesis |
| CTGAN | Adult (TSTR) | N/A | N/A | TSTR ~82% | Strong tabular data synthesis |
| TimeGAN | Stocks | N/A | N/A | Disc. 0.106, Pred. 0.051 | Sequential data generation |

## Discussion

### Evaluation and Benchmarking

While FID is widely accepted for image generation, it has known limitations:

- depends on reference set size
- sensitive to feature extractor choice
- cannot fully detect memorization

For tabular and time-series synthesis, evaluation is even less standardized.

Metrics such as TSTR, discriminative score, and predictive score are useful, but results are often dataset-specific and difficult to compare across studies.

### Privacy and Differential Privacy

The survey also discusses privacy-preserving GAN variants such as PATE-GAN, which aim to provide formal differential privacy guarantees for synthetic data generation.

### Governance and Dual Use

GANs have valuable uses in:

- medical imaging
- data augmentation
- privacy-preserving data sharing

But they also pose risks in:

- deepfake generation
- misinformation
- synthetic identity misuse

This makes governance, watermarking, provenance, and policy frameworks increasingly important.

## Applications of GANs

The paper highlights GAN applications in:

- Image synthesis
- Medical image augmentation
- Data privacy and synthetic sharing
- Tabular data generation
- Financial and healthcare time-series simulation
- Image-to-image translation
- Representation learning

## Source Code / Implementation Companion

If paired with implementation, this survey can be supported by a practical benchmark including:

- Vanilla GAN
- DCGAN
- WGAN
- WGAN-GP
- SN-GAN
- Conditional GAN
- CTGAN
- TimeGAN

along with evaluation metrics such as:

- Inception Score
- FID
- TSTR
- Discriminative score

## Requirements

If you include the implementation code in the repository, the following dependencies are recommended:

### Dependencies

- Python 3.8+
- PyTorch
- torchvision
- NumPy
- SciPy
- scikit-learn
- matplotlib
- seaborn

### Hardware

- GPU recommended for training larger GAN variants
- Minimum 8GB RAM
- More storage required for image datasets such as CIFAR-10, FFHQ, or ImageNet subsets

## How to Use

### 1. Read the paper

Open the survey PDF for the full discussion of architectures, taxonomy, results, and references.

### 2. Review the figures

The paper includes:

- distribution of GAN architectures introduced over the years
- three-tier taxonomy of GAN models
- evaluation protocol diagrams

### 3. Run the implementation

If you include your GAN benchmark code:

```bash
python gan_architectures_evaluation_suite.py
```

This can be used to:

- train representative GAN variants
- compare simplified metrics
- generate summary outputs

## Conclusion

This work presents a broad and structured survey of GAN research over the past decade. It shows that GAN development has largely been driven by efforts to address mode collapse, training instability, fidelity, and domain adaptation. While image GANs have achieved remarkable maturity, synthetic data generation for tabular and time-series domains still requires better benchmarks, lighter architectures, and stronger privacy safeguards.

The paper emphasizes that future GAN research should focus on:

- standardized evaluation protocols
- computational efficiency
- privacy-by-design training
- responsible deployment and governance


## Citation

If you use this work, please cite:

```bibtex
@article{yadav2026gansurvey,
  title={Generative Adversarial Networks for Synthetic Data Generation: A Comprehensive Survey of Architectures, Applications and Challenges},
  author={Yadav, Lakshya and Singh, Gaurav and Gehlawat, Eklavya and Garg, Sudhanshi},
  year={2026},
  note={Research survey paper, Chitkara University}
}
```

## Authors and Contact

**Lakshya Yadav** — 2210991846
**Gaurav Singh** — 2210991572
**Eklavya Gehlawat** — 2210991553
**Sudhanshi Garg** — 2210992413

Department of Computer Science and Engineering
Chitkara University, Punjab 140401, India

**Emails:**
- lakshya1846.be22@chitkara.edu.in
- gaurav1572.be22@chitkara.edu.in
- eklavya1553.be22@chitkara.edu.in
- sudhanshi2413.be22@chitkara.edu.in

## Acknowledgments

The authors thank the Department of Computer Science and Engineering at Chitkara University for providing academic guidance and computational resources for this work.

## License

This project is intended for academic and research purposes only.
