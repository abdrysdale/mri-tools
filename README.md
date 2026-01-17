# MRI Tools
![license](https://img.shields.io/github/license/abdrysdale/mri-tools.svg)

A collection of free and open-source software software tools for use in MRI.
Free is meant as in free beer (gratis) and freedom (libre).

To add a project, add the project url to the `urls.toml` file.

## Table of Contents
- [summary](#summary)
- [stats](#stats)
- [tags](#tags)
	- [mri](#mri)
	- [medical-imaging](#medical-imaging)
	- [deep-learning](#deep-learning)
	- [python](#python)
	- [neuroimaging](#neuroimaging)
	- [pytorch](#pytorch)
	- [machine-learning](#machine-learning)
	- [medical-image-processing](#medical-image-processing)
	- [segmentation](#segmentation)
	- [brain-imaging](#brain-imaging)
	- [quality-control](#quality-control)
	- [convolutional-neural-networks](#convolutional-neural-networks)
	- [medical-image-computing](#medical-image-computing)
	- [mri-images](#mri-images)
	- [diffusion-mri](#diffusion-mri)
	- [image-processing](#image-processing)
	- [quality-assurance](#quality-assurance)
	- [itk](#itk)
	- [fmri](#fmri)
	- [julia](#julia)
	- [fetal](#fetal)
	- [fastmri-challenge](#fastmri-challenge)
	- [mri-reconstruction](#mri-reconstruction)
	- [tensorflow](#tensorflow)
	- [medical-image-analysis](#medical-image-analysis)
	- [neuroscience](#neuroscience)
	- [qa](#qa)
	- [bids](#bids)
	- [super-resolution](#super-resolution)
	- [brain-connectivity](#brain-connectivity)
	- [r](#r)
	- [tractography](#tractography)
	- [image-registration](#image-registration)
	- [magnetic-resonance-imaging](#magnetic-resonance-imaging)
	- [c-plus-plus](#c-plus-plus)
	- [registration](#registration)
	- [simulation](#simulation)
	- [nifti](#nifti)
	- [medical-images](#medical-images)
	- [medical-physics](#medical-physics)
	- [computer-vision](#computer-vision)
	- [image-reconstruction](#image-reconstruction)
	- [dicom](#dicom)
- [languages](#languages)
	- [python](#python)
	- [c++](#c++)
	- [julia](#julia)
	- [jupyter-notebook](#jupyter-notebook)
	- [c](#c)
	- [javascript](#javascript)
	- [r](#r)

## Summary
| Repository | Description | Stars | Forks | Last Updated |
|---|---|---|---|---|
| MONAI | AI Toolkit for Healthcare Imaging | 7753 | 1403 | 2026-01-16 |
| torchio | Medical imaging processing for AI applications. | 2345 | 255 | 2026-01-13 |
| Slicer | Multi-platform, free open source software for visualization and image computing. | 2282 | 685 | 2026-01-16 |
| MedicalZooPytorch | A pytorch-based deep learning framework for multi-modal 2D/3D medical image segmentation | 1893 | 305 | 2026-01-13 |
| fastMRI | A large-scale dataset of both raw MRI measurements and clinical MRI images. | 1498 | 416 | 2026-01-10 |
| nilearn | Machine learning for NeuroImaging in Python | 1353 | 638 | 2026-01-16 |
| medicaldetectiontoolkit | The Medical Detection Toolkit contains 2D + 3D implementations of prevalent object detectors such as Mask R-CNN, Retina Net, Retina U-Net, as well as a training and inference framework focused on dealing with medical images.   | 1345 | 292 | 2026-01-15 |
| deepmedic | Efficient Multi-Scale 3D Convolutional Neural Network for Segmentation of 3D Medical Scans | 1058 | 347 | 2025-12-23 |
| SimpleITK | SimpleITK: a layer built on top of the Insight Toolkit (ITK), intended to simplify and facilitate ITK's use in rapid prototyping, education and interpreted languages. | 1032 | 223 | 2026-01-16 |
| medicaltorch | A medical imaging framework for Pytorch | 870 | 128 | 2025-11-13 |
| nipype | Workflows and interfaces for neuroimaging packages | 802 | 538 | 2026-01-14 |
| freesurfer | Neuroimaging analysis and visualization suite | 773 | 278 | 2026-01-16 |
| nibabel | Python package to access a cacophony of neuro-imaging file formats | 759 | 273 | 2026-01-15 |
| SynthSeg | Contrast-agnostic segmentation of MRI scans | 529 | 145 | 2026-01-15 |
| brainchop | Brainchop: In-browser 3D MRI rendering and segmentation | 512 | 61 | 2026-01-16 |
| nipy | Neuroimaging in Python FMRI analysis package | 407 | 146 | 2026-01-08 |
| mriviewer | MRI Viewer is a high performance web tool for advanced 2-D and 3-D medical visualizations. | 360 | 104 | 2026-01-13 |
| bart | BART: Toolbox for Computational Magnetic Resonance Imaging | 354 | 175 | 2026-01-14 |
| mriqc | Automated Quality Control and visual reports for Quality Assessment of structural (T1w, T2w) and functional MRI of the brain | 343 | 134 | 2026-01-16 |
| intensity-normalization | Normalize MR image intensities in Python | 340 | 58 | 2025-12-19 |
| mrtrix3 | MRtrix3 provides a set of tools to perform various advanced diffusion MRI analyses, including constrained spherical deconvolution (CSD), probabilistic tractography, track-density imaging, and apparent fibre density | 335 | 190 | 2026-01-14 |
| PyMVPA | MultiVariate Pattern Analysis in Python | 323 | 137 | 2025-12-05 |
| direct | Deep learning framework for MRI reconstruction | 294 | 47 | 2026-01-09 |
| nitime | Timeseries analysis for neuroscience data | 256 | 84 | 2026-01-13 |
| gadgetron | Gadgetron - Medical Image Reconstruction Framework | 252 | 164 | 2026-01-09 |
| TractSeg | Automatic White Matter Bundle Segmentation | 251 | 78 | 2026-01-13 |
| spinalcordtoolbox | Comprehensive and open-source library of analysis tools for MRI of the spinal cord. | 251 | 113 | 2026-01-16 |
| brainGraph | Graph theory analysis of brain MRI data | 191 | 54 | 2025-11-13 |
| pypulseq | Pulseq in Python | 186 | 77 | 2026-01-16 |
| clinicadl | Framework for the reproducible processing of neuroimaging data with deep learning methods | 176 | 60 | 2025-12-31 |
| KomaMRI.jl | Koma is a Pulseq-compatible framework to efficiently simulate Magnetic Resonance Imaging (MRI) acquisitions. The main focus of this package is to simulate general scenarios that could arise in pulse sequence development. | 176 | 31 | 2026-01-06 |
| qsiprep | Preprocessing of diffusion MRI | 172 | 62 | 2026-01-16 |
| smriprep | Structural MRI PREProcessing (sMRIPrep) workflows for NIPreps (NeuroImaging PREProcessing tools) | 162 | 47 | 2026-01-15 |
| NiftyMIC | NiftyMIC is a research-focused toolkit for motion correction and volumetric image reconstruction of 2D ultra-fast MRI. | 160 | 38 | 2025-12-23 |
| mritopng | A simple python module to make it easy to batch convert DICOM files to PNG images. | 146 | 51 | 2025-10-31 |
| pydeface | defacing utility for MRI images | 131 | 43 | 2025-12-16 |
| openMorph | Curated list of open-access databases with human structural MRI data | 129 | 38 | 2025-07-11 |
| gif_your_nifti | How to create fancy GIFs from an MRI brain image | 124 | 35 | 2025-11-16 |
| ismrmrd | ISMRM Raw Data Format | 120 | 95 | 2025-12-12 |
| RadQy | RadQy is a quality assurance and checking tool for quantitative assessment of magnetic resonance imaging (MRI) and computed tomography (CT) data. | 109 | 37 | 2026-01-13 |
| niworkflows | Common workflows for MRI (anatomical, functional, diffusion, etc) | 107 | 54 | 2026-01-13 |
| quickNAT_pytorch | PyTorch Implementation of QuickNAT and Bayesian QuickNAT, a fast brain MRI segmentation framework with segmentation Quality control using structure-wise uncertainty | 104 | 36 | 2025-12-03 |
| BraTS-Toolkit | Code to preprocess, segment, and fuse glioma MRI scans based on the BraTS Toolkit manuscript. | 98 | 14 | 2026-01-13 |
| NeSVoR | NeSVoR is a package for GPU-accelerated slice-to-volume reconstruction. | 97 | 21 | 2026-01-07 |
| MRIReco.jl | Julia Package for MRI Reconstruction | 95 | 23 | 2025-12-19 |
| NIfTI.jl | Julia module for reading/writing NIfTI MRI files | 82 | 35 | 2025-12-27 |
| virtual-scanner | An end-to-end hybrid MR simulator/console | 74 | 21 | 2026-01-06 |
| SIRF | Main repository for the CCP SynerBI software | 68 | 29 | 2026-01-15 |
| SVRTK | MIRTK based SVR reconstruction | 63 | 8 | 2025-12-25 |
| QUIT | A set of tools for processing Quantitative MR Images | 63 | 21 | 2025-09-11 |
| tensorflow-mri | A Library of TensorFlow Operators for Computational MRI | 47 | 6 | 2025-11-25 |
| DCEMRI.jl | World's fastest DCE MRI analysis toolkit | 39 | 16 | 2026-01-12 |
| DECAES.jl | DEcomposition and Component Analysis of Exponential Signals (DECAES) - a Julia implementation of the UBC Myelin Water Imaging (MWI) toolbox for computing voxelwise T2-distributions of multi spin-echo MRI images. | 35 | 6 | 2025-12-13 |
| popeye | A population receptive field estimation tool | 34 | 15 | 2025-08-18 |
| ukftractography | None | 31 | 31 | 2025-12-10 |
| mialsuperresolutiontoolkit | The Medical Image Analysis Laboratory Super-Resolution ToolKit (MIALSRTK) consists of a set of C++ and Python processing and workflow tools necessary to perform motion-robust super-resolution fetal MRI reconstruction in the BIDS Apps framework. | 30 | 15 | 2025-12-30 |
| hazen | Quality assurance framework for Magnetic Resonance Imaging | 29 | 13 | 2026-01-14 |
| DL-DiReCT | DL+DiReCT - Direct Cortical Thickness Estimation using Deep Learning-based Anatomy Segmentation and Cortex Parcellation | 28 | 6 | 2025-07-23 |
| disimpy | Massively parallel Monte Carlo diffusion MR simulator written in Python. | 27 | 9 | 2026-01-13 |
| MriResearchTools.jl | Specialized tools for MRI | 26 | 8 | 2026-01-08 |
| gropt | A toolbox for MRI gradient design | 25 | 16 | 2025-09-24 |
| nlsam | The reference implementation for the Non Local Spatial and Angular Matching (NLSAM) denoising algorithm for diffusion MRI | 24 | 11 | 2025-11-17 |
| flow4D | Python code for processing 4D flow dicoms and write velocity profiles for CFD simulations. | 24 | 6 | 2025-09-09 |
| pyCoilGen | Magnetic Field Coil Generator for Python, ported from CoilGen | 20 | 8 | 2025-12-03 |
| MRIgeneralizedBloch.jl | None | 19 | 3 | 2025-12-04 |
| dafne | Dafne (Deep Anatomical Federated Network) is a collaborative platform to annotate MRI images and train machine learning models without your data ever leaving your machine. | 18 | 6 | 2025-12-09 |
| eptlib | EPTlib - An open-source, extensible C++ library of electric properties tomography methods | 17 | 2 | 2025-10-06 |
| sHDR | HDR-MRI Algorithms from "Local contrast-enhanced MR images via high dynamic range processing" published in MRM | 16 | 0 | 2024-11-29 |
| scanhub | ScanHub combines multimodal data acquisition and complex data processing in one cloud platform. | 16 | 3 | 2025-08-30 |
| mrQA | mrQA: tools for quality assurance in medical imaging datasets, including protocol compliance | 13 | 6 | 2026-01-14 |
| PowerGrid | GPU accelerated non-Cartesian magnetic resonance imaging reconstruction toolkit | 13 | 13 | 2024-12-13 |
| ukat | UKRIN Kidney Analysis Toolbox | 12 | 4 | 2024-10-30 |
| MyoQMRI | Quantitative methods for muscle MRI | 12 | 3 | 2025-12-08 |
| CoSimPy | Python electromagnetic cosimulation library | 12 | 4 | 2026-01-01 |
| vespa | Python tools for Magnetic Resonance Spectroscopy - Pulses, Simulation and Analysis | 11 | 6 | 2025-11-12 |
| AFFIRM | A deep recursive fetal motion estimation and correction framework based on slice and volume affinity fusion | 8 | 1 | 2025-07-27 |
| fetal-IQA | Image quality assessment for fetal MRI | 6 | 0 | 2024-10-12 |
| dwybss | Blind Source Separation of diffusion MRI for free-water elimination and tissue characterization. | 1 | 0 | 2025-10-20 |
| MRISafety.jl | MRI safety checks | 0 | 0 | 2025-01-04 |
| madym_python | Mirror of python wrappers to Madym hosted on Manchester QBI GitLab project | 0 | 0 | 2021-11-22 |
| MRDQED | A Magnetic Resonance Data Quality Evaluation Dashboard | 0 | 1 | 2021-01-31 |
## Stats
- Total repos: 81
- Languages:

| Language | Count |
|---|---|
| python | 46 |
| c++ | 12 |
| julia | 7 |
| jupyter notebook | 5 |
| c | 3 |
| javascript | 3 |
| r | 2 |

- Tags:

| Tag | Count |
|---|---|
| mri | 25 |
| medical-imaging | 17 |
| deep-learning | 16 |
| python | 16 |
| neuroimaging | 11 |
| pytorch | 10 |
| machine-learning | 9 |
| medical-image-processing | 7 |
| segmentation | 7 |
| brain-imaging | 6 |
| quality-control | 5 |
| convolutional-neural-networks | 4 |
| medical-image-computing | 4 |
| mri-images | 4 |
| diffusion-mri | 4 |
| image-processing | 4 |
| quality-assurance | 3 |
| itk | 3 |
| fmri | 3 |
| julia | 2 |
| fetal | 2 |
| fastmri-challenge | 2 |
| mri-reconstruction | 2 |
| tensorflow | 2 |
| medical-image-analysis | 2 |
| neuroscience | 2 |
| qa | 2 |
| bids | 2 |
| super-resolution | 2 |
| brain-connectivity | 2 |
| r | 2 |
| tractography | 2 |
| image-registration | 2 |
| magnetic-resonance-imaging | 2 |
| c-plus-plus | 2 |
| registration | 2 |
| simulation | 2 |
| nifti | 2 |
| medical-images | 2 |
| medical-physics | 2 |
| computer-vision | 2 |
| image-reconstruction | 2 |
| dicom | 2 |

- Licenses:

| Licence | Count |
|---|---|
| other | 21 |
| mit license | 18 |
| apache license 2.0 | 16 |
| bsd 3-clause "new" or "revised" license | 9 |
| gnu general public license v3.0 | 6 |
| none | 4 |
| gnu affero general public license v3.0 | 3 |
| mozilla public license 2.0 | 2 |
| gnu lesser general public license v3.0 | 2 |




## Tags
### Mri <a name="mri"></a>
- [fastMRI](https://github.com/facebookresearch/fastMRI)
>- A large-scale dataset of both raw MRI measurements and clinical MRI images.

>- License: MIT License
>- Languages: `Python`
>- Tags: convolutional-neural-networks, deep-learning, fastmri, fastmri-challenge, fastmri-dataset, medical-imaging, mri, mri-reconstruction, pytorch
>- Forks:	416 
>- Issues:	18
>- Watchers:	1498
>- Last updated: 2026-01-10

- [freesurfer](https://github.com/freesurfer/freesurfer)
>- Neuroimaging analysis and visualization suite

>- License: Other
>- Languages: `C++`
>- Tags: freesurfer, lcn, mri, neuroimaging
>- Forks:	278 
>- Issues:	35
>- Watchers:	773
>- Last updated: 2026-01-16

- [brainchop](https://github.com/neuroneural/brainchop)
>- Brainchop: In-browser 3D MRI rendering and segmentation

>- License: MIT License
>- Languages: `Javascript`
>- Tags: 3d-segmentation, deep-learning, frontend-app, javascript, medical-imaging, mri, mri-segmentation, neuroimaging, pyodide, tensorflowjs, three-js
>- Forks:	61 
>- Issues:	5
>- Watchers:	512
>- Last updated: 2026-01-16

- [bart](https://github.com/mrirecon/bart)
>- BART: Toolbox for Computational Magnetic Resonance Imaging

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `C`
>- Tags: bart-toolbox, compressed-sensing, computational-imaging, deep-learning, iterative-methods, mri
>- Forks:	175 
>- Issues:	22
>- Watchers:	354
>- Last updated: 2026-01-14

- [mriqc](https://github.com/nipreps/mriqc)
>- Automated Quality Control and visual reports for Quality Assessment of structural (T1w, T2w) and functional MRI of the brain

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: machine-learning, mri, neuroimaging, quality-control, quality-reporter
>- Forks:	134 
>- Issues:	94
>- Watchers:	343
>- Last updated: 2026-01-16

- [intensity-normalization](https://github.com/jcreinhold/intensity-normalization)
>- Normalize MR image intensities in Python

>- License: MIT License
>- Languages: `Python`
>- Tags: fcm, harmonization, intensity-normalization, mri, neuroimaging, normalization, standardization, whitestripe, zscore
>- Forks:	58 
>- Issues:	0
>- Watchers:	340
>- Last updated: 2025-12-19

- [spinalcordtoolbox](https://github.com/spinalcordtoolbox/spinalcordtoolbox)
>- Comprehensive and open-source library of analysis tools for MRI of the spinal cord.

>- License: GNU Lesser General Public License v3.0
>- Languages: `Python`
>- Tags: mri, neuroimage, python, spinalcord
>- Forks:	113 
>- Issues:	375
>- Watchers:	251
>- Last updated: 2026-01-16

- [brainGraph](https://github.com/cwatson/brainGraph)
>- Graph theory analysis of brain MRI data

>- License: None
>- Languages: `R`
>- Tags: brain-connectivity, brain-imaging, complex-networks, connectome, connectomics, fmri, graph-theory, mri, network-analysis, neuroimaging, neuroscience, r, statistics, tractography
>- Forks:	54 
>- Issues:	11
>- Watchers:	191
>- Last updated: 2025-11-13

- [pypulseq](https://github.com/imr-framework/pypulseq)
>- Pulseq in Python

>- License: GNU Affero General Public License v3.0
>- Languages: `Python`
>- Tags: mri, mri-sequences, pulse-sequences, pulseq, python
>- Forks:	77 
>- Issues:	19
>- Watchers:	186
>- Last updated: 2026-01-16

- [KomaMRI.jl](https://github.com/JuliaHealth/KomaMRI.jl)
>- Koma is a Pulseq-compatible framework to efficiently simulate Magnetic Resonance Imaging (MRI) acquisitions. The main focus of this package is to simulate general scenarios that could arise in pulse sequence development.

>- License: MIT License
>- Languages: `Julia`
>- Tags: cardiac, diffusion, diffusion-mri, gpu-acceleration, mri, simulation
>- Forks:	31 
>- Issues:	103
>- Watchers:	176
>- Last updated: 2026-01-06

- [smriprep](https://github.com/nipreps/smriprep)
>- Structural MRI PREProcessing (sMRIPrep) workflows for NIPreps (NeuroImaging PREProcessing tools)

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: image-processing, image-registration, image-segmentation, mri, structural-mri, surface-reconstruction
>- Forks:	47 
>- Issues:	73
>- Watchers:	162
>- Last updated: 2026-01-15

- [RadQy](https://github.com/ccipd/MRQy)
>- RadQy is a quality assurance and checking tool for quantitative assessment of magnetic resonance imaging (MRI) and computed tomography (CT) data.

>- License: None
>- Languages: `Javascript`
>- Tags: imaging, machine-learning, medical-image-processing, medical-imaging, medical-physics, mri, python, quality-assurance, quality-control, quality-metrics
>- Forks:	37 
>- Issues:	1
>- Watchers:	109
>- Last updated: 2026-01-13

- [BraTS-Toolkit](https://github.com/neuronflow/BraTS-Toolkit)
>- Code to preprocess, segment, and fuse glioma MRI scans based on the BraTS Toolkit manuscript.

>- License: GNU Affero General Public License v3.0
>- Languages: `Python`
>- Tags: glioblastoma, glioma, medical-imaging, mri, segmentation
>- Forks:	14 
>- Issues:	9
>- Watchers:	98
>- Last updated: 2026-01-13

- [NeSVoR](https://github.com/daviddmc/NeSVoR)
>- NeSVoR is a package for GPU-accelerated slice-to-volume reconstruction.

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-reconstruction, 3d-visualization, deep-learning, image-reconstruction, image-registration, implicit-neural-representation, medical-imaging, mri, nerf, neural-network, neural-rendering, pytorch, segmentation, super-resolution, transformers
>- Forks:	21 
>- Issues:	7
>- Watchers:	97
>- Last updated: 2026-01-07

- [NIfTI.jl](https://github.com/JuliaNeuroscience/NIfTI.jl)
>- Julia module for reading/writing NIfTI MRI files

>- License: Other
>- Languages: `Julia`
>- Tags: fmri, julia, mri, mri-images, nifti
>- Forks:	35 
>- Issues:	30
>- Watchers:	82
>- Last updated: 2025-12-27

- [virtual-scanner](https://github.com/imr-framework/virtual-scanner)
>- An end-to-end hybrid MR simulator/console

>- License: GNU Affero General Public License v3.0
>- Languages: `Jupyter Notebook`
>- Tags: mri
>- Forks:	21 
>- Issues:	15
>- Watchers:	74
>- Last updated: 2026-01-06

- [SVRTK](https://github.com/SVRTK/SVRTK)
>- MIRTK based SVR reconstruction

>- License: Apache License 2.0
>- Languages: `C++`
>- Tags: fetal, mri, reconstruction, retrospecitve, slice-to-volume
>- Forks:	8 
>- Issues:	5
>- Watchers:	63
>- Last updated: 2025-12-25

- [tensorflow-mri](https://github.com/mrphys/tensorflow-mri)
>- A Library of TensorFlow Operators for Computational MRI

>- License: Apache License 2.0
>- Languages: `Jupyter Notebook`
>- Tags: machine-learning, magnetic-resonance-imaging, ml, mri, python, tensorflow
>- Forks:	6 
>- Issues:	10
>- Watchers:	47
>- Last updated: 2025-11-25

- [mialsuperresolutiontoolkit](https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit)
>- The Medical Image Analysis Laboratory Super-Resolution ToolKit (MIALSRTK) consists of a set of C++ and Python processing and workflow tools necessary to perform motion-robust super-resolution fetal MRI reconstruction in the BIDS Apps framework.

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `C++`
>- Tags: bids, bids-apps, fetal, itk, mri, nipype, super-resolution, workflow
>- Forks:	15 
>- Issues:	17
>- Watchers:	30
>- Last updated: 2025-12-30

- [hazen](https://github.com/GSTT-CSC/hazen)
>- Quality assurance framework for Magnetic Resonance Imaging

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: image-processing, mri, mri-phantoms, python, qa, quality-assurance
>- Forks:	13 
>- Issues:	58
>- Watchers:	29
>- Last updated: 2026-01-14

- [DL-DiReCT](https://github.com/SCAN-NRAD/DL-DiReCT)
>- DL+DiReCT - Direct Cortical Thickness Estimation using Deep Learning-based Anatomy Segmentation and Cortex Parcellation

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: cortical-thickness, deep-learning, morphometry, mri
>- Forks:	6 
>- Issues:	4
>- Watchers:	28
>- Last updated: 2025-07-23

- [MriResearchTools.jl](https://github.com/korbinian90/MriResearchTools.jl)
>- Specialized tools for MRI

>- License: MIT License
>- Languages: `Julia`
>- Tags: mri, mri-images
>- Forks:	8 
>- Issues:	3
>- Watchers:	26
>- Last updated: 2026-01-08

- [pyCoilGen](https://github.com/kev-m/pyCoilGen)
>- Magnetic Field Coil Generator for Python, ported from CoilGen

>- License: GNU General Public License v3.0
>- Languages: `Python`
>- Tags: magnetic-field-solver, magnetic-resonance-imaging, medical-physics, mri, nmr, physics
>- Forks:	8 
>- Issues:	3
>- Watchers:	20
>- Last updated: 2025-12-03

- [sHDR](https://github.com/shakes76/sHDR)
>- HDR-MRI Algorithms from "Local contrast-enhanced MR images via high dynamic range processing" published in MRM

>- License: Other
>- Languages: `C++`
>- Tags: fusion, hdr, image, medical-image-processing, medical-imaging, mri
>- Forks:	0 
>- Issues:	0
>- Watchers:	16
>- Last updated: 2024-11-29

- [mrQA](https://github.com/Open-Minds-Lab/mrQA)
>- mrQA: tools for quality assurance in medical imaging datasets, including protocol compliance

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: brain, ismrm, mr-image, mri, mri-brain, mri-images, neuroimaging, neuroscience, niqc, qa, quality-assurance, quality-control
>- Forks:	6 
>- Issues:	38
>- Watchers:	13
>- Last updated: 2026-01-14

### Medical-Imaging <a name="medical-imaging"></a>
- [Slicer](https://github.com/Slicer/Slicer)
>- Multi-platform, free open source software for visualization and image computing.

>- License: Other
>- Languages: `C++`
>- Tags: 3d-printing, 3d-slicer, c-plus-plus, computed-tomography, image-guided-therapy, image-processing, itk, kitware, medical-image-computing, medical-imaging, national-institutes-of-health, neuroimaging, nih, python, qt, registration, segmentation, tcia-dac, tractography, vtk
>- Forks:	685 
>- Issues:	647
>- Watchers:	2282
>- Last updated: 2026-01-16

- [MedicalZooPytorch](https://github.com/black0017/MedicalZooPytorch)
>- A pytorch-based deep learning framework for multi-modal 2D/3D medical image segmentation

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-convolutional-network, brats2018, brats2019, deep-learning, densenet, iseg, iseg-challenge, medical-image-processing, medical-image-segmentation, medical-imaging, mrbrains18, pytorch, resnet, segmentation, segmentation-models, unet, unet-image-segmentation
>- Forks:	305 
>- Issues:	21
>- Watchers:	1893
>- Last updated: 2026-01-13

- [fastMRI](https://github.com/facebookresearch/fastMRI)
>- A large-scale dataset of both raw MRI measurements and clinical MRI images.

>- License: MIT License
>- Languages: `Python`
>- Tags: convolutional-neural-networks, deep-learning, fastmri, fastmri-challenge, fastmri-dataset, medical-imaging, mri, mri-reconstruction, pytorch
>- Forks:	416 
>- Issues:	18
>- Watchers:	1498
>- Last updated: 2026-01-10

- [medicaldetectiontoolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit)
>- The Medical Detection Toolkit contains 2D + 3D implementations of prevalent object detectors such as Mask R-CNN, Retina Net, Retina U-Net, as well as a training and inference framework focused on dealing with medical images.  

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: 3d-mask-rcnn, 3d-models, 3d-object-detection, deep-learning, deep-neural-networks, detection, mask-rcnn, medical-image-analysis, medical-image-computing, medical-image-processing, medical-imaging, object-detection, pytorch-cnn, pytorch-deeplearning, pytorch-implementation, retina-net, retina-unet, segmentation, semantic-segmentation, u-net
>- Forks:	292 
>- Issues:	47
>- Watchers:	1345
>- Last updated: 2026-01-15

- [deepmedic](https://github.com/deepmedic/deepmedic)
>- Efficient Multi-Scale 3D Convolutional Neural Network for Segmentation of 3D Medical Scans

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: convolutional-neural-networks, deep-learning, machine-learning, medical-imaging, neural-networks
>- Forks:	347 
>- Issues:	23
>- Watchers:	1058
>- Last updated: 2025-12-23

- [medicaltorch](https://github.com/perone/medicaltorch?tab=readme-ov-file)
>- A medical imaging framework for Pytorch

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: computer-vision, deep-learning, machine-learning, medical-imaging, python, pytorch
>- Forks:	128 
>- Issues:	17
>- Watchers:	870
>- Last updated: 2025-11-13

- [brainchop](https://github.com/neuroneural/brainchop)
>- Brainchop: In-browser 3D MRI rendering and segmentation

>- License: MIT License
>- Languages: `Javascript`
>- Tags: 3d-segmentation, deep-learning, frontend-app, javascript, medical-imaging, mri, mri-segmentation, neuroimaging, pyodide, tensorflowjs, three-js
>- Forks:	61 
>- Issues:	5
>- Watchers:	512
>- Last updated: 2026-01-16

- [direct](https://github.com/NKI-AI/direct)
>- Deep learning framework for MRI reconstruction

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: deep-learning, fastmri-challenge, inverse-problems, medical-imaging, mri-reconstruction, pytorch
>- Forks:	47 
>- Issues:	6
>- Watchers:	294
>- Last updated: 2026-01-09

- [clinicadl](https://github.com/aramis-lab/clinicadl)
>- Framework for the reproducible processing of neuroimaging data with deep learning methods

>- License: MIT License
>- Languages: `Python`
>- Tags: alzheimer-disease, brain-imaging, convolutional-neural-network, deep-learning, medical-imaging, neuroimaging, python, pytorch
>- Forks:	60 
>- Issues:	65
>- Watchers:	176
>- Last updated: 2025-12-31

- [RadQy](https://github.com/ccipd/MRQy)
>- RadQy is a quality assurance and checking tool for quantitative assessment of magnetic resonance imaging (MRI) and computed tomography (CT) data.

>- License: None
>- Languages: `Javascript`
>- Tags: imaging, machine-learning, medical-image-processing, medical-imaging, medical-physics, mri, python, quality-assurance, quality-control, quality-metrics
>- Forks:	37 
>- Issues:	1
>- Watchers:	109
>- Last updated: 2026-01-13

- [quickNAT_pytorch](https://github.com/ai-med/quickNAT_pytorch)
>- PyTorch Implementation of QuickNAT and Bayesian QuickNAT, a fast brain MRI segmentation framework with segmentation Quality control using structure-wise uncertainty

>- License: MIT License
>- Languages: `Python`
>- Tags: ai, bayesian, biomarkers, brain-imaging, computer-vision, convolutional-neural-networks, deep-learning, machine-learning, medical-imaging, mri-images, neuroanatomy, pytorch, quality-control, segmentation, uncertainty
>- Forks:	36 
>- Issues:	7
>- Watchers:	104
>- Last updated: 2025-12-03

- [BraTS-Toolkit](https://github.com/neuronflow/BraTS-Toolkit)
>- Code to preprocess, segment, and fuse glioma MRI scans based on the BraTS Toolkit manuscript.

>- License: GNU Affero General Public License v3.0
>- Languages: `Python`
>- Tags: glioblastoma, glioma, medical-imaging, mri, segmentation
>- Forks:	14 
>- Issues:	9
>- Watchers:	98
>- Last updated: 2026-01-13

- [NeSVoR](https://github.com/daviddmc/NeSVoR)
>- NeSVoR is a package for GPU-accelerated slice-to-volume reconstruction.

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-reconstruction, 3d-visualization, deep-learning, image-reconstruction, image-registration, implicit-neural-representation, medical-imaging, mri, nerf, neural-network, neural-rendering, pytorch, segmentation, super-resolution, transformers
>- Forks:	21 
>- Issues:	7
>- Watchers:	97
>- Last updated: 2026-01-07

- [SIRF](https://github.com/SyneRBI/SIRF?tab=readme-ov-file)
>- Main repository for the CCP SynerBI software

>- License: Other
>- Languages: `C++`
>- Tags: image-reconstruction, medical-imaging, pet-mr
>- Forks:	29 
>- Issues:	167
>- Watchers:	68
>- Last updated: 2026-01-15

- [DCEMRI.jl](https://github.com/davidssmith/DCEMRI.jl)
>- World's fastest DCE MRI analysis toolkit

>- License: Other
>- Languages: `Julia`
>- Tags: analysis, cancer-imaging-research, dce-mri, julia, mat-files, medical-image-processing, medical-imaging
>- Forks:	16 
>- Issues:	5
>- Watchers:	39
>- Last updated: 2026-01-12

- [sHDR](https://github.com/shakes76/sHDR)
>- HDR-MRI Algorithms from "Local contrast-enhanced MR images via high dynamic range processing" published in MRM

>- License: Other
>- Languages: `C++`
>- Tags: fusion, hdr, image, medical-image-processing, medical-imaging, mri
>- Forks:	0 
>- Issues:	0
>- Watchers:	16
>- Last updated: 2024-11-29

- [fetal-IQA](https://github.com/daviddmc/fetal-IQA)
>- Image quality assessment for fetal MRI

>- License: MIT License
>- Languages: `Python`
>- Tags: convolutional-neural-networks, deep-learning, fetal-mri, medical-imaging, pytorch, quality-control, semi-supervised-learning, tensorflow
>- Forks:	0 
>- Issues:	0
>- Watchers:	6
>- Last updated: 2024-10-12

### Deep-Learning <a name="deep-learning"></a>
- [MONAI](https://github.com/Project-MONAI/MONAI)
>- AI Toolkit for Healthcare Imaging

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: deep-learning, healthcare-imaging, medical-image-computing, medical-image-processing, monai, python3, pytorch
>- Forks:	1403 
>- Issues:	505
>- Watchers:	7753
>- Last updated: 2026-01-16

- [torchio](https://github.com/fepegar/torchio)
>- Medical imaging processing for AI applications.

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: augmentation, data-augmentation, deep-learning, machine-learning, medical-image-analysis, medical-image-computing, medical-image-processing, medical-images, medical-imaging-datasets, medical-imaging-with-deep-learning, python, pytorch
>- Forks:	255 
>- Issues:	39
>- Watchers:	2345
>- Last updated: 2026-01-13

- [MedicalZooPytorch](https://github.com/black0017/MedicalZooPytorch)
>- A pytorch-based deep learning framework for multi-modal 2D/3D medical image segmentation

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-convolutional-network, brats2018, brats2019, deep-learning, densenet, iseg, iseg-challenge, medical-image-processing, medical-image-segmentation, medical-imaging, mrbrains18, pytorch, resnet, segmentation, segmentation-models, unet, unet-image-segmentation
>- Forks:	305 
>- Issues:	21
>- Watchers:	1893
>- Last updated: 2026-01-13

- [fastMRI](https://github.com/facebookresearch/fastMRI)
>- A large-scale dataset of both raw MRI measurements and clinical MRI images.

>- License: MIT License
>- Languages: `Python`
>- Tags: convolutional-neural-networks, deep-learning, fastmri, fastmri-challenge, fastmri-dataset, medical-imaging, mri, mri-reconstruction, pytorch
>- Forks:	416 
>- Issues:	18
>- Watchers:	1498
>- Last updated: 2026-01-10

- [medicaldetectiontoolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit)
>- The Medical Detection Toolkit contains 2D + 3D implementations of prevalent object detectors such as Mask R-CNN, Retina Net, Retina U-Net, as well as a training and inference framework focused on dealing with medical images.  

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: 3d-mask-rcnn, 3d-models, 3d-object-detection, deep-learning, deep-neural-networks, detection, mask-rcnn, medical-image-analysis, medical-image-computing, medical-image-processing, medical-imaging, object-detection, pytorch-cnn, pytorch-deeplearning, pytorch-implementation, retina-net, retina-unet, segmentation, semantic-segmentation, u-net
>- Forks:	292 
>- Issues:	47
>- Watchers:	1345
>- Last updated: 2026-01-15

- [deepmedic](https://github.com/deepmedic/deepmedic)
>- Efficient Multi-Scale 3D Convolutional Neural Network for Segmentation of 3D Medical Scans

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: convolutional-neural-networks, deep-learning, machine-learning, medical-imaging, neural-networks
>- Forks:	347 
>- Issues:	23
>- Watchers:	1058
>- Last updated: 2025-12-23

- [medicaltorch](https://github.com/perone/medicaltorch?tab=readme-ov-file)
>- A medical imaging framework for Pytorch

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: computer-vision, deep-learning, machine-learning, medical-imaging, python, pytorch
>- Forks:	128 
>- Issues:	17
>- Watchers:	870
>- Last updated: 2025-11-13

- [brainchop](https://github.com/neuroneural/brainchop)
>- Brainchop: In-browser 3D MRI rendering and segmentation

>- License: MIT License
>- Languages: `Javascript`
>- Tags: 3d-segmentation, deep-learning, frontend-app, javascript, medical-imaging, mri, mri-segmentation, neuroimaging, pyodide, tensorflowjs, three-js
>- Forks:	61 
>- Issues:	5
>- Watchers:	512
>- Last updated: 2026-01-16

- [bart](https://github.com/mrirecon/bart)
>- BART: Toolbox for Computational Magnetic Resonance Imaging

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `C`
>- Tags: bart-toolbox, compressed-sensing, computational-imaging, deep-learning, iterative-methods, mri
>- Forks:	175 
>- Issues:	22
>- Watchers:	354
>- Last updated: 2026-01-14

- [direct](https://github.com/NKI-AI/direct)
>- Deep learning framework for MRI reconstruction

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: deep-learning, fastmri-challenge, inverse-problems, medical-imaging, mri-reconstruction, pytorch
>- Forks:	47 
>- Issues:	6
>- Watchers:	294
>- Last updated: 2026-01-09

- [clinicadl](https://github.com/aramis-lab/clinicadl)
>- Framework for the reproducible processing of neuroimaging data with deep learning methods

>- License: MIT License
>- Languages: `Python`
>- Tags: alzheimer-disease, brain-imaging, convolutional-neural-network, deep-learning, medical-imaging, neuroimaging, python, pytorch
>- Forks:	60 
>- Issues:	65
>- Watchers:	176
>- Last updated: 2025-12-31

- [quickNAT_pytorch](https://github.com/ai-med/quickNAT_pytorch)
>- PyTorch Implementation of QuickNAT and Bayesian QuickNAT, a fast brain MRI segmentation framework with segmentation Quality control using structure-wise uncertainty

>- License: MIT License
>- Languages: `Python`
>- Tags: ai, bayesian, biomarkers, brain-imaging, computer-vision, convolutional-neural-networks, deep-learning, machine-learning, medical-imaging, mri-images, neuroanatomy, pytorch, quality-control, segmentation, uncertainty
>- Forks:	36 
>- Issues:	7
>- Watchers:	104
>- Last updated: 2025-12-03

- [NeSVoR](https://github.com/daviddmc/NeSVoR)
>- NeSVoR is a package for GPU-accelerated slice-to-volume reconstruction.

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-reconstruction, 3d-visualization, deep-learning, image-reconstruction, image-registration, implicit-neural-representation, medical-imaging, mri, nerf, neural-network, neural-rendering, pytorch, segmentation, super-resolution, transformers
>- Forks:	21 
>- Issues:	7
>- Watchers:	97
>- Last updated: 2026-01-07

- [DL-DiReCT](https://github.com/SCAN-NRAD/DL-DiReCT)
>- DL+DiReCT - Direct Cortical Thickness Estimation using Deep Learning-based Anatomy Segmentation and Cortex Parcellation

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: cortical-thickness, deep-learning, morphometry, mri
>- Forks:	6 
>- Issues:	4
>- Watchers:	28
>- Last updated: 2025-07-23

- [AFFIRM](https://github.com/allard-shi/affirm)
>- A deep recursive fetal motion estimation and correction framework based on slice and volume affinity fusion

>- License: MIT License
>- Languages: `Python`
>- Tags: deep-learning, fetus, motion
>- Forks:	1 
>- Issues:	0
>- Watchers:	8
>- Last updated: 2025-07-27

- [fetal-IQA](https://github.com/daviddmc/fetal-IQA)
>- Image quality assessment for fetal MRI

>- License: MIT License
>- Languages: `Python`
>- Tags: convolutional-neural-networks, deep-learning, fetal-mri, medical-imaging, pytorch, quality-control, semi-supervised-learning, tensorflow
>- Forks:	0 
>- Issues:	0
>- Watchers:	6
>- Last updated: 2024-10-12

### Python <a name="python"></a>
- [torchio](https://github.com/fepegar/torchio)
>- Medical imaging processing for AI applications.

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: augmentation, data-augmentation, deep-learning, machine-learning, medical-image-analysis, medical-image-computing, medical-image-processing, medical-images, medical-imaging-datasets, medical-imaging-with-deep-learning, python, pytorch
>- Forks:	255 
>- Issues:	39
>- Watchers:	2345
>- Last updated: 2026-01-13

- [Slicer](https://github.com/Slicer/Slicer)
>- Multi-platform, free open source software for visualization and image computing.

>- License: Other
>- Languages: `C++`
>- Tags: 3d-printing, 3d-slicer, c-plus-plus, computed-tomography, image-guided-therapy, image-processing, itk, kitware, medical-image-computing, medical-imaging, national-institutes-of-health, neuroimaging, nih, python, qt, registration, segmentation, tcia-dac, tractography, vtk
>- Forks:	685 
>- Issues:	647
>- Watchers:	2282
>- Last updated: 2026-01-16

- [nilearn](https://github.com/nilearn/nilearn)
>- Machine learning for NeuroImaging in Python

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: brain-connectivity, brain-imaging, brain-mri, decoding, fmri, machine-learning, mvpa, neuroimaging, python
>- Forks:	638 
>- Issues:	272
>- Watchers:	1353
>- Last updated: 2026-01-16

- [SimpleITK](https://github.com/SimpleITK/SimpleITK)
>- SimpleITK: a layer built on top of the Insight Toolkit (ITK), intended to simplify and facilitate ITK's use in rapid prototyping, education and interpreted languages.

>- License: Apache License 2.0
>- Languages: `Swig`
>- Tags: c-plus-plus, csharp, image-analysis, image-processing, itk, java, lua, python, r, registration, ruby, segmentation, simpleitk, swig, tcl
>- Forks:	223 
>- Issues:	71
>- Watchers:	1032
>- Last updated: 2026-01-16

- [medicaltorch](https://github.com/perone/medicaltorch?tab=readme-ov-file)
>- A medical imaging framework for Pytorch

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: computer-vision, deep-learning, machine-learning, medical-imaging, python, pytorch
>- Forks:	128 
>- Issues:	17
>- Watchers:	870
>- Last updated: 2025-11-13

- [nipype](https://github.com/nipy/nipype)
>- Workflows and interfaces for neuroimaging packages

>- License: Other
>- Languages: `Python`
>- Tags: big-data, brain-imaging, brainweb, data-science, dataflow, dataflow-programming, neuroimaging, python, workflow-engine
>- Forks:	538 
>- Issues:	434
>- Watchers:	802
>- Last updated: 2026-01-14

- [nibabel](https://github.com/nipy/nibabel)
>- Python package to access a cacophony of neuro-imaging file formats

>- License: Other
>- Languages: `Python`
>- Tags: afni-brik-head, brain-imaging, cifti-2, data-formats, dicom, ecat, gifti, minc, neuroimaging, nifti, python, streamlines, tck, trk
>- Forks:	273 
>- Issues:	145
>- Watchers:	759
>- Last updated: 2026-01-15

- [spinalcordtoolbox](https://github.com/spinalcordtoolbox/spinalcordtoolbox)
>- Comprehensive and open-source library of analysis tools for MRI of the spinal cord.

>- License: GNU Lesser General Public License v3.0
>- Languages: `Python`
>- Tags: mri, neuroimage, python, spinalcord
>- Forks:	113 
>- Issues:	375
>- Watchers:	251
>- Last updated: 2026-01-16

- [pypulseq](https://github.com/imr-framework/pypulseq)
>- Pulseq in Python

>- License: GNU Affero General Public License v3.0
>- Languages: `Python`
>- Tags: mri, mri-sequences, pulse-sequences, pulseq, python
>- Forks:	77 
>- Issues:	19
>- Watchers:	186
>- Last updated: 2026-01-16

- [clinicadl](https://github.com/aramis-lab/clinicadl)
>- Framework for the reproducible processing of neuroimaging data with deep learning methods

>- License: MIT License
>- Languages: `Python`
>- Tags: alzheimer-disease, brain-imaging, convolutional-neural-network, deep-learning, medical-imaging, neuroimaging, python, pytorch
>- Forks:	60 
>- Issues:	65
>- Watchers:	176
>- Last updated: 2025-12-31

- [mritopng](https://github.com/danishm/mritopng)
>- A simple python module to make it easy to batch convert DICOM files to PNG images.

>- License: MIT License
>- Languages: `Python`
>- Tags: dicom, dicom-converter, dicom-images, medical, medical-images, png, python
>- Forks:	51 
>- Issues:	5
>- Watchers:	146
>- Last updated: 2025-10-31

- [RadQy](https://github.com/ccipd/MRQy)
>- RadQy is a quality assurance and checking tool for quantitative assessment of magnetic resonance imaging (MRI) and computed tomography (CT) data.

>- License: None
>- Languages: `Javascript`
>- Tags: imaging, machine-learning, medical-image-processing, medical-imaging, medical-physics, mri, python, quality-assurance, quality-control, quality-metrics
>- Forks:	37 
>- Issues:	1
>- Watchers:	109
>- Last updated: 2026-01-13

- [tensorflow-mri](https://github.com/mrphys/tensorflow-mri)
>- A Library of TensorFlow Operators for Computational MRI

>- License: Apache License 2.0
>- Languages: `Jupyter Notebook`
>- Tags: machine-learning, magnetic-resonance-imaging, ml, mri, python, tensorflow
>- Forks:	6 
>- Issues:	10
>- Watchers:	47
>- Last updated: 2025-11-25

- [hazen](https://github.com/GSTT-CSC/hazen)
>- Quality assurance framework for Magnetic Resonance Imaging

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: image-processing, mri, mri-phantoms, python, qa, quality-assurance
>- Forks:	13 
>- Issues:	58
>- Watchers:	29
>- Last updated: 2026-01-14

- [nlsam](https://github.com/samuelstjean/nlsam)
>- The reference implementation for the Non Local Spatial and Angular Matching (NLSAM) denoising algorithm for diffusion MRI

>- License: GNU General Public License v3.0
>- Languages: `Python`
>- Tags: denoising-algorithm, diffusion-mri, machine-learning, python
>- Forks:	11 
>- Issues:	1
>- Watchers:	24
>- Last updated: 2025-11-17

- [vespa](https://github.com/vespa-mrs/vespa)
>- Python tools for Magnetic Resonance Spectroscopy - Pulses, Simulation and Analysis

>- License: Other
>- Languages: `Python`
>- Tags: fitting, mrs, python, rf-pulse, simulation, spectroscopy, wxpython
>- Forks:	6 
>- Issues:	6
>- Watchers:	11
>- Last updated: 2025-11-12

### Neuroimaging <a name="neuroimaging"></a>
- [Slicer](https://github.com/Slicer/Slicer)
>- Multi-platform, free open source software for visualization and image computing.

>- License: Other
>- Languages: `C++`
>- Tags: 3d-printing, 3d-slicer, c-plus-plus, computed-tomography, image-guided-therapy, image-processing, itk, kitware, medical-image-computing, medical-imaging, national-institutes-of-health, neuroimaging, nih, python, qt, registration, segmentation, tcia-dac, tractography, vtk
>- Forks:	685 
>- Issues:	647
>- Watchers:	2282
>- Last updated: 2026-01-16

- [nilearn](https://github.com/nilearn/nilearn)
>- Machine learning for NeuroImaging in Python

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: brain-connectivity, brain-imaging, brain-mri, decoding, fmri, machine-learning, mvpa, neuroimaging, python
>- Forks:	638 
>- Issues:	272
>- Watchers:	1353
>- Last updated: 2026-01-16

- [nipype](https://github.com/nipy/nipype)
>- Workflows and interfaces for neuroimaging packages

>- License: Other
>- Languages: `Python`
>- Tags: big-data, brain-imaging, brainweb, data-science, dataflow, dataflow-programming, neuroimaging, python, workflow-engine
>- Forks:	538 
>- Issues:	434
>- Watchers:	802
>- Last updated: 2026-01-14

- [freesurfer](https://github.com/freesurfer/freesurfer)
>- Neuroimaging analysis and visualization suite

>- License: Other
>- Languages: `C++`
>- Tags: freesurfer, lcn, mri, neuroimaging
>- Forks:	278 
>- Issues:	35
>- Watchers:	773
>- Last updated: 2026-01-16

- [nibabel](https://github.com/nipy/nibabel)
>- Python package to access a cacophony of neuro-imaging file formats

>- License: Other
>- Languages: `Python`
>- Tags: afni-brik-head, brain-imaging, cifti-2, data-formats, dicom, ecat, gifti, minc, neuroimaging, nifti, python, streamlines, tck, trk
>- Forks:	273 
>- Issues:	145
>- Watchers:	759
>- Last updated: 2026-01-15

- [brainchop](https://github.com/neuroneural/brainchop)
>- Brainchop: In-browser 3D MRI rendering and segmentation

>- License: MIT License
>- Languages: `Javascript`
>- Tags: 3d-segmentation, deep-learning, frontend-app, javascript, medical-imaging, mri, mri-segmentation, neuroimaging, pyodide, tensorflowjs, three-js
>- Forks:	61 
>- Issues:	5
>- Watchers:	512
>- Last updated: 2026-01-16

- [mriqc](https://github.com/nipreps/mriqc)
>- Automated Quality Control and visual reports for Quality Assessment of structural (T1w, T2w) and functional MRI of the brain

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: machine-learning, mri, neuroimaging, quality-control, quality-reporter
>- Forks:	134 
>- Issues:	94
>- Watchers:	343
>- Last updated: 2026-01-16

- [intensity-normalization](https://github.com/jcreinhold/intensity-normalization)
>- Normalize MR image intensities in Python

>- License: MIT License
>- Languages: `Python`
>- Tags: fcm, harmonization, intensity-normalization, mri, neuroimaging, normalization, standardization, whitestripe, zscore
>- Forks:	58 
>- Issues:	0
>- Watchers:	340
>- Last updated: 2025-12-19

- [brainGraph](https://github.com/cwatson/brainGraph)
>- Graph theory analysis of brain MRI data

>- License: None
>- Languages: `R`
>- Tags: brain-connectivity, brain-imaging, complex-networks, connectome, connectomics, fmri, graph-theory, mri, network-analysis, neuroimaging, neuroscience, r, statistics, tractography
>- Forks:	54 
>- Issues:	11
>- Watchers:	191
>- Last updated: 2025-11-13

- [clinicadl](https://github.com/aramis-lab/clinicadl)
>- Framework for the reproducible processing of neuroimaging data with deep learning methods

>- License: MIT License
>- Languages: `Python`
>- Tags: alzheimer-disease, brain-imaging, convolutional-neural-network, deep-learning, medical-imaging, neuroimaging, python, pytorch
>- Forks:	60 
>- Issues:	65
>- Watchers:	176
>- Last updated: 2025-12-31

- [mrQA](https://github.com/Open-Minds-Lab/mrQA)
>- mrQA: tools for quality assurance in medical imaging datasets, including protocol compliance

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: brain, ismrm, mr-image, mri, mri-brain, mri-images, neuroimaging, neuroscience, niqc, qa, quality-assurance, quality-control
>- Forks:	6 
>- Issues:	38
>- Watchers:	13
>- Last updated: 2026-01-14

### Pytorch <a name="pytorch"></a>
- [MONAI](https://github.com/Project-MONAI/MONAI)
>- AI Toolkit for Healthcare Imaging

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: deep-learning, healthcare-imaging, medical-image-computing, medical-image-processing, monai, python3, pytorch
>- Forks:	1403 
>- Issues:	505
>- Watchers:	7753
>- Last updated: 2026-01-16

- [torchio](https://github.com/fepegar/torchio)
>- Medical imaging processing for AI applications.

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: augmentation, data-augmentation, deep-learning, machine-learning, medical-image-analysis, medical-image-computing, medical-image-processing, medical-images, medical-imaging-datasets, medical-imaging-with-deep-learning, python, pytorch
>- Forks:	255 
>- Issues:	39
>- Watchers:	2345
>- Last updated: 2026-01-13

- [MedicalZooPytorch](https://github.com/black0017/MedicalZooPytorch)
>- A pytorch-based deep learning framework for multi-modal 2D/3D medical image segmentation

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-convolutional-network, brats2018, brats2019, deep-learning, densenet, iseg, iseg-challenge, medical-image-processing, medical-image-segmentation, medical-imaging, mrbrains18, pytorch, resnet, segmentation, segmentation-models, unet, unet-image-segmentation
>- Forks:	305 
>- Issues:	21
>- Watchers:	1893
>- Last updated: 2026-01-13

- [fastMRI](https://github.com/facebookresearch/fastMRI)
>- A large-scale dataset of both raw MRI measurements and clinical MRI images.

>- License: MIT License
>- Languages: `Python`
>- Tags: convolutional-neural-networks, deep-learning, fastmri, fastmri-challenge, fastmri-dataset, medical-imaging, mri, mri-reconstruction, pytorch
>- Forks:	416 
>- Issues:	18
>- Watchers:	1498
>- Last updated: 2026-01-10

- [medicaltorch](https://github.com/perone/medicaltorch?tab=readme-ov-file)
>- A medical imaging framework for Pytorch

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: computer-vision, deep-learning, machine-learning, medical-imaging, python, pytorch
>- Forks:	128 
>- Issues:	17
>- Watchers:	870
>- Last updated: 2025-11-13

- [direct](https://github.com/NKI-AI/direct)
>- Deep learning framework for MRI reconstruction

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: deep-learning, fastmri-challenge, inverse-problems, medical-imaging, mri-reconstruction, pytorch
>- Forks:	47 
>- Issues:	6
>- Watchers:	294
>- Last updated: 2026-01-09

- [clinicadl](https://github.com/aramis-lab/clinicadl)
>- Framework for the reproducible processing of neuroimaging data with deep learning methods

>- License: MIT License
>- Languages: `Python`
>- Tags: alzheimer-disease, brain-imaging, convolutional-neural-network, deep-learning, medical-imaging, neuroimaging, python, pytorch
>- Forks:	60 
>- Issues:	65
>- Watchers:	176
>- Last updated: 2025-12-31

- [quickNAT_pytorch](https://github.com/ai-med/quickNAT_pytorch)
>- PyTorch Implementation of QuickNAT and Bayesian QuickNAT, a fast brain MRI segmentation framework with segmentation Quality control using structure-wise uncertainty

>- License: MIT License
>- Languages: `Python`
>- Tags: ai, bayesian, biomarkers, brain-imaging, computer-vision, convolutional-neural-networks, deep-learning, machine-learning, medical-imaging, mri-images, neuroanatomy, pytorch, quality-control, segmentation, uncertainty
>- Forks:	36 
>- Issues:	7
>- Watchers:	104
>- Last updated: 2025-12-03

- [NeSVoR](https://github.com/daviddmc/NeSVoR)
>- NeSVoR is a package for GPU-accelerated slice-to-volume reconstruction.

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-reconstruction, 3d-visualization, deep-learning, image-reconstruction, image-registration, implicit-neural-representation, medical-imaging, mri, nerf, neural-network, neural-rendering, pytorch, segmentation, super-resolution, transformers
>- Forks:	21 
>- Issues:	7
>- Watchers:	97
>- Last updated: 2026-01-07

- [fetal-IQA](https://github.com/daviddmc/fetal-IQA)
>- Image quality assessment for fetal MRI

>- License: MIT License
>- Languages: `Python`
>- Tags: convolutional-neural-networks, deep-learning, fetal-mri, medical-imaging, pytorch, quality-control, semi-supervised-learning, tensorflow
>- Forks:	0 
>- Issues:	0
>- Watchers:	6
>- Last updated: 2024-10-12

### Machine-Learning <a name="machine-learning"></a>
- [torchio](https://github.com/fepegar/torchio)
>- Medical imaging processing for AI applications.

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: augmentation, data-augmentation, deep-learning, machine-learning, medical-image-analysis, medical-image-computing, medical-image-processing, medical-images, medical-imaging-datasets, medical-imaging-with-deep-learning, python, pytorch
>- Forks:	255 
>- Issues:	39
>- Watchers:	2345
>- Last updated: 2026-01-13

- [nilearn](https://github.com/nilearn/nilearn)
>- Machine learning for NeuroImaging in Python

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: brain-connectivity, brain-imaging, brain-mri, decoding, fmri, machine-learning, mvpa, neuroimaging, python
>- Forks:	638 
>- Issues:	272
>- Watchers:	1353
>- Last updated: 2026-01-16

- [deepmedic](https://github.com/deepmedic/deepmedic)
>- Efficient Multi-Scale 3D Convolutional Neural Network for Segmentation of 3D Medical Scans

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: convolutional-neural-networks, deep-learning, machine-learning, medical-imaging, neural-networks
>- Forks:	347 
>- Issues:	23
>- Watchers:	1058
>- Last updated: 2025-12-23

- [medicaltorch](https://github.com/perone/medicaltorch?tab=readme-ov-file)
>- A medical imaging framework for Pytorch

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: computer-vision, deep-learning, machine-learning, medical-imaging, python, pytorch
>- Forks:	128 
>- Issues:	17
>- Watchers:	870
>- Last updated: 2025-11-13

- [mriqc](https://github.com/nipreps/mriqc)
>- Automated Quality Control and visual reports for Quality Assessment of structural (T1w, T2w) and functional MRI of the brain

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: machine-learning, mri, neuroimaging, quality-control, quality-reporter
>- Forks:	134 
>- Issues:	94
>- Watchers:	343
>- Last updated: 2026-01-16

- [RadQy](https://github.com/ccipd/MRQy)
>- RadQy is a quality assurance and checking tool for quantitative assessment of magnetic resonance imaging (MRI) and computed tomography (CT) data.

>- License: None
>- Languages: `Javascript`
>- Tags: imaging, machine-learning, medical-image-processing, medical-imaging, medical-physics, mri, python, quality-assurance, quality-control, quality-metrics
>- Forks:	37 
>- Issues:	1
>- Watchers:	109
>- Last updated: 2026-01-13

- [quickNAT_pytorch](https://github.com/ai-med/quickNAT_pytorch)
>- PyTorch Implementation of QuickNAT and Bayesian QuickNAT, a fast brain MRI segmentation framework with segmentation Quality control using structure-wise uncertainty

>- License: MIT License
>- Languages: `Python`
>- Tags: ai, bayesian, biomarkers, brain-imaging, computer-vision, convolutional-neural-networks, deep-learning, machine-learning, medical-imaging, mri-images, neuroanatomy, pytorch, quality-control, segmentation, uncertainty
>- Forks:	36 
>- Issues:	7
>- Watchers:	104
>- Last updated: 2025-12-03

- [tensorflow-mri](https://github.com/mrphys/tensorflow-mri)
>- A Library of TensorFlow Operators for Computational MRI

>- License: Apache License 2.0
>- Languages: `Jupyter Notebook`
>- Tags: machine-learning, magnetic-resonance-imaging, ml, mri, python, tensorflow
>- Forks:	6 
>- Issues:	10
>- Watchers:	47
>- Last updated: 2025-11-25

- [nlsam](https://github.com/samuelstjean/nlsam)
>- The reference implementation for the Non Local Spatial and Angular Matching (NLSAM) denoising algorithm for diffusion MRI

>- License: GNU General Public License v3.0
>- Languages: `Python`
>- Tags: denoising-algorithm, diffusion-mri, machine-learning, python
>- Forks:	11 
>- Issues:	1
>- Watchers:	24
>- Last updated: 2025-11-17

### Medical-Image-Processing <a name="medical-image-processing"></a>
- [MONAI](https://github.com/Project-MONAI/MONAI)
>- AI Toolkit for Healthcare Imaging

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: deep-learning, healthcare-imaging, medical-image-computing, medical-image-processing, monai, python3, pytorch
>- Forks:	1403 
>- Issues:	505
>- Watchers:	7753
>- Last updated: 2026-01-16

- [torchio](https://github.com/fepegar/torchio)
>- Medical imaging processing for AI applications.

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: augmentation, data-augmentation, deep-learning, machine-learning, medical-image-analysis, medical-image-computing, medical-image-processing, medical-images, medical-imaging-datasets, medical-imaging-with-deep-learning, python, pytorch
>- Forks:	255 
>- Issues:	39
>- Watchers:	2345
>- Last updated: 2026-01-13

- [MedicalZooPytorch](https://github.com/black0017/MedicalZooPytorch)
>- A pytorch-based deep learning framework for multi-modal 2D/3D medical image segmentation

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-convolutional-network, brats2018, brats2019, deep-learning, densenet, iseg, iseg-challenge, medical-image-processing, medical-image-segmentation, medical-imaging, mrbrains18, pytorch, resnet, segmentation, segmentation-models, unet, unet-image-segmentation
>- Forks:	305 
>- Issues:	21
>- Watchers:	1893
>- Last updated: 2026-01-13

- [medicaldetectiontoolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit)
>- The Medical Detection Toolkit contains 2D + 3D implementations of prevalent object detectors such as Mask R-CNN, Retina Net, Retina U-Net, as well as a training and inference framework focused on dealing with medical images.  

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: 3d-mask-rcnn, 3d-models, 3d-object-detection, deep-learning, deep-neural-networks, detection, mask-rcnn, medical-image-analysis, medical-image-computing, medical-image-processing, medical-imaging, object-detection, pytorch-cnn, pytorch-deeplearning, pytorch-implementation, retina-net, retina-unet, segmentation, semantic-segmentation, u-net
>- Forks:	292 
>- Issues:	47
>- Watchers:	1345
>- Last updated: 2026-01-15

- [RadQy](https://github.com/ccipd/MRQy)
>- RadQy is a quality assurance and checking tool for quantitative assessment of magnetic resonance imaging (MRI) and computed tomography (CT) data.

>- License: None
>- Languages: `Javascript`
>- Tags: imaging, machine-learning, medical-image-processing, medical-imaging, medical-physics, mri, python, quality-assurance, quality-control, quality-metrics
>- Forks:	37 
>- Issues:	1
>- Watchers:	109
>- Last updated: 2026-01-13

- [DCEMRI.jl](https://github.com/davidssmith/DCEMRI.jl)
>- World's fastest DCE MRI analysis toolkit

>- License: Other
>- Languages: `Julia`
>- Tags: analysis, cancer-imaging-research, dce-mri, julia, mat-files, medical-image-processing, medical-imaging
>- Forks:	16 
>- Issues:	5
>- Watchers:	39
>- Last updated: 2026-01-12

- [sHDR](https://github.com/shakes76/sHDR)
>- HDR-MRI Algorithms from "Local contrast-enhanced MR images via high dynamic range processing" published in MRM

>- License: Other
>- Languages: `C++`
>- Tags: fusion, hdr, image, medical-image-processing, medical-imaging, mri
>- Forks:	0 
>- Issues:	0
>- Watchers:	16
>- Last updated: 2024-11-29

### Segmentation <a name="segmentation"></a>
- [Slicer](https://github.com/Slicer/Slicer)
>- Multi-platform, free open source software for visualization and image computing.

>- License: Other
>- Languages: `C++`
>- Tags: 3d-printing, 3d-slicer, c-plus-plus, computed-tomography, image-guided-therapy, image-processing, itk, kitware, medical-image-computing, medical-imaging, national-institutes-of-health, neuroimaging, nih, python, qt, registration, segmentation, tcia-dac, tractography, vtk
>- Forks:	685 
>- Issues:	647
>- Watchers:	2282
>- Last updated: 2026-01-16

- [MedicalZooPytorch](https://github.com/black0017/MedicalZooPytorch)
>- A pytorch-based deep learning framework for multi-modal 2D/3D medical image segmentation

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-convolutional-network, brats2018, brats2019, deep-learning, densenet, iseg, iseg-challenge, medical-image-processing, medical-image-segmentation, medical-imaging, mrbrains18, pytorch, resnet, segmentation, segmentation-models, unet, unet-image-segmentation
>- Forks:	305 
>- Issues:	21
>- Watchers:	1893
>- Last updated: 2026-01-13

- [medicaldetectiontoolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit)
>- The Medical Detection Toolkit contains 2D + 3D implementations of prevalent object detectors such as Mask R-CNN, Retina Net, Retina U-Net, as well as a training and inference framework focused on dealing with medical images.  

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: 3d-mask-rcnn, 3d-models, 3d-object-detection, deep-learning, deep-neural-networks, detection, mask-rcnn, medical-image-analysis, medical-image-computing, medical-image-processing, medical-imaging, object-detection, pytorch-cnn, pytorch-deeplearning, pytorch-implementation, retina-net, retina-unet, segmentation, semantic-segmentation, u-net
>- Forks:	292 
>- Issues:	47
>- Watchers:	1345
>- Last updated: 2026-01-15

- [SimpleITK](https://github.com/SimpleITK/SimpleITK)
>- SimpleITK: a layer built on top of the Insight Toolkit (ITK), intended to simplify and facilitate ITK's use in rapid prototyping, education and interpreted languages.

>- License: Apache License 2.0
>- Languages: `Swig`
>- Tags: c-plus-plus, csharp, image-analysis, image-processing, itk, java, lua, python, r, registration, ruby, segmentation, simpleitk, swig, tcl
>- Forks:	223 
>- Issues:	71
>- Watchers:	1032
>- Last updated: 2026-01-16

- [quickNAT_pytorch](https://github.com/ai-med/quickNAT_pytorch)
>- PyTorch Implementation of QuickNAT and Bayesian QuickNAT, a fast brain MRI segmentation framework with segmentation Quality control using structure-wise uncertainty

>- License: MIT License
>- Languages: `Python`
>- Tags: ai, bayesian, biomarkers, brain-imaging, computer-vision, convolutional-neural-networks, deep-learning, machine-learning, medical-imaging, mri-images, neuroanatomy, pytorch, quality-control, segmentation, uncertainty
>- Forks:	36 
>- Issues:	7
>- Watchers:	104
>- Last updated: 2025-12-03

- [BraTS-Toolkit](https://github.com/neuronflow/BraTS-Toolkit)
>- Code to preprocess, segment, and fuse glioma MRI scans based on the BraTS Toolkit manuscript.

>- License: GNU Affero General Public License v3.0
>- Languages: `Python`
>- Tags: glioblastoma, glioma, medical-imaging, mri, segmentation
>- Forks:	14 
>- Issues:	9
>- Watchers:	98
>- Last updated: 2026-01-13

- [NeSVoR](https://github.com/daviddmc/NeSVoR)
>- NeSVoR is a package for GPU-accelerated slice-to-volume reconstruction.

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-reconstruction, 3d-visualization, deep-learning, image-reconstruction, image-registration, implicit-neural-representation, medical-imaging, mri, nerf, neural-network, neural-rendering, pytorch, segmentation, super-resolution, transformers
>- Forks:	21 
>- Issues:	7
>- Watchers:	97
>- Last updated: 2026-01-07

### Brain-Imaging <a name="brain-imaging"></a>
- [nilearn](https://github.com/nilearn/nilearn)
>- Machine learning for NeuroImaging in Python

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: brain-connectivity, brain-imaging, brain-mri, decoding, fmri, machine-learning, mvpa, neuroimaging, python
>- Forks:	638 
>- Issues:	272
>- Watchers:	1353
>- Last updated: 2026-01-16

- [nipype](https://github.com/nipy/nipype)
>- Workflows and interfaces for neuroimaging packages

>- License: Other
>- Languages: `Python`
>- Tags: big-data, brain-imaging, brainweb, data-science, dataflow, dataflow-programming, neuroimaging, python, workflow-engine
>- Forks:	538 
>- Issues:	434
>- Watchers:	802
>- Last updated: 2026-01-14

- [nibabel](https://github.com/nipy/nibabel)
>- Python package to access a cacophony of neuro-imaging file formats

>- License: Other
>- Languages: `Python`
>- Tags: afni-brik-head, brain-imaging, cifti-2, data-formats, dicom, ecat, gifti, minc, neuroimaging, nifti, python, streamlines, tck, trk
>- Forks:	273 
>- Issues:	145
>- Watchers:	759
>- Last updated: 2026-01-15

- [brainGraph](https://github.com/cwatson/brainGraph)
>- Graph theory analysis of brain MRI data

>- License: None
>- Languages: `R`
>- Tags: brain-connectivity, brain-imaging, complex-networks, connectome, connectomics, fmri, graph-theory, mri, network-analysis, neuroimaging, neuroscience, r, statistics, tractography
>- Forks:	54 
>- Issues:	11
>- Watchers:	191
>- Last updated: 2025-11-13

- [clinicadl](https://github.com/aramis-lab/clinicadl)
>- Framework for the reproducible processing of neuroimaging data with deep learning methods

>- License: MIT License
>- Languages: `Python`
>- Tags: alzheimer-disease, brain-imaging, convolutional-neural-network, deep-learning, medical-imaging, neuroimaging, python, pytorch
>- Forks:	60 
>- Issues:	65
>- Watchers:	176
>- Last updated: 2025-12-31

- [quickNAT_pytorch](https://github.com/ai-med/quickNAT_pytorch)
>- PyTorch Implementation of QuickNAT and Bayesian QuickNAT, a fast brain MRI segmentation framework with segmentation Quality control using structure-wise uncertainty

>- License: MIT License
>- Languages: `Python`
>- Tags: ai, bayesian, biomarkers, brain-imaging, computer-vision, convolutional-neural-networks, deep-learning, machine-learning, medical-imaging, mri-images, neuroanatomy, pytorch, quality-control, segmentation, uncertainty
>- Forks:	36 
>- Issues:	7
>- Watchers:	104
>- Last updated: 2025-12-03

### Quality-Control <a name="quality-control"></a>
- [mriqc](https://github.com/nipreps/mriqc)
>- Automated Quality Control and visual reports for Quality Assessment of structural (T1w, T2w) and functional MRI of the brain

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: machine-learning, mri, neuroimaging, quality-control, quality-reporter
>- Forks:	134 
>- Issues:	94
>- Watchers:	343
>- Last updated: 2026-01-16

- [RadQy](https://github.com/ccipd/MRQy)
>- RadQy is a quality assurance and checking tool for quantitative assessment of magnetic resonance imaging (MRI) and computed tomography (CT) data.

>- License: None
>- Languages: `Javascript`
>- Tags: imaging, machine-learning, medical-image-processing, medical-imaging, medical-physics, mri, python, quality-assurance, quality-control, quality-metrics
>- Forks:	37 
>- Issues:	1
>- Watchers:	109
>- Last updated: 2026-01-13

- [quickNAT_pytorch](https://github.com/ai-med/quickNAT_pytorch)
>- PyTorch Implementation of QuickNAT and Bayesian QuickNAT, a fast brain MRI segmentation framework with segmentation Quality control using structure-wise uncertainty

>- License: MIT License
>- Languages: `Python`
>- Tags: ai, bayesian, biomarkers, brain-imaging, computer-vision, convolutional-neural-networks, deep-learning, machine-learning, medical-imaging, mri-images, neuroanatomy, pytorch, quality-control, segmentation, uncertainty
>- Forks:	36 
>- Issues:	7
>- Watchers:	104
>- Last updated: 2025-12-03

- [mrQA](https://github.com/Open-Minds-Lab/mrQA)
>- mrQA: tools for quality assurance in medical imaging datasets, including protocol compliance

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: brain, ismrm, mr-image, mri, mri-brain, mri-images, neuroimaging, neuroscience, niqc, qa, quality-assurance, quality-control
>- Forks:	6 
>- Issues:	38
>- Watchers:	13
>- Last updated: 2026-01-14

- [fetal-IQA](https://github.com/daviddmc/fetal-IQA)
>- Image quality assessment for fetal MRI

>- License: MIT License
>- Languages: `Python`
>- Tags: convolutional-neural-networks, deep-learning, fetal-mri, medical-imaging, pytorch, quality-control, semi-supervised-learning, tensorflow
>- Forks:	0 
>- Issues:	0
>- Watchers:	6
>- Last updated: 2024-10-12

### Convolutional-Neural-Networks <a name="convolutional-neural-networks"></a>
- [fastMRI](https://github.com/facebookresearch/fastMRI)
>- A large-scale dataset of both raw MRI measurements and clinical MRI images.

>- License: MIT License
>- Languages: `Python`
>- Tags: convolutional-neural-networks, deep-learning, fastmri, fastmri-challenge, fastmri-dataset, medical-imaging, mri, mri-reconstruction, pytorch
>- Forks:	416 
>- Issues:	18
>- Watchers:	1498
>- Last updated: 2026-01-10

- [deepmedic](https://github.com/deepmedic/deepmedic)
>- Efficient Multi-Scale 3D Convolutional Neural Network for Segmentation of 3D Medical Scans

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: convolutional-neural-networks, deep-learning, machine-learning, medical-imaging, neural-networks
>- Forks:	347 
>- Issues:	23
>- Watchers:	1058
>- Last updated: 2025-12-23

- [quickNAT_pytorch](https://github.com/ai-med/quickNAT_pytorch)
>- PyTorch Implementation of QuickNAT and Bayesian QuickNAT, a fast brain MRI segmentation framework with segmentation Quality control using structure-wise uncertainty

>- License: MIT License
>- Languages: `Python`
>- Tags: ai, bayesian, biomarkers, brain-imaging, computer-vision, convolutional-neural-networks, deep-learning, machine-learning, medical-imaging, mri-images, neuroanatomy, pytorch, quality-control, segmentation, uncertainty
>- Forks:	36 
>- Issues:	7
>- Watchers:	104
>- Last updated: 2025-12-03

- [fetal-IQA](https://github.com/daviddmc/fetal-IQA)
>- Image quality assessment for fetal MRI

>- License: MIT License
>- Languages: `Python`
>- Tags: convolutional-neural-networks, deep-learning, fetal-mri, medical-imaging, pytorch, quality-control, semi-supervised-learning, tensorflow
>- Forks:	0 
>- Issues:	0
>- Watchers:	6
>- Last updated: 2024-10-12

### Medical-Image-Computing <a name="medical-image-computing"></a>
- [MONAI](https://github.com/Project-MONAI/MONAI)
>- AI Toolkit for Healthcare Imaging

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: deep-learning, healthcare-imaging, medical-image-computing, medical-image-processing, monai, python3, pytorch
>- Forks:	1403 
>- Issues:	505
>- Watchers:	7753
>- Last updated: 2026-01-16

- [torchio](https://github.com/fepegar/torchio)
>- Medical imaging processing for AI applications.

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: augmentation, data-augmentation, deep-learning, machine-learning, medical-image-analysis, medical-image-computing, medical-image-processing, medical-images, medical-imaging-datasets, medical-imaging-with-deep-learning, python, pytorch
>- Forks:	255 
>- Issues:	39
>- Watchers:	2345
>- Last updated: 2026-01-13

- [Slicer](https://github.com/Slicer/Slicer)
>- Multi-platform, free open source software for visualization and image computing.

>- License: Other
>- Languages: `C++`
>- Tags: 3d-printing, 3d-slicer, c-plus-plus, computed-tomography, image-guided-therapy, image-processing, itk, kitware, medical-image-computing, medical-imaging, national-institutes-of-health, neuroimaging, nih, python, qt, registration, segmentation, tcia-dac, tractography, vtk
>- Forks:	685 
>- Issues:	647
>- Watchers:	2282
>- Last updated: 2026-01-16

- [medicaldetectiontoolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit)
>- The Medical Detection Toolkit contains 2D + 3D implementations of prevalent object detectors such as Mask R-CNN, Retina Net, Retina U-Net, as well as a training and inference framework focused on dealing with medical images.  

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: 3d-mask-rcnn, 3d-models, 3d-object-detection, deep-learning, deep-neural-networks, detection, mask-rcnn, medical-image-analysis, medical-image-computing, medical-image-processing, medical-imaging, object-detection, pytorch-cnn, pytorch-deeplearning, pytorch-implementation, retina-net, retina-unet, segmentation, semantic-segmentation, u-net
>- Forks:	292 
>- Issues:	47
>- Watchers:	1345
>- Last updated: 2026-01-15

### Mri-Images <a name="mri-images"></a>
- [quickNAT_pytorch](https://github.com/ai-med/quickNAT_pytorch)
>- PyTorch Implementation of QuickNAT and Bayesian QuickNAT, a fast brain MRI segmentation framework with segmentation Quality control using structure-wise uncertainty

>- License: MIT License
>- Languages: `Python`
>- Tags: ai, bayesian, biomarkers, brain-imaging, computer-vision, convolutional-neural-networks, deep-learning, machine-learning, medical-imaging, mri-images, neuroanatomy, pytorch, quality-control, segmentation, uncertainty
>- Forks:	36 
>- Issues:	7
>- Watchers:	104
>- Last updated: 2025-12-03

- [NIfTI.jl](https://github.com/JuliaNeuroscience/NIfTI.jl)
>- Julia module for reading/writing NIfTI MRI files

>- License: Other
>- Languages: `Julia`
>- Tags: fmri, julia, mri, mri-images, nifti
>- Forks:	35 
>- Issues:	30
>- Watchers:	82
>- Last updated: 2025-12-27

- [MriResearchTools.jl](https://github.com/korbinian90/MriResearchTools.jl)
>- Specialized tools for MRI

>- License: MIT License
>- Languages: `Julia`
>- Tags: mri, mri-images
>- Forks:	8 
>- Issues:	3
>- Watchers:	26
>- Last updated: 2026-01-08

- [mrQA](https://github.com/Open-Minds-Lab/mrQA)
>- mrQA: tools for quality assurance in medical imaging datasets, including protocol compliance

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: brain, ismrm, mr-image, mri, mri-brain, mri-images, neuroimaging, neuroscience, niqc, qa, quality-assurance, quality-control
>- Forks:	6 
>- Issues:	38
>- Watchers:	13
>- Last updated: 2026-01-14

### Diffusion-Mri <a name="diffusion-mri"></a>
- [KomaMRI.jl](https://github.com/JuliaHealth/KomaMRI.jl)
>- Koma is a Pulseq-compatible framework to efficiently simulate Magnetic Resonance Imaging (MRI) acquisitions. The main focus of this package is to simulate general scenarios that could arise in pulse sequence development.

>- License: MIT License
>- Languages: `Julia`
>- Tags: cardiac, diffusion, diffusion-mri, gpu-acceleration, mri, simulation
>- Forks:	31 
>- Issues:	103
>- Watchers:	176
>- Last updated: 2026-01-06

- [qsiprep](https://github.com/PennLINC/qsiprep)
>- Preprocessing of diffusion MRI

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: bids, denoising-images, diffusion-mri, distortion-correction, motion-correction, pipelines
>- Forks:	62 
>- Issues:	105
>- Watchers:	172
>- Last updated: 2026-01-16

- [disimpy](https://github.com/kerkelae/disimpy)
>- Massively parallel Monte Carlo diffusion MR simulator written in Python.

>- License: MIT License
>- Languages: `Python`
>- Tags: cuda, diffusion-mri, gpu-computing, monte-carlo-simulation
>- Forks:	9 
>- Issues:	5
>- Watchers:	27
>- Last updated: 2026-01-13

- [nlsam](https://github.com/samuelstjean/nlsam)
>- The reference implementation for the Non Local Spatial and Angular Matching (NLSAM) denoising algorithm for diffusion MRI

>- License: GNU General Public License v3.0
>- Languages: `Python`
>- Tags: denoising-algorithm, diffusion-mri, machine-learning, python
>- Forks:	11 
>- Issues:	1
>- Watchers:	24
>- Last updated: 2025-11-17

### Image-Processing <a name="image-processing"></a>
- [Slicer](https://github.com/Slicer/Slicer)
>- Multi-platform, free open source software for visualization and image computing.

>- License: Other
>- Languages: `C++`
>- Tags: 3d-printing, 3d-slicer, c-plus-plus, computed-tomography, image-guided-therapy, image-processing, itk, kitware, medical-image-computing, medical-imaging, national-institutes-of-health, neuroimaging, nih, python, qt, registration, segmentation, tcia-dac, tractography, vtk
>- Forks:	685 
>- Issues:	647
>- Watchers:	2282
>- Last updated: 2026-01-16

- [SimpleITK](https://github.com/SimpleITK/SimpleITK)
>- SimpleITK: a layer built on top of the Insight Toolkit (ITK), intended to simplify and facilitate ITK's use in rapid prototyping, education and interpreted languages.

>- License: Apache License 2.0
>- Languages: `Swig`
>- Tags: c-plus-plus, csharp, image-analysis, image-processing, itk, java, lua, python, r, registration, ruby, segmentation, simpleitk, swig, tcl
>- Forks:	223 
>- Issues:	71
>- Watchers:	1032
>- Last updated: 2026-01-16

- [smriprep](https://github.com/nipreps/smriprep)
>- Structural MRI PREProcessing (sMRIPrep) workflows for NIPreps (NeuroImaging PREProcessing tools)

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: image-processing, image-registration, image-segmentation, mri, structural-mri, surface-reconstruction
>- Forks:	47 
>- Issues:	73
>- Watchers:	162
>- Last updated: 2026-01-15

- [hazen](https://github.com/GSTT-CSC/hazen)
>- Quality assurance framework for Magnetic Resonance Imaging

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: image-processing, mri, mri-phantoms, python, qa, quality-assurance
>- Forks:	13 
>- Issues:	58
>- Watchers:	29
>- Last updated: 2026-01-14

### Quality-Assurance <a name="quality-assurance"></a>
- [RadQy](https://github.com/ccipd/MRQy)
>- RadQy is a quality assurance and checking tool for quantitative assessment of magnetic resonance imaging (MRI) and computed tomography (CT) data.

>- License: None
>- Languages: `Javascript`
>- Tags: imaging, machine-learning, medical-image-processing, medical-imaging, medical-physics, mri, python, quality-assurance, quality-control, quality-metrics
>- Forks:	37 
>- Issues:	1
>- Watchers:	109
>- Last updated: 2026-01-13

- [hazen](https://github.com/GSTT-CSC/hazen)
>- Quality assurance framework for Magnetic Resonance Imaging

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: image-processing, mri, mri-phantoms, python, qa, quality-assurance
>- Forks:	13 
>- Issues:	58
>- Watchers:	29
>- Last updated: 2026-01-14

- [mrQA](https://github.com/Open-Minds-Lab/mrQA)
>- mrQA: tools for quality assurance in medical imaging datasets, including protocol compliance

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: brain, ismrm, mr-image, mri, mri-brain, mri-images, neuroimaging, neuroscience, niqc, qa, quality-assurance, quality-control
>- Forks:	6 
>- Issues:	38
>- Watchers:	13
>- Last updated: 2026-01-14

### Itk <a name="itk"></a>
- [Slicer](https://github.com/Slicer/Slicer)
>- Multi-platform, free open source software for visualization and image computing.

>- License: Other
>- Languages: `C++`
>- Tags: 3d-printing, 3d-slicer, c-plus-plus, computed-tomography, image-guided-therapy, image-processing, itk, kitware, medical-image-computing, medical-imaging, national-institutes-of-health, neuroimaging, nih, python, qt, registration, segmentation, tcia-dac, tractography, vtk
>- Forks:	685 
>- Issues:	647
>- Watchers:	2282
>- Last updated: 2026-01-16

- [SimpleITK](https://github.com/SimpleITK/SimpleITK)
>- SimpleITK: a layer built on top of the Insight Toolkit (ITK), intended to simplify and facilitate ITK's use in rapid prototyping, education and interpreted languages.

>- License: Apache License 2.0
>- Languages: `Swig`
>- Tags: c-plus-plus, csharp, image-analysis, image-processing, itk, java, lua, python, r, registration, ruby, segmentation, simpleitk, swig, tcl
>- Forks:	223 
>- Issues:	71
>- Watchers:	1032
>- Last updated: 2026-01-16

- [mialsuperresolutiontoolkit](https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit)
>- The Medical Image Analysis Laboratory Super-Resolution ToolKit (MIALSRTK) consists of a set of C++ and Python processing and workflow tools necessary to perform motion-robust super-resolution fetal MRI reconstruction in the BIDS Apps framework.

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `C++`
>- Tags: bids, bids-apps, fetal, itk, mri, nipype, super-resolution, workflow
>- Forks:	15 
>- Issues:	17
>- Watchers:	30
>- Last updated: 2025-12-30

### Fmri <a name="fmri"></a>
- [nilearn](https://github.com/nilearn/nilearn)
>- Machine learning for NeuroImaging in Python

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: brain-connectivity, brain-imaging, brain-mri, decoding, fmri, machine-learning, mvpa, neuroimaging, python
>- Forks:	638 
>- Issues:	272
>- Watchers:	1353
>- Last updated: 2026-01-16

- [brainGraph](https://github.com/cwatson/brainGraph)
>- Graph theory analysis of brain MRI data

>- License: None
>- Languages: `R`
>- Tags: brain-connectivity, brain-imaging, complex-networks, connectome, connectomics, fmri, graph-theory, mri, network-analysis, neuroimaging, neuroscience, r, statistics, tractography
>- Forks:	54 
>- Issues:	11
>- Watchers:	191
>- Last updated: 2025-11-13

- [NIfTI.jl](https://github.com/JuliaNeuroscience/NIfTI.jl)
>- Julia module for reading/writing NIfTI MRI files

>- License: Other
>- Languages: `Julia`
>- Tags: fmri, julia, mri, mri-images, nifti
>- Forks:	35 
>- Issues:	30
>- Watchers:	82
>- Last updated: 2025-12-27

### Julia <a name="julia"></a>
- [NIfTI.jl](https://github.com/JuliaNeuroscience/NIfTI.jl)
>- Julia module for reading/writing NIfTI MRI files

>- License: Other
>- Languages: `Julia`
>- Tags: fmri, julia, mri, mri-images, nifti
>- Forks:	35 
>- Issues:	30
>- Watchers:	82
>- Last updated: 2025-12-27

- [DCEMRI.jl](https://github.com/davidssmith/DCEMRI.jl)
>- World's fastest DCE MRI analysis toolkit

>- License: Other
>- Languages: `Julia`
>- Tags: analysis, cancer-imaging-research, dce-mri, julia, mat-files, medical-image-processing, medical-imaging
>- Forks:	16 
>- Issues:	5
>- Watchers:	39
>- Last updated: 2026-01-12

### Fetal <a name="fetal"></a>
- [SVRTK](https://github.com/SVRTK/SVRTK)
>- MIRTK based SVR reconstruction

>- License: Apache License 2.0
>- Languages: `C++`
>- Tags: fetal, mri, reconstruction, retrospecitve, slice-to-volume
>- Forks:	8 
>- Issues:	5
>- Watchers:	63
>- Last updated: 2025-12-25

- [mialsuperresolutiontoolkit](https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit)
>- The Medical Image Analysis Laboratory Super-Resolution ToolKit (MIALSRTK) consists of a set of C++ and Python processing and workflow tools necessary to perform motion-robust super-resolution fetal MRI reconstruction in the BIDS Apps framework.

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `C++`
>- Tags: bids, bids-apps, fetal, itk, mri, nipype, super-resolution, workflow
>- Forks:	15 
>- Issues:	17
>- Watchers:	30
>- Last updated: 2025-12-30

### Fastmri-Challenge <a name="fastmri-challenge"></a>
- [fastMRI](https://github.com/facebookresearch/fastMRI)
>- A large-scale dataset of both raw MRI measurements and clinical MRI images.

>- License: MIT License
>- Languages: `Python`
>- Tags: convolutional-neural-networks, deep-learning, fastmri, fastmri-challenge, fastmri-dataset, medical-imaging, mri, mri-reconstruction, pytorch
>- Forks:	416 
>- Issues:	18
>- Watchers:	1498
>- Last updated: 2026-01-10

- [direct](https://github.com/NKI-AI/direct)
>- Deep learning framework for MRI reconstruction

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: deep-learning, fastmri-challenge, inverse-problems, medical-imaging, mri-reconstruction, pytorch
>- Forks:	47 
>- Issues:	6
>- Watchers:	294
>- Last updated: 2026-01-09

### Mri-Reconstruction <a name="mri-reconstruction"></a>
- [fastMRI](https://github.com/facebookresearch/fastMRI)
>- A large-scale dataset of both raw MRI measurements and clinical MRI images.

>- License: MIT License
>- Languages: `Python`
>- Tags: convolutional-neural-networks, deep-learning, fastmri, fastmri-challenge, fastmri-dataset, medical-imaging, mri, mri-reconstruction, pytorch
>- Forks:	416 
>- Issues:	18
>- Watchers:	1498
>- Last updated: 2026-01-10

- [direct](https://github.com/NKI-AI/direct)
>- Deep learning framework for MRI reconstruction

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: deep-learning, fastmri-challenge, inverse-problems, medical-imaging, mri-reconstruction, pytorch
>- Forks:	47 
>- Issues:	6
>- Watchers:	294
>- Last updated: 2026-01-09

### Tensorflow <a name="tensorflow"></a>
- [tensorflow-mri](https://github.com/mrphys/tensorflow-mri)
>- A Library of TensorFlow Operators for Computational MRI

>- License: Apache License 2.0
>- Languages: `Jupyter Notebook`
>- Tags: machine-learning, magnetic-resonance-imaging, ml, mri, python, tensorflow
>- Forks:	6 
>- Issues:	10
>- Watchers:	47
>- Last updated: 2025-11-25

- [fetal-IQA](https://github.com/daviddmc/fetal-IQA)
>- Image quality assessment for fetal MRI

>- License: MIT License
>- Languages: `Python`
>- Tags: convolutional-neural-networks, deep-learning, fetal-mri, medical-imaging, pytorch, quality-control, semi-supervised-learning, tensorflow
>- Forks:	0 
>- Issues:	0
>- Watchers:	6
>- Last updated: 2024-10-12

### Medical-Image-Analysis <a name="medical-image-analysis"></a>
- [torchio](https://github.com/fepegar/torchio)
>- Medical imaging processing for AI applications.

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: augmentation, data-augmentation, deep-learning, machine-learning, medical-image-analysis, medical-image-computing, medical-image-processing, medical-images, medical-imaging-datasets, medical-imaging-with-deep-learning, python, pytorch
>- Forks:	255 
>- Issues:	39
>- Watchers:	2345
>- Last updated: 2026-01-13

- [medicaldetectiontoolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit)
>- The Medical Detection Toolkit contains 2D + 3D implementations of prevalent object detectors such as Mask R-CNN, Retina Net, Retina U-Net, as well as a training and inference framework focused on dealing with medical images.  

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: 3d-mask-rcnn, 3d-models, 3d-object-detection, deep-learning, deep-neural-networks, detection, mask-rcnn, medical-image-analysis, medical-image-computing, medical-image-processing, medical-imaging, object-detection, pytorch-cnn, pytorch-deeplearning, pytorch-implementation, retina-net, retina-unet, segmentation, semantic-segmentation, u-net
>- Forks:	292 
>- Issues:	47
>- Watchers:	1345
>- Last updated: 2026-01-15

### Neuroscience <a name="neuroscience"></a>
- [brainGraph](https://github.com/cwatson/brainGraph)
>- Graph theory analysis of brain MRI data

>- License: None
>- Languages: `R`
>- Tags: brain-connectivity, brain-imaging, complex-networks, connectome, connectomics, fmri, graph-theory, mri, network-analysis, neuroimaging, neuroscience, r, statistics, tractography
>- Forks:	54 
>- Issues:	11
>- Watchers:	191
>- Last updated: 2025-11-13

- [mrQA](https://github.com/Open-Minds-Lab/mrQA)
>- mrQA: tools for quality assurance in medical imaging datasets, including protocol compliance

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: brain, ismrm, mr-image, mri, mri-brain, mri-images, neuroimaging, neuroscience, niqc, qa, quality-assurance, quality-control
>- Forks:	6 
>- Issues:	38
>- Watchers:	13
>- Last updated: 2026-01-14

### Qa <a name="qa"></a>
- [hazen](https://github.com/GSTT-CSC/hazen)
>- Quality assurance framework for Magnetic Resonance Imaging

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: image-processing, mri, mri-phantoms, python, qa, quality-assurance
>- Forks:	13 
>- Issues:	58
>- Watchers:	29
>- Last updated: 2026-01-14

- [mrQA](https://github.com/Open-Minds-Lab/mrQA)
>- mrQA: tools for quality assurance in medical imaging datasets, including protocol compliance

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: brain, ismrm, mr-image, mri, mri-brain, mri-images, neuroimaging, neuroscience, niqc, qa, quality-assurance, quality-control
>- Forks:	6 
>- Issues:	38
>- Watchers:	13
>- Last updated: 2026-01-14

### Bids <a name="bids"></a>
- [qsiprep](https://github.com/PennLINC/qsiprep)
>- Preprocessing of diffusion MRI

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: bids, denoising-images, diffusion-mri, distortion-correction, motion-correction, pipelines
>- Forks:	62 
>- Issues:	105
>- Watchers:	172
>- Last updated: 2026-01-16

- [mialsuperresolutiontoolkit](https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit)
>- The Medical Image Analysis Laboratory Super-Resolution ToolKit (MIALSRTK) consists of a set of C++ and Python processing and workflow tools necessary to perform motion-robust super-resolution fetal MRI reconstruction in the BIDS Apps framework.

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `C++`
>- Tags: bids, bids-apps, fetal, itk, mri, nipype, super-resolution, workflow
>- Forks:	15 
>- Issues:	17
>- Watchers:	30
>- Last updated: 2025-12-30

### Super-Resolution <a name="super-resolution"></a>
- [NeSVoR](https://github.com/daviddmc/NeSVoR)
>- NeSVoR is a package for GPU-accelerated slice-to-volume reconstruction.

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-reconstruction, 3d-visualization, deep-learning, image-reconstruction, image-registration, implicit-neural-representation, medical-imaging, mri, nerf, neural-network, neural-rendering, pytorch, segmentation, super-resolution, transformers
>- Forks:	21 
>- Issues:	7
>- Watchers:	97
>- Last updated: 2026-01-07

- [mialsuperresolutiontoolkit](https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit)
>- The Medical Image Analysis Laboratory Super-Resolution ToolKit (MIALSRTK) consists of a set of C++ and Python processing and workflow tools necessary to perform motion-robust super-resolution fetal MRI reconstruction in the BIDS Apps framework.

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `C++`
>- Tags: bids, bids-apps, fetal, itk, mri, nipype, super-resolution, workflow
>- Forks:	15 
>- Issues:	17
>- Watchers:	30
>- Last updated: 2025-12-30

### Brain-Connectivity <a name="brain-connectivity"></a>
- [nilearn](https://github.com/nilearn/nilearn)
>- Machine learning for NeuroImaging in Python

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: brain-connectivity, brain-imaging, brain-mri, decoding, fmri, machine-learning, mvpa, neuroimaging, python
>- Forks:	638 
>- Issues:	272
>- Watchers:	1353
>- Last updated: 2026-01-16

- [brainGraph](https://github.com/cwatson/brainGraph)
>- Graph theory analysis of brain MRI data

>- License: None
>- Languages: `R`
>- Tags: brain-connectivity, brain-imaging, complex-networks, connectome, connectomics, fmri, graph-theory, mri, network-analysis, neuroimaging, neuroscience, r, statistics, tractography
>- Forks:	54 
>- Issues:	11
>- Watchers:	191
>- Last updated: 2025-11-13

### R <a name="r"></a>
- [SimpleITK](https://github.com/SimpleITK/SimpleITK)
>- SimpleITK: a layer built on top of the Insight Toolkit (ITK), intended to simplify and facilitate ITK's use in rapid prototyping, education and interpreted languages.

>- License: Apache License 2.0
>- Languages: `Swig`
>- Tags: c-plus-plus, csharp, image-analysis, image-processing, itk, java, lua, python, r, registration, ruby, segmentation, simpleitk, swig, tcl
>- Forks:	223 
>- Issues:	71
>- Watchers:	1032
>- Last updated: 2026-01-16

- [brainGraph](https://github.com/cwatson/brainGraph)
>- Graph theory analysis of brain MRI data

>- License: None
>- Languages: `R`
>- Tags: brain-connectivity, brain-imaging, complex-networks, connectome, connectomics, fmri, graph-theory, mri, network-analysis, neuroimaging, neuroscience, r, statistics, tractography
>- Forks:	54 
>- Issues:	11
>- Watchers:	191
>- Last updated: 2025-11-13

### Tractography <a name="tractography"></a>
- [Slicer](https://github.com/Slicer/Slicer)
>- Multi-platform, free open source software for visualization and image computing.

>- License: Other
>- Languages: `C++`
>- Tags: 3d-printing, 3d-slicer, c-plus-plus, computed-tomography, image-guided-therapy, image-processing, itk, kitware, medical-image-computing, medical-imaging, national-institutes-of-health, neuroimaging, nih, python, qt, registration, segmentation, tcia-dac, tractography, vtk
>- Forks:	685 
>- Issues:	647
>- Watchers:	2282
>- Last updated: 2026-01-16

- [brainGraph](https://github.com/cwatson/brainGraph)
>- Graph theory analysis of brain MRI data

>- License: None
>- Languages: `R`
>- Tags: brain-connectivity, brain-imaging, complex-networks, connectome, connectomics, fmri, graph-theory, mri, network-analysis, neuroimaging, neuroscience, r, statistics, tractography
>- Forks:	54 
>- Issues:	11
>- Watchers:	191
>- Last updated: 2025-11-13

### Image-Registration <a name="image-registration"></a>
- [smriprep](https://github.com/nipreps/smriprep)
>- Structural MRI PREProcessing (sMRIPrep) workflows for NIPreps (NeuroImaging PREProcessing tools)

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: image-processing, image-registration, image-segmentation, mri, structural-mri, surface-reconstruction
>- Forks:	47 
>- Issues:	73
>- Watchers:	162
>- Last updated: 2026-01-15

- [NeSVoR](https://github.com/daviddmc/NeSVoR)
>- NeSVoR is a package for GPU-accelerated slice-to-volume reconstruction.

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-reconstruction, 3d-visualization, deep-learning, image-reconstruction, image-registration, implicit-neural-representation, medical-imaging, mri, nerf, neural-network, neural-rendering, pytorch, segmentation, super-resolution, transformers
>- Forks:	21 
>- Issues:	7
>- Watchers:	97
>- Last updated: 2026-01-07

### Magnetic-Resonance-Imaging <a name="magnetic-resonance-imaging"></a>
- [tensorflow-mri](https://github.com/mrphys/tensorflow-mri)
>- A Library of TensorFlow Operators for Computational MRI

>- License: Apache License 2.0
>- Languages: `Jupyter Notebook`
>- Tags: machine-learning, magnetic-resonance-imaging, ml, mri, python, tensorflow
>- Forks:	6 
>- Issues:	10
>- Watchers:	47
>- Last updated: 2025-11-25

- [pyCoilGen](https://github.com/kev-m/pyCoilGen)
>- Magnetic Field Coil Generator for Python, ported from CoilGen

>- License: GNU General Public License v3.0
>- Languages: `Python`
>- Tags: magnetic-field-solver, magnetic-resonance-imaging, medical-physics, mri, nmr, physics
>- Forks:	8 
>- Issues:	3
>- Watchers:	20
>- Last updated: 2025-12-03

### C-Plus-Plus <a name="c-plus-plus"></a>
- [Slicer](https://github.com/Slicer/Slicer)
>- Multi-platform, free open source software for visualization and image computing.

>- License: Other
>- Languages: `C++`
>- Tags: 3d-printing, 3d-slicer, c-plus-plus, computed-tomography, image-guided-therapy, image-processing, itk, kitware, medical-image-computing, medical-imaging, national-institutes-of-health, neuroimaging, nih, python, qt, registration, segmentation, tcia-dac, tractography, vtk
>- Forks:	685 
>- Issues:	647
>- Watchers:	2282
>- Last updated: 2026-01-16

- [SimpleITK](https://github.com/SimpleITK/SimpleITK)
>- SimpleITK: a layer built on top of the Insight Toolkit (ITK), intended to simplify and facilitate ITK's use in rapid prototyping, education and interpreted languages.

>- License: Apache License 2.0
>- Languages: `Swig`
>- Tags: c-plus-plus, csharp, image-analysis, image-processing, itk, java, lua, python, r, registration, ruby, segmentation, simpleitk, swig, tcl
>- Forks:	223 
>- Issues:	71
>- Watchers:	1032
>- Last updated: 2026-01-16

### Registration <a name="registration"></a>
- [Slicer](https://github.com/Slicer/Slicer)
>- Multi-platform, free open source software for visualization and image computing.

>- License: Other
>- Languages: `C++`
>- Tags: 3d-printing, 3d-slicer, c-plus-plus, computed-tomography, image-guided-therapy, image-processing, itk, kitware, medical-image-computing, medical-imaging, national-institutes-of-health, neuroimaging, nih, python, qt, registration, segmentation, tcia-dac, tractography, vtk
>- Forks:	685 
>- Issues:	647
>- Watchers:	2282
>- Last updated: 2026-01-16

- [SimpleITK](https://github.com/SimpleITK/SimpleITK)
>- SimpleITK: a layer built on top of the Insight Toolkit (ITK), intended to simplify and facilitate ITK's use in rapid prototyping, education and interpreted languages.

>- License: Apache License 2.0
>- Languages: `Swig`
>- Tags: c-plus-plus, csharp, image-analysis, image-processing, itk, java, lua, python, r, registration, ruby, segmentation, simpleitk, swig, tcl
>- Forks:	223 
>- Issues:	71
>- Watchers:	1032
>- Last updated: 2026-01-16

### Simulation <a name="simulation"></a>
- [KomaMRI.jl](https://github.com/JuliaHealth/KomaMRI.jl)
>- Koma is a Pulseq-compatible framework to efficiently simulate Magnetic Resonance Imaging (MRI) acquisitions. The main focus of this package is to simulate general scenarios that could arise in pulse sequence development.

>- License: MIT License
>- Languages: `Julia`
>- Tags: cardiac, diffusion, diffusion-mri, gpu-acceleration, mri, simulation
>- Forks:	31 
>- Issues:	103
>- Watchers:	176
>- Last updated: 2026-01-06

- [vespa](https://github.com/vespa-mrs/vespa)
>- Python tools for Magnetic Resonance Spectroscopy - Pulses, Simulation and Analysis

>- License: Other
>- Languages: `Python`
>- Tags: fitting, mrs, python, rf-pulse, simulation, spectroscopy, wxpython
>- Forks:	6 
>- Issues:	6
>- Watchers:	11
>- Last updated: 2025-11-12

### Nifti <a name="nifti"></a>
- [nibabel](https://github.com/nipy/nibabel)
>- Python package to access a cacophony of neuro-imaging file formats

>- License: Other
>- Languages: `Python`
>- Tags: afni-brik-head, brain-imaging, cifti-2, data-formats, dicom, ecat, gifti, minc, neuroimaging, nifti, python, streamlines, tck, trk
>- Forks:	273 
>- Issues:	145
>- Watchers:	759
>- Last updated: 2026-01-15

- [NIfTI.jl](https://github.com/JuliaNeuroscience/NIfTI.jl)
>- Julia module for reading/writing NIfTI MRI files

>- License: Other
>- Languages: `Julia`
>- Tags: fmri, julia, mri, mri-images, nifti
>- Forks:	35 
>- Issues:	30
>- Watchers:	82
>- Last updated: 2025-12-27

### Medical-Images <a name="medical-images"></a>
- [torchio](https://github.com/fepegar/torchio)
>- Medical imaging processing for AI applications.

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: augmentation, data-augmentation, deep-learning, machine-learning, medical-image-analysis, medical-image-computing, medical-image-processing, medical-images, medical-imaging-datasets, medical-imaging-with-deep-learning, python, pytorch
>- Forks:	255 
>- Issues:	39
>- Watchers:	2345
>- Last updated: 2026-01-13

- [mritopng](https://github.com/danishm/mritopng)
>- A simple python module to make it easy to batch convert DICOM files to PNG images.

>- License: MIT License
>- Languages: `Python`
>- Tags: dicom, dicom-converter, dicom-images, medical, medical-images, png, python
>- Forks:	51 
>- Issues:	5
>- Watchers:	146
>- Last updated: 2025-10-31

### Medical-Physics <a name="medical-physics"></a>
- [RadQy](https://github.com/ccipd/MRQy)
>- RadQy is a quality assurance and checking tool for quantitative assessment of magnetic resonance imaging (MRI) and computed tomography (CT) data.

>- License: None
>- Languages: `Javascript`
>- Tags: imaging, machine-learning, medical-image-processing, medical-imaging, medical-physics, mri, python, quality-assurance, quality-control, quality-metrics
>- Forks:	37 
>- Issues:	1
>- Watchers:	109
>- Last updated: 2026-01-13

- [pyCoilGen](https://github.com/kev-m/pyCoilGen)
>- Magnetic Field Coil Generator for Python, ported from CoilGen

>- License: GNU General Public License v3.0
>- Languages: `Python`
>- Tags: magnetic-field-solver, magnetic-resonance-imaging, medical-physics, mri, nmr, physics
>- Forks:	8 
>- Issues:	3
>- Watchers:	20
>- Last updated: 2025-12-03

### Computer-Vision <a name="computer-vision"></a>
- [medicaltorch](https://github.com/perone/medicaltorch?tab=readme-ov-file)
>- A medical imaging framework for Pytorch

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: computer-vision, deep-learning, machine-learning, medical-imaging, python, pytorch
>- Forks:	128 
>- Issues:	17
>- Watchers:	870
>- Last updated: 2025-11-13

- [quickNAT_pytorch](https://github.com/ai-med/quickNAT_pytorch)
>- PyTorch Implementation of QuickNAT and Bayesian QuickNAT, a fast brain MRI segmentation framework with segmentation Quality control using structure-wise uncertainty

>- License: MIT License
>- Languages: `Python`
>- Tags: ai, bayesian, biomarkers, brain-imaging, computer-vision, convolutional-neural-networks, deep-learning, machine-learning, medical-imaging, mri-images, neuroanatomy, pytorch, quality-control, segmentation, uncertainty
>- Forks:	36 
>- Issues:	7
>- Watchers:	104
>- Last updated: 2025-12-03

### Image-Reconstruction <a name="image-reconstruction"></a>
- [NeSVoR](https://github.com/daviddmc/NeSVoR)
>- NeSVoR is a package for GPU-accelerated slice-to-volume reconstruction.

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-reconstruction, 3d-visualization, deep-learning, image-reconstruction, image-registration, implicit-neural-representation, medical-imaging, mri, nerf, neural-network, neural-rendering, pytorch, segmentation, super-resolution, transformers
>- Forks:	21 
>- Issues:	7
>- Watchers:	97
>- Last updated: 2026-01-07

- [SIRF](https://github.com/SyneRBI/SIRF?tab=readme-ov-file)
>- Main repository for the CCP SynerBI software

>- License: Other
>- Languages: `C++`
>- Tags: image-reconstruction, medical-imaging, pet-mr
>- Forks:	29 
>- Issues:	167
>- Watchers:	68
>- Last updated: 2026-01-15

### Dicom <a name="dicom"></a>
- [nibabel](https://github.com/nipy/nibabel)
>- Python package to access a cacophony of neuro-imaging file formats

>- License: Other
>- Languages: `Python`
>- Tags: afni-brik-head, brain-imaging, cifti-2, data-formats, dicom, ecat, gifti, minc, neuroimaging, nifti, python, streamlines, tck, trk
>- Forks:	273 
>- Issues:	145
>- Watchers:	759
>- Last updated: 2026-01-15

- [mritopng](https://github.com/danishm/mritopng)
>- A simple python module to make it easy to batch convert DICOM files to PNG images.

>- License: MIT License
>- Languages: `Python`
>- Tags: dicom, dicom-converter, dicom-images, medical, medical-images, png, python
>- Forks:	51 
>- Issues:	5
>- Watchers:	146
>- Last updated: 2025-10-31

### Analysis <a name="analysis"></a>
- [DCEMRI.jl](https://github.com/davidssmith/DCEMRI.jl)
>- World's fastest DCE MRI analysis toolkit

>- License: Other
>- Languages: `Julia`
>- Tags: analysis, cancer-imaging-research, dce-mri, julia, mat-files, medical-image-processing, medical-imaging
>- Forks:	16 
>- Issues:	5
>- Watchers:	39
>- Last updated: 2026-01-12

### Cancer-Imaging-Research <a name="cancer-imaging-research"></a>
- [DCEMRI.jl](https://github.com/davidssmith/DCEMRI.jl)
>- World's fastest DCE MRI analysis toolkit

>- License: Other
>- Languages: `Julia`
>- Tags: analysis, cancer-imaging-research, dce-mri, julia, mat-files, medical-image-processing, medical-imaging
>- Forks:	16 
>- Issues:	5
>- Watchers:	39
>- Last updated: 2026-01-12

### Dce-Mri <a name="dce-mri"></a>
- [DCEMRI.jl](https://github.com/davidssmith/DCEMRI.jl)
>- World's fastest DCE MRI analysis toolkit

>- License: Other
>- Languages: `Julia`
>- Tags: analysis, cancer-imaging-research, dce-mri, julia, mat-files, medical-image-processing, medical-imaging
>- Forks:	16 
>- Issues:	5
>- Watchers:	39
>- Last updated: 2026-01-12

### Mat-Files <a name="mat-files"></a>
- [DCEMRI.jl](https://github.com/davidssmith/DCEMRI.jl)
>- World's fastest DCE MRI analysis toolkit

>- License: Other
>- Languages: `Julia`
>- Tags: analysis, cancer-imaging-research, dce-mri, julia, mat-files, medical-image-processing, medical-imaging
>- Forks:	16 
>- Issues:	5
>- Watchers:	39
>- Last updated: 2026-01-12

### Reconstruction <a name="reconstruction"></a>
- [SVRTK](https://github.com/SVRTK/SVRTK)
>- MIRTK based SVR reconstruction

>- License: Apache License 2.0
>- Languages: `C++`
>- Tags: fetal, mri, reconstruction, retrospecitve, slice-to-volume
>- Forks:	8 
>- Issues:	5
>- Watchers:	63
>- Last updated: 2025-12-25

### Retrospecitve <a name="retrospecitve"></a>
- [SVRTK](https://github.com/SVRTK/SVRTK)
>- MIRTK based SVR reconstruction

>- License: Apache License 2.0
>- Languages: `C++`
>- Tags: fetal, mri, reconstruction, retrospecitve, slice-to-volume
>- Forks:	8 
>- Issues:	5
>- Watchers:	63
>- Last updated: 2025-12-25

### Slice-To-Volume <a name="slice-to-volume"></a>
- [SVRTK](https://github.com/SVRTK/SVRTK)
>- MIRTK based SVR reconstruction

>- License: Apache License 2.0
>- Languages: `C++`
>- Tags: fetal, mri, reconstruction, retrospecitve, slice-to-volume
>- Forks:	8 
>- Issues:	5
>- Watchers:	63
>- Last updated: 2025-12-25

### Inverse-Problems <a name="inverse-problems"></a>
- [direct](https://github.com/NKI-AI/direct)
>- Deep learning framework for MRI reconstruction

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: deep-learning, fastmri-challenge, inverse-problems, medical-imaging, mri-reconstruction, pytorch
>- Forks:	47 
>- Issues:	6
>- Watchers:	294
>- Last updated: 2026-01-09

### Big-Data <a name="big-data"></a>
- [nipype](https://github.com/nipy/nipype)
>- Workflows and interfaces for neuroimaging packages

>- License: Other
>- Languages: `Python`
>- Tags: big-data, brain-imaging, brainweb, data-science, dataflow, dataflow-programming, neuroimaging, python, workflow-engine
>- Forks:	538 
>- Issues:	434
>- Watchers:	802
>- Last updated: 2026-01-14

### Brainweb <a name="brainweb"></a>
- [nipype](https://github.com/nipy/nipype)
>- Workflows and interfaces for neuroimaging packages

>- License: Other
>- Languages: `Python`
>- Tags: big-data, brain-imaging, brainweb, data-science, dataflow, dataflow-programming, neuroimaging, python, workflow-engine
>- Forks:	538 
>- Issues:	434
>- Watchers:	802
>- Last updated: 2026-01-14

### Data-Science <a name="data-science"></a>
- [nipype](https://github.com/nipy/nipype)
>- Workflows and interfaces for neuroimaging packages

>- License: Other
>- Languages: `Python`
>- Tags: big-data, brain-imaging, brainweb, data-science, dataflow, dataflow-programming, neuroimaging, python, workflow-engine
>- Forks:	538 
>- Issues:	434
>- Watchers:	802
>- Last updated: 2026-01-14

### Dataflow <a name="dataflow"></a>
- [nipype](https://github.com/nipy/nipype)
>- Workflows and interfaces for neuroimaging packages

>- License: Other
>- Languages: `Python`
>- Tags: big-data, brain-imaging, brainweb, data-science, dataflow, dataflow-programming, neuroimaging, python, workflow-engine
>- Forks:	538 
>- Issues:	434
>- Watchers:	802
>- Last updated: 2026-01-14

### Dataflow-Programming <a name="dataflow-programming"></a>
- [nipype](https://github.com/nipy/nipype)
>- Workflows and interfaces for neuroimaging packages

>- License: Other
>- Languages: `Python`
>- Tags: big-data, brain-imaging, brainweb, data-science, dataflow, dataflow-programming, neuroimaging, python, workflow-engine
>- Forks:	538 
>- Issues:	434
>- Watchers:	802
>- Last updated: 2026-01-14

### Workflow-Engine <a name="workflow-engine"></a>
- [nipype](https://github.com/nipy/nipype)
>- Workflows and interfaces for neuroimaging packages

>- License: Other
>- Languages: `Python`
>- Tags: big-data, brain-imaging, brainweb, data-science, dataflow, dataflow-programming, neuroimaging, python, workflow-engine
>- Forks:	538 
>- Issues:	434
>- Watchers:	802
>- Last updated: 2026-01-14

### Freesurfer <a name="freesurfer"></a>
- [freesurfer](https://github.com/freesurfer/freesurfer)
>- Neuroimaging analysis and visualization suite

>- License: Other
>- Languages: `C++`
>- Tags: freesurfer, lcn, mri, neuroimaging
>- Forks:	278 
>- Issues:	35
>- Watchers:	773
>- Last updated: 2026-01-16

### Lcn <a name="lcn"></a>
- [freesurfer](https://github.com/freesurfer/freesurfer)
>- Neuroimaging analysis and visualization suite

>- License: Other
>- Languages: `C++`
>- Tags: freesurfer, lcn, mri, neuroimaging
>- Forks:	278 
>- Issues:	35
>- Watchers:	773
>- Last updated: 2026-01-16

### 3D-Convolutional-Network <a name="3d-convolutional-network"></a>
- [MedicalZooPytorch](https://github.com/black0017/MedicalZooPytorch)
>- A pytorch-based deep learning framework for multi-modal 2D/3D medical image segmentation

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-convolutional-network, brats2018, brats2019, deep-learning, densenet, iseg, iseg-challenge, medical-image-processing, medical-image-segmentation, medical-imaging, mrbrains18, pytorch, resnet, segmentation, segmentation-models, unet, unet-image-segmentation
>- Forks:	305 
>- Issues:	21
>- Watchers:	1893
>- Last updated: 2026-01-13

### Brats2018 <a name="brats2018"></a>
- [MedicalZooPytorch](https://github.com/black0017/MedicalZooPytorch)
>- A pytorch-based deep learning framework for multi-modal 2D/3D medical image segmentation

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-convolutional-network, brats2018, brats2019, deep-learning, densenet, iseg, iseg-challenge, medical-image-processing, medical-image-segmentation, medical-imaging, mrbrains18, pytorch, resnet, segmentation, segmentation-models, unet, unet-image-segmentation
>- Forks:	305 
>- Issues:	21
>- Watchers:	1893
>- Last updated: 2026-01-13

### Brats2019 <a name="brats2019"></a>
- [MedicalZooPytorch](https://github.com/black0017/MedicalZooPytorch)
>- A pytorch-based deep learning framework for multi-modal 2D/3D medical image segmentation

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-convolutional-network, brats2018, brats2019, deep-learning, densenet, iseg, iseg-challenge, medical-image-processing, medical-image-segmentation, medical-imaging, mrbrains18, pytorch, resnet, segmentation, segmentation-models, unet, unet-image-segmentation
>- Forks:	305 
>- Issues:	21
>- Watchers:	1893
>- Last updated: 2026-01-13

### Densenet <a name="densenet"></a>
- [MedicalZooPytorch](https://github.com/black0017/MedicalZooPytorch)
>- A pytorch-based deep learning framework for multi-modal 2D/3D medical image segmentation

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-convolutional-network, brats2018, brats2019, deep-learning, densenet, iseg, iseg-challenge, medical-image-processing, medical-image-segmentation, medical-imaging, mrbrains18, pytorch, resnet, segmentation, segmentation-models, unet, unet-image-segmentation
>- Forks:	305 
>- Issues:	21
>- Watchers:	1893
>- Last updated: 2026-01-13

### Iseg <a name="iseg"></a>
- [MedicalZooPytorch](https://github.com/black0017/MedicalZooPytorch)
>- A pytorch-based deep learning framework for multi-modal 2D/3D medical image segmentation

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-convolutional-network, brats2018, brats2019, deep-learning, densenet, iseg, iseg-challenge, medical-image-processing, medical-image-segmentation, medical-imaging, mrbrains18, pytorch, resnet, segmentation, segmentation-models, unet, unet-image-segmentation
>- Forks:	305 
>- Issues:	21
>- Watchers:	1893
>- Last updated: 2026-01-13

### Iseg-Challenge <a name="iseg-challenge"></a>
- [MedicalZooPytorch](https://github.com/black0017/MedicalZooPytorch)
>- A pytorch-based deep learning framework for multi-modal 2D/3D medical image segmentation

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-convolutional-network, brats2018, brats2019, deep-learning, densenet, iseg, iseg-challenge, medical-image-processing, medical-image-segmentation, medical-imaging, mrbrains18, pytorch, resnet, segmentation, segmentation-models, unet, unet-image-segmentation
>- Forks:	305 
>- Issues:	21
>- Watchers:	1893
>- Last updated: 2026-01-13

### Medical-Image-Segmentation <a name="medical-image-segmentation"></a>
- [MedicalZooPytorch](https://github.com/black0017/MedicalZooPytorch)
>- A pytorch-based deep learning framework for multi-modal 2D/3D medical image segmentation

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-convolutional-network, brats2018, brats2019, deep-learning, densenet, iseg, iseg-challenge, medical-image-processing, medical-image-segmentation, medical-imaging, mrbrains18, pytorch, resnet, segmentation, segmentation-models, unet, unet-image-segmentation
>- Forks:	305 
>- Issues:	21
>- Watchers:	1893
>- Last updated: 2026-01-13

### Mrbrains18 <a name="mrbrains18"></a>
- [MedicalZooPytorch](https://github.com/black0017/MedicalZooPytorch)
>- A pytorch-based deep learning framework for multi-modal 2D/3D medical image segmentation

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-convolutional-network, brats2018, brats2019, deep-learning, densenet, iseg, iseg-challenge, medical-image-processing, medical-image-segmentation, medical-imaging, mrbrains18, pytorch, resnet, segmentation, segmentation-models, unet, unet-image-segmentation
>- Forks:	305 
>- Issues:	21
>- Watchers:	1893
>- Last updated: 2026-01-13

### Resnet <a name="resnet"></a>
- [MedicalZooPytorch](https://github.com/black0017/MedicalZooPytorch)
>- A pytorch-based deep learning framework for multi-modal 2D/3D medical image segmentation

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-convolutional-network, brats2018, brats2019, deep-learning, densenet, iseg, iseg-challenge, medical-image-processing, medical-image-segmentation, medical-imaging, mrbrains18, pytorch, resnet, segmentation, segmentation-models, unet, unet-image-segmentation
>- Forks:	305 
>- Issues:	21
>- Watchers:	1893
>- Last updated: 2026-01-13

### Segmentation-Models <a name="segmentation-models"></a>
- [MedicalZooPytorch](https://github.com/black0017/MedicalZooPytorch)
>- A pytorch-based deep learning framework for multi-modal 2D/3D medical image segmentation

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-convolutional-network, brats2018, brats2019, deep-learning, densenet, iseg, iseg-challenge, medical-image-processing, medical-image-segmentation, medical-imaging, mrbrains18, pytorch, resnet, segmentation, segmentation-models, unet, unet-image-segmentation
>- Forks:	305 
>- Issues:	21
>- Watchers:	1893
>- Last updated: 2026-01-13

### Unet <a name="unet"></a>
- [MedicalZooPytorch](https://github.com/black0017/MedicalZooPytorch)
>- A pytorch-based deep learning framework for multi-modal 2D/3D medical image segmentation

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-convolutional-network, brats2018, brats2019, deep-learning, densenet, iseg, iseg-challenge, medical-image-processing, medical-image-segmentation, medical-imaging, mrbrains18, pytorch, resnet, segmentation, segmentation-models, unet, unet-image-segmentation
>- Forks:	305 
>- Issues:	21
>- Watchers:	1893
>- Last updated: 2026-01-13

### Unet-Image-Segmentation <a name="unet-image-segmentation"></a>
- [MedicalZooPytorch](https://github.com/black0017/MedicalZooPytorch)
>- A pytorch-based deep learning framework for multi-modal 2D/3D medical image segmentation

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-convolutional-network, brats2018, brats2019, deep-learning, densenet, iseg, iseg-challenge, medical-image-processing, medical-image-segmentation, medical-imaging, mrbrains18, pytorch, resnet, segmentation, segmentation-models, unet, unet-image-segmentation
>- Forks:	305 
>- Issues:	21
>- Watchers:	1893
>- Last updated: 2026-01-13

### Fetal-Mri <a name="fetal-mri"></a>
- [fetal-IQA](https://github.com/daviddmc/fetal-IQA)
>- Image quality assessment for fetal MRI

>- License: MIT License
>- Languages: `Python`
>- Tags: convolutional-neural-networks, deep-learning, fetal-mri, medical-imaging, pytorch, quality-control, semi-supervised-learning, tensorflow
>- Forks:	0 
>- Issues:	0
>- Watchers:	6
>- Last updated: 2024-10-12

### Semi-Supervised-Learning <a name="semi-supervised-learning"></a>
- [fetal-IQA](https://github.com/daviddmc/fetal-IQA)
>- Image quality assessment for fetal MRI

>- License: MIT License
>- Languages: `Python`
>- Tags: convolutional-neural-networks, deep-learning, fetal-mri, medical-imaging, pytorch, quality-control, semi-supervised-learning, tensorflow
>- Forks:	0 
>- Issues:	0
>- Watchers:	6
>- Last updated: 2024-10-12

### Cortical-Thickness <a name="cortical-thickness"></a>
- [DL-DiReCT](https://github.com/SCAN-NRAD/DL-DiReCT)
>- DL+DiReCT - Direct Cortical Thickness Estimation using Deep Learning-based Anatomy Segmentation and Cortex Parcellation

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: cortical-thickness, deep-learning, morphometry, mri
>- Forks:	6 
>- Issues:	4
>- Watchers:	28
>- Last updated: 2025-07-23

### Morphometry <a name="morphometry"></a>
- [DL-DiReCT](https://github.com/SCAN-NRAD/DL-DiReCT)
>- DL+DiReCT - Direct Cortical Thickness Estimation using Deep Learning-based Anatomy Segmentation and Cortex Parcellation

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: cortical-thickness, deep-learning, morphometry, mri
>- Forks:	6 
>- Issues:	4
>- Watchers:	28
>- Last updated: 2025-07-23

### 3D-Mask-Rcnn <a name="3d-mask-rcnn"></a>
- [medicaldetectiontoolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit)
>- The Medical Detection Toolkit contains 2D + 3D implementations of prevalent object detectors such as Mask R-CNN, Retina Net, Retina U-Net, as well as a training and inference framework focused on dealing with medical images.  

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: 3d-mask-rcnn, 3d-models, 3d-object-detection, deep-learning, deep-neural-networks, detection, mask-rcnn, medical-image-analysis, medical-image-computing, medical-image-processing, medical-imaging, object-detection, pytorch-cnn, pytorch-deeplearning, pytorch-implementation, retina-net, retina-unet, segmentation, semantic-segmentation, u-net
>- Forks:	292 
>- Issues:	47
>- Watchers:	1345
>- Last updated: 2026-01-15

### 3D-Models <a name="3d-models"></a>
- [medicaldetectiontoolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit)
>- The Medical Detection Toolkit contains 2D + 3D implementations of prevalent object detectors such as Mask R-CNN, Retina Net, Retina U-Net, as well as a training and inference framework focused on dealing with medical images.  

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: 3d-mask-rcnn, 3d-models, 3d-object-detection, deep-learning, deep-neural-networks, detection, mask-rcnn, medical-image-analysis, medical-image-computing, medical-image-processing, medical-imaging, object-detection, pytorch-cnn, pytorch-deeplearning, pytorch-implementation, retina-net, retina-unet, segmentation, semantic-segmentation, u-net
>- Forks:	292 
>- Issues:	47
>- Watchers:	1345
>- Last updated: 2026-01-15

### 3D-Object-Detection <a name="3d-object-detection"></a>
- [medicaldetectiontoolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit)
>- The Medical Detection Toolkit contains 2D + 3D implementations of prevalent object detectors such as Mask R-CNN, Retina Net, Retina U-Net, as well as a training and inference framework focused on dealing with medical images.  

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: 3d-mask-rcnn, 3d-models, 3d-object-detection, deep-learning, deep-neural-networks, detection, mask-rcnn, medical-image-analysis, medical-image-computing, medical-image-processing, medical-imaging, object-detection, pytorch-cnn, pytorch-deeplearning, pytorch-implementation, retina-net, retina-unet, segmentation, semantic-segmentation, u-net
>- Forks:	292 
>- Issues:	47
>- Watchers:	1345
>- Last updated: 2026-01-15

### Deep-Neural-Networks <a name="deep-neural-networks"></a>
- [medicaldetectiontoolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit)
>- The Medical Detection Toolkit contains 2D + 3D implementations of prevalent object detectors such as Mask R-CNN, Retina Net, Retina U-Net, as well as a training and inference framework focused on dealing with medical images.  

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: 3d-mask-rcnn, 3d-models, 3d-object-detection, deep-learning, deep-neural-networks, detection, mask-rcnn, medical-image-analysis, medical-image-computing, medical-image-processing, medical-imaging, object-detection, pytorch-cnn, pytorch-deeplearning, pytorch-implementation, retina-net, retina-unet, segmentation, semantic-segmentation, u-net
>- Forks:	292 
>- Issues:	47
>- Watchers:	1345
>- Last updated: 2026-01-15

### Detection <a name="detection"></a>
- [medicaldetectiontoolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit)
>- The Medical Detection Toolkit contains 2D + 3D implementations of prevalent object detectors such as Mask R-CNN, Retina Net, Retina U-Net, as well as a training and inference framework focused on dealing with medical images.  

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: 3d-mask-rcnn, 3d-models, 3d-object-detection, deep-learning, deep-neural-networks, detection, mask-rcnn, medical-image-analysis, medical-image-computing, medical-image-processing, medical-imaging, object-detection, pytorch-cnn, pytorch-deeplearning, pytorch-implementation, retina-net, retina-unet, segmentation, semantic-segmentation, u-net
>- Forks:	292 
>- Issues:	47
>- Watchers:	1345
>- Last updated: 2026-01-15

### Mask-Rcnn <a name="mask-rcnn"></a>
- [medicaldetectiontoolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit)
>- The Medical Detection Toolkit contains 2D + 3D implementations of prevalent object detectors such as Mask R-CNN, Retina Net, Retina U-Net, as well as a training and inference framework focused on dealing with medical images.  

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: 3d-mask-rcnn, 3d-models, 3d-object-detection, deep-learning, deep-neural-networks, detection, mask-rcnn, medical-image-analysis, medical-image-computing, medical-image-processing, medical-imaging, object-detection, pytorch-cnn, pytorch-deeplearning, pytorch-implementation, retina-net, retina-unet, segmentation, semantic-segmentation, u-net
>- Forks:	292 
>- Issues:	47
>- Watchers:	1345
>- Last updated: 2026-01-15

### Object-Detection <a name="object-detection"></a>
- [medicaldetectiontoolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit)
>- The Medical Detection Toolkit contains 2D + 3D implementations of prevalent object detectors such as Mask R-CNN, Retina Net, Retina U-Net, as well as a training and inference framework focused on dealing with medical images.  

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: 3d-mask-rcnn, 3d-models, 3d-object-detection, deep-learning, deep-neural-networks, detection, mask-rcnn, medical-image-analysis, medical-image-computing, medical-image-processing, medical-imaging, object-detection, pytorch-cnn, pytorch-deeplearning, pytorch-implementation, retina-net, retina-unet, segmentation, semantic-segmentation, u-net
>- Forks:	292 
>- Issues:	47
>- Watchers:	1345
>- Last updated: 2026-01-15

### Pytorch-Cnn <a name="pytorch-cnn"></a>
- [medicaldetectiontoolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit)
>- The Medical Detection Toolkit contains 2D + 3D implementations of prevalent object detectors such as Mask R-CNN, Retina Net, Retina U-Net, as well as a training and inference framework focused on dealing with medical images.  

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: 3d-mask-rcnn, 3d-models, 3d-object-detection, deep-learning, deep-neural-networks, detection, mask-rcnn, medical-image-analysis, medical-image-computing, medical-image-processing, medical-imaging, object-detection, pytorch-cnn, pytorch-deeplearning, pytorch-implementation, retina-net, retina-unet, segmentation, semantic-segmentation, u-net
>- Forks:	292 
>- Issues:	47
>- Watchers:	1345
>- Last updated: 2026-01-15

### Pytorch-Deeplearning <a name="pytorch-deeplearning"></a>
- [medicaldetectiontoolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit)
>- The Medical Detection Toolkit contains 2D + 3D implementations of prevalent object detectors such as Mask R-CNN, Retina Net, Retina U-Net, as well as a training and inference framework focused on dealing with medical images.  

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: 3d-mask-rcnn, 3d-models, 3d-object-detection, deep-learning, deep-neural-networks, detection, mask-rcnn, medical-image-analysis, medical-image-computing, medical-image-processing, medical-imaging, object-detection, pytorch-cnn, pytorch-deeplearning, pytorch-implementation, retina-net, retina-unet, segmentation, semantic-segmentation, u-net
>- Forks:	292 
>- Issues:	47
>- Watchers:	1345
>- Last updated: 2026-01-15

### Pytorch-Implementation <a name="pytorch-implementation"></a>
- [medicaldetectiontoolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit)
>- The Medical Detection Toolkit contains 2D + 3D implementations of prevalent object detectors such as Mask R-CNN, Retina Net, Retina U-Net, as well as a training and inference framework focused on dealing with medical images.  

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: 3d-mask-rcnn, 3d-models, 3d-object-detection, deep-learning, deep-neural-networks, detection, mask-rcnn, medical-image-analysis, medical-image-computing, medical-image-processing, medical-imaging, object-detection, pytorch-cnn, pytorch-deeplearning, pytorch-implementation, retina-net, retina-unet, segmentation, semantic-segmentation, u-net
>- Forks:	292 
>- Issues:	47
>- Watchers:	1345
>- Last updated: 2026-01-15

### Retina-Net <a name="retina-net"></a>
- [medicaldetectiontoolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit)
>- The Medical Detection Toolkit contains 2D + 3D implementations of prevalent object detectors such as Mask R-CNN, Retina Net, Retina U-Net, as well as a training and inference framework focused on dealing with medical images.  

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: 3d-mask-rcnn, 3d-models, 3d-object-detection, deep-learning, deep-neural-networks, detection, mask-rcnn, medical-image-analysis, medical-image-computing, medical-image-processing, medical-imaging, object-detection, pytorch-cnn, pytorch-deeplearning, pytorch-implementation, retina-net, retina-unet, segmentation, semantic-segmentation, u-net
>- Forks:	292 
>- Issues:	47
>- Watchers:	1345
>- Last updated: 2026-01-15

### Retina-Unet <a name="retina-unet"></a>
- [medicaldetectiontoolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit)
>- The Medical Detection Toolkit contains 2D + 3D implementations of prevalent object detectors such as Mask R-CNN, Retina Net, Retina U-Net, as well as a training and inference framework focused on dealing with medical images.  

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: 3d-mask-rcnn, 3d-models, 3d-object-detection, deep-learning, deep-neural-networks, detection, mask-rcnn, medical-image-analysis, medical-image-computing, medical-image-processing, medical-imaging, object-detection, pytorch-cnn, pytorch-deeplearning, pytorch-implementation, retina-net, retina-unet, segmentation, semantic-segmentation, u-net
>- Forks:	292 
>- Issues:	47
>- Watchers:	1345
>- Last updated: 2026-01-15

### Semantic-Segmentation <a name="semantic-segmentation"></a>
- [medicaldetectiontoolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit)
>- The Medical Detection Toolkit contains 2D + 3D implementations of prevalent object detectors such as Mask R-CNN, Retina Net, Retina U-Net, as well as a training and inference framework focused on dealing with medical images.  

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: 3d-mask-rcnn, 3d-models, 3d-object-detection, deep-learning, deep-neural-networks, detection, mask-rcnn, medical-image-analysis, medical-image-computing, medical-image-processing, medical-imaging, object-detection, pytorch-cnn, pytorch-deeplearning, pytorch-implementation, retina-net, retina-unet, segmentation, semantic-segmentation, u-net
>- Forks:	292 
>- Issues:	47
>- Watchers:	1345
>- Last updated: 2026-01-15

### U-Net <a name="u-net"></a>
- [medicaldetectiontoolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit)
>- The Medical Detection Toolkit contains 2D + 3D implementations of prevalent object detectors such as Mask R-CNN, Retina Net, Retina U-Net, as well as a training and inference framework focused on dealing with medical images.  

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: 3d-mask-rcnn, 3d-models, 3d-object-detection, deep-learning, deep-neural-networks, detection, mask-rcnn, medical-image-analysis, medical-image-computing, medical-image-processing, medical-imaging, object-detection, pytorch-cnn, pytorch-deeplearning, pytorch-implementation, retina-net, retina-unet, segmentation, semantic-segmentation, u-net
>- Forks:	292 
>- Issues:	47
>- Watchers:	1345
>- Last updated: 2026-01-15

### Healthcare-Imaging <a name="healthcare-imaging"></a>
- [MONAI](https://github.com/Project-MONAI/MONAI)
>- AI Toolkit for Healthcare Imaging

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: deep-learning, healthcare-imaging, medical-image-computing, medical-image-processing, monai, python3, pytorch
>- Forks:	1403 
>- Issues:	505
>- Watchers:	7753
>- Last updated: 2026-01-16

### Monai <a name="monai"></a>
- [MONAI](https://github.com/Project-MONAI/MONAI)
>- AI Toolkit for Healthcare Imaging

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: deep-learning, healthcare-imaging, medical-image-computing, medical-image-processing, monai, python3, pytorch
>- Forks:	1403 
>- Issues:	505
>- Watchers:	7753
>- Last updated: 2026-01-16

### Python3 <a name="python3"></a>
- [MONAI](https://github.com/Project-MONAI/MONAI)
>- AI Toolkit for Healthcare Imaging

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: deep-learning, healthcare-imaging, medical-image-computing, medical-image-processing, monai, python3, pytorch
>- Forks:	1403 
>- Issues:	505
>- Watchers:	7753
>- Last updated: 2026-01-16

### Brain <a name="brain"></a>
- [mrQA](https://github.com/Open-Minds-Lab/mrQA)
>- mrQA: tools for quality assurance in medical imaging datasets, including protocol compliance

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: brain, ismrm, mr-image, mri, mri-brain, mri-images, neuroimaging, neuroscience, niqc, qa, quality-assurance, quality-control
>- Forks:	6 
>- Issues:	38
>- Watchers:	13
>- Last updated: 2026-01-14

### Ismrm <a name="ismrm"></a>
- [mrQA](https://github.com/Open-Minds-Lab/mrQA)
>- mrQA: tools for quality assurance in medical imaging datasets, including protocol compliance

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: brain, ismrm, mr-image, mri, mri-brain, mri-images, neuroimaging, neuroscience, niqc, qa, quality-assurance, quality-control
>- Forks:	6 
>- Issues:	38
>- Watchers:	13
>- Last updated: 2026-01-14

### Mr-Image <a name="mr-image"></a>
- [mrQA](https://github.com/Open-Minds-Lab/mrQA)
>- mrQA: tools for quality assurance in medical imaging datasets, including protocol compliance

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: brain, ismrm, mr-image, mri, mri-brain, mri-images, neuroimaging, neuroscience, niqc, qa, quality-assurance, quality-control
>- Forks:	6 
>- Issues:	38
>- Watchers:	13
>- Last updated: 2026-01-14

### Mri-Brain <a name="mri-brain"></a>
- [mrQA](https://github.com/Open-Minds-Lab/mrQA)
>- mrQA: tools for quality assurance in medical imaging datasets, including protocol compliance

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: brain, ismrm, mr-image, mri, mri-brain, mri-images, neuroimaging, neuroscience, niqc, qa, quality-assurance, quality-control
>- Forks:	6 
>- Issues:	38
>- Watchers:	13
>- Last updated: 2026-01-14

### Niqc <a name="niqc"></a>
- [mrQA](https://github.com/Open-Minds-Lab/mrQA)
>- mrQA: tools for quality assurance in medical imaging datasets, including protocol compliance

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: brain, ismrm, mr-image, mri, mri-brain, mri-images, neuroimaging, neuroscience, niqc, qa, quality-assurance, quality-control
>- Forks:	6 
>- Issues:	38
>- Watchers:	13
>- Last updated: 2026-01-14

### Denoising-Algorithm <a name="denoising-algorithm"></a>
- [nlsam](https://github.com/samuelstjean/nlsam)
>- The reference implementation for the Non Local Spatial and Angular Matching (NLSAM) denoising algorithm for diffusion MRI

>- License: GNU General Public License v3.0
>- Languages: `Python`
>- Tags: denoising-algorithm, diffusion-mri, machine-learning, python
>- Forks:	11 
>- Issues:	1
>- Watchers:	24
>- Last updated: 2025-11-17

### Bids-Apps <a name="bids-apps"></a>
- [mialsuperresolutiontoolkit](https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit)
>- The Medical Image Analysis Laboratory Super-Resolution ToolKit (MIALSRTK) consists of a set of C++ and Python processing and workflow tools necessary to perform motion-robust super-resolution fetal MRI reconstruction in the BIDS Apps framework.

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `C++`
>- Tags: bids, bids-apps, fetal, itk, mri, nipype, super-resolution, workflow
>- Forks:	15 
>- Issues:	17
>- Watchers:	30
>- Last updated: 2025-12-30

### Nipype <a name="nipype"></a>
- [mialsuperresolutiontoolkit](https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit)
>- The Medical Image Analysis Laboratory Super-Resolution ToolKit (MIALSRTK) consists of a set of C++ and Python processing and workflow tools necessary to perform motion-robust super-resolution fetal MRI reconstruction in the BIDS Apps framework.

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `C++`
>- Tags: bids, bids-apps, fetal, itk, mri, nipype, super-resolution, workflow
>- Forks:	15 
>- Issues:	17
>- Watchers:	30
>- Last updated: 2025-12-30

### Workflow <a name="workflow"></a>
- [mialsuperresolutiontoolkit](https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit)
>- The Medical Image Analysis Laboratory Super-Resolution ToolKit (MIALSRTK) consists of a set of C++ and Python processing and workflow tools necessary to perform motion-robust super-resolution fetal MRI reconstruction in the BIDS Apps framework.

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `C++`
>- Tags: bids, bids-apps, fetal, itk, mri, nipype, super-resolution, workflow
>- Forks:	15 
>- Issues:	17
>- Watchers:	30
>- Last updated: 2025-12-30

### Complex-Networks <a name="complex-networks"></a>
- [brainGraph](https://github.com/cwatson/brainGraph)
>- Graph theory analysis of brain MRI data

>- License: None
>- Languages: `R`
>- Tags: brain-connectivity, brain-imaging, complex-networks, connectome, connectomics, fmri, graph-theory, mri, network-analysis, neuroimaging, neuroscience, r, statistics, tractography
>- Forks:	54 
>- Issues:	11
>- Watchers:	191
>- Last updated: 2025-11-13

### Connectome <a name="connectome"></a>
- [brainGraph](https://github.com/cwatson/brainGraph)
>- Graph theory analysis of brain MRI data

>- License: None
>- Languages: `R`
>- Tags: brain-connectivity, brain-imaging, complex-networks, connectome, connectomics, fmri, graph-theory, mri, network-analysis, neuroimaging, neuroscience, r, statistics, tractography
>- Forks:	54 
>- Issues:	11
>- Watchers:	191
>- Last updated: 2025-11-13

### Connectomics <a name="connectomics"></a>
- [brainGraph](https://github.com/cwatson/brainGraph)
>- Graph theory analysis of brain MRI data

>- License: None
>- Languages: `R`
>- Tags: brain-connectivity, brain-imaging, complex-networks, connectome, connectomics, fmri, graph-theory, mri, network-analysis, neuroimaging, neuroscience, r, statistics, tractography
>- Forks:	54 
>- Issues:	11
>- Watchers:	191
>- Last updated: 2025-11-13

### Graph-Theory <a name="graph-theory"></a>
- [brainGraph](https://github.com/cwatson/brainGraph)
>- Graph theory analysis of brain MRI data

>- License: None
>- Languages: `R`
>- Tags: brain-connectivity, brain-imaging, complex-networks, connectome, connectomics, fmri, graph-theory, mri, network-analysis, neuroimaging, neuroscience, r, statistics, tractography
>- Forks:	54 
>- Issues:	11
>- Watchers:	191
>- Last updated: 2025-11-13

### Network-Analysis <a name="network-analysis"></a>
- [brainGraph](https://github.com/cwatson/brainGraph)
>- Graph theory analysis of brain MRI data

>- License: None
>- Languages: `R`
>- Tags: brain-connectivity, brain-imaging, complex-networks, connectome, connectomics, fmri, graph-theory, mri, network-analysis, neuroimaging, neuroscience, r, statistics, tractography
>- Forks:	54 
>- Issues:	11
>- Watchers:	191
>- Last updated: 2025-11-13

### Statistics <a name="statistics"></a>
- [brainGraph](https://github.com/cwatson/brainGraph)
>- Graph theory analysis of brain MRI data

>- License: None
>- Languages: `R`
>- Tags: brain-connectivity, brain-imaging, complex-networks, connectome, connectomics, fmri, graph-theory, mri, network-analysis, neuroimaging, neuroscience, r, statistics, tractography
>- Forks:	54 
>- Issues:	11
>- Watchers:	191
>- Last updated: 2025-11-13

### Image-Segmentation <a name="image-segmentation"></a>
- [smriprep](https://github.com/nipreps/smriprep)
>- Structural MRI PREProcessing (sMRIPrep) workflows for NIPreps (NeuroImaging PREProcessing tools)

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: image-processing, image-registration, image-segmentation, mri, structural-mri, surface-reconstruction
>- Forks:	47 
>- Issues:	73
>- Watchers:	162
>- Last updated: 2026-01-15

### Structural-Mri <a name="structural-mri"></a>
- [smriprep](https://github.com/nipreps/smriprep)
>- Structural MRI PREProcessing (sMRIPrep) workflows for NIPreps (NeuroImaging PREProcessing tools)

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: image-processing, image-registration, image-segmentation, mri, structural-mri, surface-reconstruction
>- Forks:	47 
>- Issues:	73
>- Watchers:	162
>- Last updated: 2026-01-15

### Surface-Reconstruction <a name="surface-reconstruction"></a>
- [smriprep](https://github.com/nipreps/smriprep)
>- Structural MRI PREProcessing (sMRIPrep) workflows for NIPreps (NeuroImaging PREProcessing tools)

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: image-processing, image-registration, image-segmentation, mri, structural-mri, surface-reconstruction
>- Forks:	47 
>- Issues:	73
>- Watchers:	162
>- Last updated: 2026-01-15

### Denoising-Images <a name="denoising-images"></a>
- [qsiprep](https://github.com/PennLINC/qsiprep)
>- Preprocessing of diffusion MRI

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: bids, denoising-images, diffusion-mri, distortion-correction, motion-correction, pipelines
>- Forks:	62 
>- Issues:	105
>- Watchers:	172
>- Last updated: 2026-01-16

### Distortion-Correction <a name="distortion-correction"></a>
- [qsiprep](https://github.com/PennLINC/qsiprep)
>- Preprocessing of diffusion MRI

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: bids, denoising-images, diffusion-mri, distortion-correction, motion-correction, pipelines
>- Forks:	62 
>- Issues:	105
>- Watchers:	172
>- Last updated: 2026-01-16

### Motion-Correction <a name="motion-correction"></a>
- [qsiprep](https://github.com/PennLINC/qsiprep)
>- Preprocessing of diffusion MRI

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: bids, denoising-images, diffusion-mri, distortion-correction, motion-correction, pipelines
>- Forks:	62 
>- Issues:	105
>- Watchers:	172
>- Last updated: 2026-01-16

### Pipelines <a name="pipelines"></a>
- [qsiprep](https://github.com/PennLINC/qsiprep)
>- Preprocessing of diffusion MRI

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: bids, denoising-images, diffusion-mri, distortion-correction, motion-correction, pipelines
>- Forks:	62 
>- Issues:	105
>- Watchers:	172
>- Last updated: 2026-01-16

### Mri-Sequences <a name="mri-sequences"></a>
- [pypulseq](https://github.com/imr-framework/pypulseq)
>- Pulseq in Python

>- License: GNU Affero General Public License v3.0
>- Languages: `Python`
>- Tags: mri, mri-sequences, pulse-sequences, pulseq, python
>- Forks:	77 
>- Issues:	19
>- Watchers:	186
>- Last updated: 2026-01-16

### Pulse-Sequences <a name="pulse-sequences"></a>
- [pypulseq](https://github.com/imr-framework/pypulseq)
>- Pulseq in Python

>- License: GNU Affero General Public License v3.0
>- Languages: `Python`
>- Tags: mri, mri-sequences, pulse-sequences, pulseq, python
>- Forks:	77 
>- Issues:	19
>- Watchers:	186
>- Last updated: 2026-01-16

### Pulseq <a name="pulseq"></a>
- [pypulseq](https://github.com/imr-framework/pypulseq)
>- Pulseq in Python

>- License: GNU Affero General Public License v3.0
>- Languages: `Python`
>- Tags: mri, mri-sequences, pulse-sequences, pulseq, python
>- Forks:	77 
>- Issues:	19
>- Watchers:	186
>- Last updated: 2026-01-16

### Neuroimage <a name="neuroimage"></a>
- [spinalcordtoolbox](https://github.com/spinalcordtoolbox/spinalcordtoolbox)
>- Comprehensive and open-source library of analysis tools for MRI of the spinal cord.

>- License: GNU Lesser General Public License v3.0
>- Languages: `Python`
>- Tags: mri, neuroimage, python, spinalcord
>- Forks:	113 
>- Issues:	375
>- Watchers:	251
>- Last updated: 2026-01-16

### Spinalcord <a name="spinalcord"></a>
- [spinalcordtoolbox](https://github.com/spinalcordtoolbox/spinalcordtoolbox)
>- Comprehensive and open-source library of analysis tools for MRI of the spinal cord.

>- License: GNU Lesser General Public License v3.0
>- Languages: `Python`
>- Tags: mri, neuroimage, python, spinalcord
>- Forks:	113 
>- Issues:	375
>- Watchers:	251
>- Last updated: 2026-01-16

### Fcm <a name="fcm"></a>
- [intensity-normalization](https://github.com/jcreinhold/intensity-normalization)
>- Normalize MR image intensities in Python

>- License: MIT License
>- Languages: `Python`
>- Tags: fcm, harmonization, intensity-normalization, mri, neuroimaging, normalization, standardization, whitestripe, zscore
>- Forks:	58 
>- Issues:	0
>- Watchers:	340
>- Last updated: 2025-12-19

### Harmonization <a name="harmonization"></a>
- [intensity-normalization](https://github.com/jcreinhold/intensity-normalization)
>- Normalize MR image intensities in Python

>- License: MIT License
>- Languages: `Python`
>- Tags: fcm, harmonization, intensity-normalization, mri, neuroimaging, normalization, standardization, whitestripe, zscore
>- Forks:	58 
>- Issues:	0
>- Watchers:	340
>- Last updated: 2025-12-19

### Intensity-Normalization <a name="intensity-normalization"></a>
- [intensity-normalization](https://github.com/jcreinhold/intensity-normalization)
>- Normalize MR image intensities in Python

>- License: MIT License
>- Languages: `Python`
>- Tags: fcm, harmonization, intensity-normalization, mri, neuroimaging, normalization, standardization, whitestripe, zscore
>- Forks:	58 
>- Issues:	0
>- Watchers:	340
>- Last updated: 2025-12-19

### Normalization <a name="normalization"></a>
- [intensity-normalization](https://github.com/jcreinhold/intensity-normalization)
>- Normalize MR image intensities in Python

>- License: MIT License
>- Languages: `Python`
>- Tags: fcm, harmonization, intensity-normalization, mri, neuroimaging, normalization, standardization, whitestripe, zscore
>- Forks:	58 
>- Issues:	0
>- Watchers:	340
>- Last updated: 2025-12-19

### Standardization <a name="standardization"></a>
- [intensity-normalization](https://github.com/jcreinhold/intensity-normalization)
>- Normalize MR image intensities in Python

>- License: MIT License
>- Languages: `Python`
>- Tags: fcm, harmonization, intensity-normalization, mri, neuroimaging, normalization, standardization, whitestripe, zscore
>- Forks:	58 
>- Issues:	0
>- Watchers:	340
>- Last updated: 2025-12-19

### Whitestripe <a name="whitestripe"></a>
- [intensity-normalization](https://github.com/jcreinhold/intensity-normalization)
>- Normalize MR image intensities in Python

>- License: MIT License
>- Languages: `Python`
>- Tags: fcm, harmonization, intensity-normalization, mri, neuroimaging, normalization, standardization, whitestripe, zscore
>- Forks:	58 
>- Issues:	0
>- Watchers:	340
>- Last updated: 2025-12-19

### Zscore <a name="zscore"></a>
- [intensity-normalization](https://github.com/jcreinhold/intensity-normalization)
>- Normalize MR image intensities in Python

>- License: MIT License
>- Languages: `Python`
>- Tags: fcm, harmonization, intensity-normalization, mri, neuroimaging, normalization, standardization, whitestripe, zscore
>- Forks:	58 
>- Issues:	0
>- Watchers:	340
>- Last updated: 2025-12-19

### Ml <a name="ml"></a>
- [tensorflow-mri](https://github.com/mrphys/tensorflow-mri)
>- A Library of TensorFlow Operators for Computational MRI

>- License: Apache License 2.0
>- Languages: `Jupyter Notebook`
>- Tags: machine-learning, magnetic-resonance-imaging, ml, mri, python, tensorflow
>- Forks:	6 
>- Issues:	10
>- Watchers:	47
>- Last updated: 2025-11-25

### Alzheimer-Disease <a name="alzheimer-disease"></a>
- [clinicadl](https://github.com/aramis-lab/clinicadl)
>- Framework for the reproducible processing of neuroimaging data with deep learning methods

>- License: MIT License
>- Languages: `Python`
>- Tags: alzheimer-disease, brain-imaging, convolutional-neural-network, deep-learning, medical-imaging, neuroimaging, python, pytorch
>- Forks:	60 
>- Issues:	65
>- Watchers:	176
>- Last updated: 2025-12-31

### Convolutional-Neural-Network <a name="convolutional-neural-network"></a>
- [clinicadl](https://github.com/aramis-lab/clinicadl)
>- Framework for the reproducible processing of neuroimaging data with deep learning methods

>- License: MIT License
>- Languages: `Python`
>- Tags: alzheimer-disease, brain-imaging, convolutional-neural-network, deep-learning, medical-imaging, neuroimaging, python, pytorch
>- Forks:	60 
>- Issues:	65
>- Watchers:	176
>- Last updated: 2025-12-31

### Csharp <a name="csharp"></a>
- [SimpleITK](https://github.com/SimpleITK/SimpleITK)
>- SimpleITK: a layer built on top of the Insight Toolkit (ITK), intended to simplify and facilitate ITK's use in rapid prototyping, education and interpreted languages.

>- License: Apache License 2.0
>- Languages: `Swig`
>- Tags: c-plus-plus, csharp, image-analysis, image-processing, itk, java, lua, python, r, registration, ruby, segmentation, simpleitk, swig, tcl
>- Forks:	223 
>- Issues:	71
>- Watchers:	1032
>- Last updated: 2026-01-16

### Image-Analysis <a name="image-analysis"></a>
- [SimpleITK](https://github.com/SimpleITK/SimpleITK)
>- SimpleITK: a layer built on top of the Insight Toolkit (ITK), intended to simplify and facilitate ITK's use in rapid prototyping, education and interpreted languages.

>- License: Apache License 2.0
>- Languages: `Swig`
>- Tags: c-plus-plus, csharp, image-analysis, image-processing, itk, java, lua, python, r, registration, ruby, segmentation, simpleitk, swig, tcl
>- Forks:	223 
>- Issues:	71
>- Watchers:	1032
>- Last updated: 2026-01-16

### Java <a name="java"></a>
- [SimpleITK](https://github.com/SimpleITK/SimpleITK)
>- SimpleITK: a layer built on top of the Insight Toolkit (ITK), intended to simplify and facilitate ITK's use in rapid prototyping, education and interpreted languages.

>- License: Apache License 2.0
>- Languages: `Swig`
>- Tags: c-plus-plus, csharp, image-analysis, image-processing, itk, java, lua, python, r, registration, ruby, segmentation, simpleitk, swig, tcl
>- Forks:	223 
>- Issues:	71
>- Watchers:	1032
>- Last updated: 2026-01-16

### Lua <a name="lua"></a>
- [SimpleITK](https://github.com/SimpleITK/SimpleITK)
>- SimpleITK: a layer built on top of the Insight Toolkit (ITK), intended to simplify and facilitate ITK's use in rapid prototyping, education and interpreted languages.

>- License: Apache License 2.0
>- Languages: `Swig`
>- Tags: c-plus-plus, csharp, image-analysis, image-processing, itk, java, lua, python, r, registration, ruby, segmentation, simpleitk, swig, tcl
>- Forks:	223 
>- Issues:	71
>- Watchers:	1032
>- Last updated: 2026-01-16

### Ruby <a name="ruby"></a>
- [SimpleITK](https://github.com/SimpleITK/SimpleITK)
>- SimpleITK: a layer built on top of the Insight Toolkit (ITK), intended to simplify and facilitate ITK's use in rapid prototyping, education and interpreted languages.

>- License: Apache License 2.0
>- Languages: `Swig`
>- Tags: c-plus-plus, csharp, image-analysis, image-processing, itk, java, lua, python, r, registration, ruby, segmentation, simpleitk, swig, tcl
>- Forks:	223 
>- Issues:	71
>- Watchers:	1032
>- Last updated: 2026-01-16

### Simpleitk <a name="simpleitk"></a>
- [SimpleITK](https://github.com/SimpleITK/SimpleITK)
>- SimpleITK: a layer built on top of the Insight Toolkit (ITK), intended to simplify and facilitate ITK's use in rapid prototyping, education and interpreted languages.

>- License: Apache License 2.0
>- Languages: `Swig`
>- Tags: c-plus-plus, csharp, image-analysis, image-processing, itk, java, lua, python, r, registration, ruby, segmentation, simpleitk, swig, tcl
>- Forks:	223 
>- Issues:	71
>- Watchers:	1032
>- Last updated: 2026-01-16

### Swig <a name="swig"></a>
- [SimpleITK](https://github.com/SimpleITK/SimpleITK)
>- SimpleITK: a layer built on top of the Insight Toolkit (ITK), intended to simplify and facilitate ITK's use in rapid prototyping, education and interpreted languages.

>- License: Apache License 2.0
>- Languages: `Swig`
>- Tags: c-plus-plus, csharp, image-analysis, image-processing, itk, java, lua, python, r, registration, ruby, segmentation, simpleitk, swig, tcl
>- Forks:	223 
>- Issues:	71
>- Watchers:	1032
>- Last updated: 2026-01-16

### Tcl <a name="tcl"></a>
- [SimpleITK](https://github.com/SimpleITK/SimpleITK)
>- SimpleITK: a layer built on top of the Insight Toolkit (ITK), intended to simplify and facilitate ITK's use in rapid prototyping, education and interpreted languages.

>- License: Apache License 2.0
>- Languages: `Swig`
>- Tags: c-plus-plus, csharp, image-analysis, image-processing, itk, java, lua, python, r, registration, ruby, segmentation, simpleitk, swig, tcl
>- Forks:	223 
>- Issues:	71
>- Watchers:	1032
>- Last updated: 2026-01-16

### Bart-Toolbox <a name="bart-toolbox"></a>
- [bart](https://github.com/mrirecon/bart)
>- BART: Toolbox for Computational Magnetic Resonance Imaging

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `C`
>- Tags: bart-toolbox, compressed-sensing, computational-imaging, deep-learning, iterative-methods, mri
>- Forks:	175 
>- Issues:	22
>- Watchers:	354
>- Last updated: 2026-01-14

### Compressed-Sensing <a name="compressed-sensing"></a>
- [bart](https://github.com/mrirecon/bart)
>- BART: Toolbox for Computational Magnetic Resonance Imaging

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `C`
>- Tags: bart-toolbox, compressed-sensing, computational-imaging, deep-learning, iterative-methods, mri
>- Forks:	175 
>- Issues:	22
>- Watchers:	354
>- Last updated: 2026-01-14

### Computational-Imaging <a name="computational-imaging"></a>
- [bart](https://github.com/mrirecon/bart)
>- BART: Toolbox for Computational Magnetic Resonance Imaging

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `C`
>- Tags: bart-toolbox, compressed-sensing, computational-imaging, deep-learning, iterative-methods, mri
>- Forks:	175 
>- Issues:	22
>- Watchers:	354
>- Last updated: 2026-01-14

### Iterative-Methods <a name="iterative-methods"></a>
- [bart](https://github.com/mrirecon/bart)
>- BART: Toolbox for Computational Magnetic Resonance Imaging

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `C`
>- Tags: bart-toolbox, compressed-sensing, computational-imaging, deep-learning, iterative-methods, mri
>- Forks:	175 
>- Issues:	22
>- Watchers:	354
>- Last updated: 2026-01-14

### Fitting <a name="fitting"></a>
- [vespa](https://github.com/vespa-mrs/vespa)
>- Python tools for Magnetic Resonance Spectroscopy - Pulses, Simulation and Analysis

>- License: Other
>- Languages: `Python`
>- Tags: fitting, mrs, python, rf-pulse, simulation, spectroscopy, wxpython
>- Forks:	6 
>- Issues:	6
>- Watchers:	11
>- Last updated: 2025-11-12

### Mrs <a name="mrs"></a>
- [vespa](https://github.com/vespa-mrs/vespa)
>- Python tools for Magnetic Resonance Spectroscopy - Pulses, Simulation and Analysis

>- License: Other
>- Languages: `Python`
>- Tags: fitting, mrs, python, rf-pulse, simulation, spectroscopy, wxpython
>- Forks:	6 
>- Issues:	6
>- Watchers:	11
>- Last updated: 2025-11-12

### Rf-Pulse <a name="rf-pulse"></a>
- [vespa](https://github.com/vespa-mrs/vespa)
>- Python tools for Magnetic Resonance Spectroscopy - Pulses, Simulation and Analysis

>- License: Other
>- Languages: `Python`
>- Tags: fitting, mrs, python, rf-pulse, simulation, spectroscopy, wxpython
>- Forks:	6 
>- Issues:	6
>- Watchers:	11
>- Last updated: 2025-11-12

### Spectroscopy <a name="spectroscopy"></a>
- [vespa](https://github.com/vespa-mrs/vespa)
>- Python tools for Magnetic Resonance Spectroscopy - Pulses, Simulation and Analysis

>- License: Other
>- Languages: `Python`
>- Tags: fitting, mrs, python, rf-pulse, simulation, spectroscopy, wxpython
>- Forks:	6 
>- Issues:	6
>- Watchers:	11
>- Last updated: 2025-11-12

### Wxpython <a name="wxpython"></a>
- [vespa](https://github.com/vespa-mrs/vespa)
>- Python tools for Magnetic Resonance Spectroscopy - Pulses, Simulation and Analysis

>- License: Other
>- Languages: `Python`
>- Tags: fitting, mrs, python, rf-pulse, simulation, spectroscopy, wxpython
>- Forks:	6 
>- Issues:	6
>- Watchers:	11
>- Last updated: 2025-11-12

### Cuda <a name="cuda"></a>
- [disimpy](https://github.com/kerkelae/disimpy)
>- Massively parallel Monte Carlo diffusion MR simulator written in Python.

>- License: MIT License
>- Languages: `Python`
>- Tags: cuda, diffusion-mri, gpu-computing, monte-carlo-simulation
>- Forks:	9 
>- Issues:	5
>- Watchers:	27
>- Last updated: 2026-01-13

### Gpu-Computing <a name="gpu-computing"></a>
- [disimpy](https://github.com/kerkelae/disimpy)
>- Massively parallel Monte Carlo diffusion MR simulator written in Python.

>- License: MIT License
>- Languages: `Python`
>- Tags: cuda, diffusion-mri, gpu-computing, monte-carlo-simulation
>- Forks:	9 
>- Issues:	5
>- Watchers:	27
>- Last updated: 2026-01-13

### Monte-Carlo-Simulation <a name="monte-carlo-simulation"></a>
- [disimpy](https://github.com/kerkelae/disimpy)
>- Massively parallel Monte Carlo diffusion MR simulator written in Python.

>- License: MIT License
>- Languages: `Python`
>- Tags: cuda, diffusion-mri, gpu-computing, monte-carlo-simulation
>- Forks:	9 
>- Issues:	5
>- Watchers:	27
>- Last updated: 2026-01-13

### Augmentation <a name="augmentation"></a>
- [torchio](https://github.com/fepegar/torchio)
>- Medical imaging processing for AI applications.

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: augmentation, data-augmentation, deep-learning, machine-learning, medical-image-analysis, medical-image-computing, medical-image-processing, medical-images, medical-imaging-datasets, medical-imaging-with-deep-learning, python, pytorch
>- Forks:	255 
>- Issues:	39
>- Watchers:	2345
>- Last updated: 2026-01-13

### Data-Augmentation <a name="data-augmentation"></a>
- [torchio](https://github.com/fepegar/torchio)
>- Medical imaging processing for AI applications.

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: augmentation, data-augmentation, deep-learning, machine-learning, medical-image-analysis, medical-image-computing, medical-image-processing, medical-images, medical-imaging-datasets, medical-imaging-with-deep-learning, python, pytorch
>- Forks:	255 
>- Issues:	39
>- Watchers:	2345
>- Last updated: 2026-01-13

### Medical-Imaging-Datasets <a name="medical-imaging-datasets"></a>
- [torchio](https://github.com/fepegar/torchio)
>- Medical imaging processing for AI applications.

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: augmentation, data-augmentation, deep-learning, machine-learning, medical-image-analysis, medical-image-computing, medical-image-processing, medical-images, medical-imaging-datasets, medical-imaging-with-deep-learning, python, pytorch
>- Forks:	255 
>- Issues:	39
>- Watchers:	2345
>- Last updated: 2026-01-13

### Medical-Imaging-With-Deep-Learning <a name="medical-imaging-with-deep-learning"></a>
- [torchio](https://github.com/fepegar/torchio)
>- Medical imaging processing for AI applications.

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: augmentation, data-augmentation, deep-learning, machine-learning, medical-image-analysis, medical-image-computing, medical-image-processing, medical-images, medical-imaging-datasets, medical-imaging-with-deep-learning, python, pytorch
>- Forks:	255 
>- Issues:	39
>- Watchers:	2345
>- Last updated: 2026-01-13

### 3D-Slicer-Extension <a name="3d-slicer-extension"></a>
- [ukftractography](https://github.com/pnlbwh/ukftractography)
>- None

>- License: Other
>- Languages: `C`
>- Tags: 3d-slicer-extension
>- Forks:	31 
>- Issues:	10
>- Watchers:	31
>- Last updated: 2025-12-10

### Fetus <a name="fetus"></a>
- [AFFIRM](https://github.com/allard-shi/affirm)
>- A deep recursive fetal motion estimation and correction framework based on slice and volume affinity fusion

>- License: MIT License
>- Languages: `Python`
>- Tags: deep-learning, fetus, motion
>- Forks:	1 
>- Issues:	0
>- Watchers:	8
>- Last updated: 2025-07-27

### Motion <a name="motion"></a>
- [AFFIRM](https://github.com/allard-shi/affirm)
>- A deep recursive fetal motion estimation and correction framework based on slice and volume affinity fusion

>- License: MIT License
>- Languages: `Python`
>- Tags: deep-learning, fetus, motion
>- Forks:	1 
>- Issues:	0
>- Watchers:	8
>- Last updated: 2025-07-27

### Imaging <a name="imaging"></a>
- [RadQy](https://github.com/ccipd/MRQy)
>- RadQy is a quality assurance and checking tool for quantitative assessment of magnetic resonance imaging (MRI) and computed tomography (CT) data.

>- License: None
>- Languages: `Javascript`
>- Tags: imaging, machine-learning, medical-image-processing, medical-imaging, medical-physics, mri, python, quality-assurance, quality-control, quality-metrics
>- Forks:	37 
>- Issues:	1
>- Watchers:	109
>- Last updated: 2026-01-13

### Quality-Metrics <a name="quality-metrics"></a>
- [RadQy](https://github.com/ccipd/MRQy)
>- RadQy is a quality assurance and checking tool for quantitative assessment of magnetic resonance imaging (MRI) and computed tomography (CT) data.

>- License: None
>- Languages: `Javascript`
>- Tags: imaging, machine-learning, medical-image-processing, medical-imaging, medical-physics, mri, python, quality-assurance, quality-control, quality-metrics
>- Forks:	37 
>- Issues:	1
>- Watchers:	109
>- Last updated: 2026-01-13

### Magnetic-Field-Solver <a name="magnetic-field-solver"></a>
- [pyCoilGen](https://github.com/kev-m/pyCoilGen)
>- Magnetic Field Coil Generator for Python, ported from CoilGen

>- License: GNU General Public License v3.0
>- Languages: `Python`
>- Tags: magnetic-field-solver, magnetic-resonance-imaging, medical-physics, mri, nmr, physics
>- Forks:	8 
>- Issues:	3
>- Watchers:	20
>- Last updated: 2025-12-03

### Nmr <a name="nmr"></a>
- [pyCoilGen](https://github.com/kev-m/pyCoilGen)
>- Magnetic Field Coil Generator for Python, ported from CoilGen

>- License: GNU General Public License v3.0
>- Languages: `Python`
>- Tags: magnetic-field-solver, magnetic-resonance-imaging, medical-physics, mri, nmr, physics
>- Forks:	8 
>- Issues:	3
>- Watchers:	20
>- Last updated: 2025-12-03

### Physics <a name="physics"></a>
- [pyCoilGen](https://github.com/kev-m/pyCoilGen)
>- Magnetic Field Coil Generator for Python, ported from CoilGen

>- License: GNU General Public License v3.0
>- Languages: `Python`
>- Tags: magnetic-field-solver, magnetic-resonance-imaging, medical-physics, mri, nmr, physics
>- Forks:	8 
>- Issues:	3
>- Watchers:	20
>- Last updated: 2025-12-03

### Fusion <a name="fusion"></a>
- [sHDR](https://github.com/shakes76/sHDR)
>- HDR-MRI Algorithms from "Local contrast-enhanced MR images via high dynamic range processing" published in MRM

>- License: Other
>- Languages: `C++`
>- Tags: fusion, hdr, image, medical-image-processing, medical-imaging, mri
>- Forks:	0 
>- Issues:	0
>- Watchers:	16
>- Last updated: 2024-11-29

### Hdr <a name="hdr"></a>
- [sHDR](https://github.com/shakes76/sHDR)
>- HDR-MRI Algorithms from "Local contrast-enhanced MR images via high dynamic range processing" published in MRM

>- License: Other
>- Languages: `C++`
>- Tags: fusion, hdr, image, medical-image-processing, medical-imaging, mri
>- Forks:	0 
>- Issues:	0
>- Watchers:	16
>- Last updated: 2024-11-29

### Image <a name="image"></a>
- [sHDR](https://github.com/shakes76/sHDR)
>- HDR-MRI Algorithms from "Local contrast-enhanced MR images via high dynamic range processing" published in MRM

>- License: Other
>- Languages: `C++`
>- Tags: fusion, hdr, image, medical-image-processing, medical-imaging, mri
>- Forks:	0 
>- Issues:	0
>- Watchers:	16
>- Last updated: 2024-11-29

### Mri-Phantoms <a name="mri-phantoms"></a>
- [hazen](https://github.com/GSTT-CSC/hazen)
>- Quality assurance framework for Magnetic Resonance Imaging

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: image-processing, mri, mri-phantoms, python, qa, quality-assurance
>- Forks:	13 
>- Issues:	58
>- Watchers:	29
>- Last updated: 2026-01-14

### 3D-Segmentation <a name="3d-segmentation"></a>
- [brainchop](https://github.com/neuroneural/brainchop)
>- Brainchop: In-browser 3D MRI rendering and segmentation

>- License: MIT License
>- Languages: `Javascript`
>- Tags: 3d-segmentation, deep-learning, frontend-app, javascript, medical-imaging, mri, mri-segmentation, neuroimaging, pyodide, tensorflowjs, three-js
>- Forks:	61 
>- Issues:	5
>- Watchers:	512
>- Last updated: 2026-01-16

### Frontend-App <a name="frontend-app"></a>
- [brainchop](https://github.com/neuroneural/brainchop)
>- Brainchop: In-browser 3D MRI rendering and segmentation

>- License: MIT License
>- Languages: `Javascript`
>- Tags: 3d-segmentation, deep-learning, frontend-app, javascript, medical-imaging, mri, mri-segmentation, neuroimaging, pyodide, tensorflowjs, three-js
>- Forks:	61 
>- Issues:	5
>- Watchers:	512
>- Last updated: 2026-01-16

### Javascript <a name="javascript"></a>
- [brainchop](https://github.com/neuroneural/brainchop)
>- Brainchop: In-browser 3D MRI rendering and segmentation

>- License: MIT License
>- Languages: `Javascript`
>- Tags: 3d-segmentation, deep-learning, frontend-app, javascript, medical-imaging, mri, mri-segmentation, neuroimaging, pyodide, tensorflowjs, three-js
>- Forks:	61 
>- Issues:	5
>- Watchers:	512
>- Last updated: 2026-01-16

### Mri-Segmentation <a name="mri-segmentation"></a>
- [brainchop](https://github.com/neuroneural/brainchop)
>- Brainchop: In-browser 3D MRI rendering and segmentation

>- License: MIT License
>- Languages: `Javascript`
>- Tags: 3d-segmentation, deep-learning, frontend-app, javascript, medical-imaging, mri, mri-segmentation, neuroimaging, pyodide, tensorflowjs, three-js
>- Forks:	61 
>- Issues:	5
>- Watchers:	512
>- Last updated: 2026-01-16

### Pyodide <a name="pyodide"></a>
- [brainchop](https://github.com/neuroneural/brainchop)
>- Brainchop: In-browser 3D MRI rendering and segmentation

>- License: MIT License
>- Languages: `Javascript`
>- Tags: 3d-segmentation, deep-learning, frontend-app, javascript, medical-imaging, mri, mri-segmentation, neuroimaging, pyodide, tensorflowjs, three-js
>- Forks:	61 
>- Issues:	5
>- Watchers:	512
>- Last updated: 2026-01-16

### Tensorflowjs <a name="tensorflowjs"></a>
- [brainchop](https://github.com/neuroneural/brainchop)
>- Brainchop: In-browser 3D MRI rendering and segmentation

>- License: MIT License
>- Languages: `Javascript`
>- Tags: 3d-segmentation, deep-learning, frontend-app, javascript, medical-imaging, mri, mri-segmentation, neuroimaging, pyodide, tensorflowjs, three-js
>- Forks:	61 
>- Issues:	5
>- Watchers:	512
>- Last updated: 2026-01-16

### Three-Js <a name="three-js"></a>
- [brainchop](https://github.com/neuroneural/brainchop)
>- Brainchop: In-browser 3D MRI rendering and segmentation

>- License: MIT License
>- Languages: `Javascript`
>- Tags: 3d-segmentation, deep-learning, frontend-app, javascript, medical-imaging, mri, mri-segmentation, neuroimaging, pyodide, tensorflowjs, three-js
>- Forks:	61 
>- Issues:	5
>- Watchers:	512
>- Last updated: 2026-01-16

### Pet-Mr <a name="pet-mr"></a>
- [SIRF](https://github.com/SyneRBI/SIRF?tab=readme-ov-file)
>- Main repository for the CCP SynerBI software

>- License: Other
>- Languages: `C++`
>- Tags: image-reconstruction, medical-imaging, pet-mr
>- Forks:	29 
>- Issues:	167
>- Watchers:	68
>- Last updated: 2026-01-15

### Quality-Reporter <a name="quality-reporter"></a>
- [mriqc](https://github.com/nipreps/mriqc)
>- Automated Quality Control and visual reports for Quality Assessment of structural (T1w, T2w) and functional MRI of the brain

>- License: Apache License 2.0
>- Languages: `Python`
>- Tags: machine-learning, mri, neuroimaging, quality-control, quality-reporter
>- Forks:	134 
>- Issues:	94
>- Watchers:	343
>- Last updated: 2026-01-16

### 3D-Printing <a name="3d-printing"></a>
- [Slicer](https://github.com/Slicer/Slicer)
>- Multi-platform, free open source software for visualization and image computing.

>- License: Other
>- Languages: `C++`
>- Tags: 3d-printing, 3d-slicer, c-plus-plus, computed-tomography, image-guided-therapy, image-processing, itk, kitware, medical-image-computing, medical-imaging, national-institutes-of-health, neuroimaging, nih, python, qt, registration, segmentation, tcia-dac, tractography, vtk
>- Forks:	685 
>- Issues:	647
>- Watchers:	2282
>- Last updated: 2026-01-16

### 3D-Slicer <a name="3d-slicer"></a>
- [Slicer](https://github.com/Slicer/Slicer)
>- Multi-platform, free open source software for visualization and image computing.

>- License: Other
>- Languages: `C++`
>- Tags: 3d-printing, 3d-slicer, c-plus-plus, computed-tomography, image-guided-therapy, image-processing, itk, kitware, medical-image-computing, medical-imaging, national-institutes-of-health, neuroimaging, nih, python, qt, registration, segmentation, tcia-dac, tractography, vtk
>- Forks:	685 
>- Issues:	647
>- Watchers:	2282
>- Last updated: 2026-01-16

### Computed-Tomography <a name="computed-tomography"></a>
- [Slicer](https://github.com/Slicer/Slicer)
>- Multi-platform, free open source software for visualization and image computing.

>- License: Other
>- Languages: `C++`
>- Tags: 3d-printing, 3d-slicer, c-plus-plus, computed-tomography, image-guided-therapy, image-processing, itk, kitware, medical-image-computing, medical-imaging, national-institutes-of-health, neuroimaging, nih, python, qt, registration, segmentation, tcia-dac, tractography, vtk
>- Forks:	685 
>- Issues:	647
>- Watchers:	2282
>- Last updated: 2026-01-16

### Image-Guided-Therapy <a name="image-guided-therapy"></a>
- [Slicer](https://github.com/Slicer/Slicer)
>- Multi-platform, free open source software for visualization and image computing.

>- License: Other
>- Languages: `C++`
>- Tags: 3d-printing, 3d-slicer, c-plus-plus, computed-tomography, image-guided-therapy, image-processing, itk, kitware, medical-image-computing, medical-imaging, national-institutes-of-health, neuroimaging, nih, python, qt, registration, segmentation, tcia-dac, tractography, vtk
>- Forks:	685 
>- Issues:	647
>- Watchers:	2282
>- Last updated: 2026-01-16

### Kitware <a name="kitware"></a>
- [Slicer](https://github.com/Slicer/Slicer)
>- Multi-platform, free open source software for visualization and image computing.

>- License: Other
>- Languages: `C++`
>- Tags: 3d-printing, 3d-slicer, c-plus-plus, computed-tomography, image-guided-therapy, image-processing, itk, kitware, medical-image-computing, medical-imaging, national-institutes-of-health, neuroimaging, nih, python, qt, registration, segmentation, tcia-dac, tractography, vtk
>- Forks:	685 
>- Issues:	647
>- Watchers:	2282
>- Last updated: 2026-01-16

### National-Institutes-Of-Health <a name="national-institutes-of-health"></a>
- [Slicer](https://github.com/Slicer/Slicer)
>- Multi-platform, free open source software for visualization and image computing.

>- License: Other
>- Languages: `C++`
>- Tags: 3d-printing, 3d-slicer, c-plus-plus, computed-tomography, image-guided-therapy, image-processing, itk, kitware, medical-image-computing, medical-imaging, national-institutes-of-health, neuroimaging, nih, python, qt, registration, segmentation, tcia-dac, tractography, vtk
>- Forks:	685 
>- Issues:	647
>- Watchers:	2282
>- Last updated: 2026-01-16

### Nih <a name="nih"></a>
- [Slicer](https://github.com/Slicer/Slicer)
>- Multi-platform, free open source software for visualization and image computing.

>- License: Other
>- Languages: `C++`
>- Tags: 3d-printing, 3d-slicer, c-plus-plus, computed-tomography, image-guided-therapy, image-processing, itk, kitware, medical-image-computing, medical-imaging, national-institutes-of-health, neuroimaging, nih, python, qt, registration, segmentation, tcia-dac, tractography, vtk
>- Forks:	685 
>- Issues:	647
>- Watchers:	2282
>- Last updated: 2026-01-16

### Qt <a name="qt"></a>
- [Slicer](https://github.com/Slicer/Slicer)
>- Multi-platform, free open source software for visualization and image computing.

>- License: Other
>- Languages: `C++`
>- Tags: 3d-printing, 3d-slicer, c-plus-plus, computed-tomography, image-guided-therapy, image-processing, itk, kitware, medical-image-computing, medical-imaging, national-institutes-of-health, neuroimaging, nih, python, qt, registration, segmentation, tcia-dac, tractography, vtk
>- Forks:	685 
>- Issues:	647
>- Watchers:	2282
>- Last updated: 2026-01-16

### Tcia-Dac <a name="tcia-dac"></a>
- [Slicer](https://github.com/Slicer/Slicer)
>- Multi-platform, free open source software for visualization and image computing.

>- License: Other
>- Languages: `C++`
>- Tags: 3d-printing, 3d-slicer, c-plus-plus, computed-tomography, image-guided-therapy, image-processing, itk, kitware, medical-image-computing, medical-imaging, national-institutes-of-health, neuroimaging, nih, python, qt, registration, segmentation, tcia-dac, tractography, vtk
>- Forks:	685 
>- Issues:	647
>- Watchers:	2282
>- Last updated: 2026-01-16

### Vtk <a name="vtk"></a>
- [Slicer](https://github.com/Slicer/Slicer)
>- Multi-platform, free open source software for visualization and image computing.

>- License: Other
>- Languages: `C++`
>- Tags: 3d-printing, 3d-slicer, c-plus-plus, computed-tomography, image-guided-therapy, image-processing, itk, kitware, medical-image-computing, medical-imaging, national-institutes-of-health, neuroimaging, nih, python, qt, registration, segmentation, tcia-dac, tractography, vtk
>- Forks:	685 
>- Issues:	647
>- Watchers:	2282
>- Last updated: 2026-01-16

### Ai <a name="ai"></a>
- [quickNAT_pytorch](https://github.com/ai-med/quickNAT_pytorch)
>- PyTorch Implementation of QuickNAT and Bayesian QuickNAT, a fast brain MRI segmentation framework with segmentation Quality control using structure-wise uncertainty

>- License: MIT License
>- Languages: `Python`
>- Tags: ai, bayesian, biomarkers, brain-imaging, computer-vision, convolutional-neural-networks, deep-learning, machine-learning, medical-imaging, mri-images, neuroanatomy, pytorch, quality-control, segmentation, uncertainty
>- Forks:	36 
>- Issues:	7
>- Watchers:	104
>- Last updated: 2025-12-03

### Bayesian <a name="bayesian"></a>
- [quickNAT_pytorch](https://github.com/ai-med/quickNAT_pytorch)
>- PyTorch Implementation of QuickNAT and Bayesian QuickNAT, a fast brain MRI segmentation framework with segmentation Quality control using structure-wise uncertainty

>- License: MIT License
>- Languages: `Python`
>- Tags: ai, bayesian, biomarkers, brain-imaging, computer-vision, convolutional-neural-networks, deep-learning, machine-learning, medical-imaging, mri-images, neuroanatomy, pytorch, quality-control, segmentation, uncertainty
>- Forks:	36 
>- Issues:	7
>- Watchers:	104
>- Last updated: 2025-12-03

### Biomarkers <a name="biomarkers"></a>
- [quickNAT_pytorch](https://github.com/ai-med/quickNAT_pytorch)
>- PyTorch Implementation of QuickNAT and Bayesian QuickNAT, a fast brain MRI segmentation framework with segmentation Quality control using structure-wise uncertainty

>- License: MIT License
>- Languages: `Python`
>- Tags: ai, bayesian, biomarkers, brain-imaging, computer-vision, convolutional-neural-networks, deep-learning, machine-learning, medical-imaging, mri-images, neuroanatomy, pytorch, quality-control, segmentation, uncertainty
>- Forks:	36 
>- Issues:	7
>- Watchers:	104
>- Last updated: 2025-12-03

### Neuroanatomy <a name="neuroanatomy"></a>
- [quickNAT_pytorch](https://github.com/ai-med/quickNAT_pytorch)
>- PyTorch Implementation of QuickNAT and Bayesian QuickNAT, a fast brain MRI segmentation framework with segmentation Quality control using structure-wise uncertainty

>- License: MIT License
>- Languages: `Python`
>- Tags: ai, bayesian, biomarkers, brain-imaging, computer-vision, convolutional-neural-networks, deep-learning, machine-learning, medical-imaging, mri-images, neuroanatomy, pytorch, quality-control, segmentation, uncertainty
>- Forks:	36 
>- Issues:	7
>- Watchers:	104
>- Last updated: 2025-12-03

### Uncertainty <a name="uncertainty"></a>
- [quickNAT_pytorch](https://github.com/ai-med/quickNAT_pytorch)
>- PyTorch Implementation of QuickNAT and Bayesian QuickNAT, a fast brain MRI segmentation framework with segmentation Quality control using structure-wise uncertainty

>- License: MIT License
>- Languages: `Python`
>- Tags: ai, bayesian, biomarkers, brain-imaging, computer-vision, convolutional-neural-networks, deep-learning, machine-learning, medical-imaging, mri-images, neuroanatomy, pytorch, quality-control, segmentation, uncertainty
>- Forks:	36 
>- Issues:	7
>- Watchers:	104
>- Last updated: 2025-12-03

### Glioblastoma <a name="glioblastoma"></a>
- [BraTS-Toolkit](https://github.com/neuronflow/BraTS-Toolkit)
>- Code to preprocess, segment, and fuse glioma MRI scans based on the BraTS Toolkit manuscript.

>- License: GNU Affero General Public License v3.0
>- Languages: `Python`
>- Tags: glioblastoma, glioma, medical-imaging, mri, segmentation
>- Forks:	14 
>- Issues:	9
>- Watchers:	98
>- Last updated: 2026-01-13

### Glioma <a name="glioma"></a>
- [BraTS-Toolkit](https://github.com/neuronflow/BraTS-Toolkit)
>- Code to preprocess, segment, and fuse glioma MRI scans based on the BraTS Toolkit manuscript.

>- License: GNU Affero General Public License v3.0
>- Languages: `Python`
>- Tags: glioblastoma, glioma, medical-imaging, mri, segmentation
>- Forks:	14 
>- Issues:	9
>- Watchers:	98
>- Last updated: 2026-01-13

### Neural-Networks <a name="neural-networks"></a>
- [deepmedic](https://github.com/deepmedic/deepmedic)
>- Efficient Multi-Scale 3D Convolutional Neural Network for Segmentation of 3D Medical Scans

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: convolutional-neural-networks, deep-learning, machine-learning, medical-imaging, neural-networks
>- Forks:	347 
>- Issues:	23
>- Watchers:	1058
>- Last updated: 2025-12-23

### Fastmri <a name="fastmri"></a>
- [fastMRI](https://github.com/facebookresearch/fastMRI)
>- A large-scale dataset of both raw MRI measurements and clinical MRI images.

>- License: MIT License
>- Languages: `Python`
>- Tags: convolutional-neural-networks, deep-learning, fastmri, fastmri-challenge, fastmri-dataset, medical-imaging, mri, mri-reconstruction, pytorch
>- Forks:	416 
>- Issues:	18
>- Watchers:	1498
>- Last updated: 2026-01-10

### Fastmri-Dataset <a name="fastmri-dataset"></a>
- [fastMRI](https://github.com/facebookresearch/fastMRI)
>- A large-scale dataset of both raw MRI measurements and clinical MRI images.

>- License: MIT License
>- Languages: `Python`
>- Tags: convolutional-neural-networks, deep-learning, fastmri, fastmri-challenge, fastmri-dataset, medical-imaging, mri, mri-reconstruction, pytorch
>- Forks:	416 
>- Issues:	18
>- Watchers:	1498
>- Last updated: 2026-01-10

### Dicom-Converter <a name="dicom-converter"></a>
- [mritopng](https://github.com/danishm/mritopng)
>- A simple python module to make it easy to batch convert DICOM files to PNG images.

>- License: MIT License
>- Languages: `Python`
>- Tags: dicom, dicom-converter, dicom-images, medical, medical-images, png, python
>- Forks:	51 
>- Issues:	5
>- Watchers:	146
>- Last updated: 2025-10-31

### Dicom-Images <a name="dicom-images"></a>
- [mritopng](https://github.com/danishm/mritopng)
>- A simple python module to make it easy to batch convert DICOM files to PNG images.

>- License: MIT License
>- Languages: `Python`
>- Tags: dicom, dicom-converter, dicom-images, medical, medical-images, png, python
>- Forks:	51 
>- Issues:	5
>- Watchers:	146
>- Last updated: 2025-10-31

### Medical <a name="medical"></a>
- [mritopng](https://github.com/danishm/mritopng)
>- A simple python module to make it easy to batch convert DICOM files to PNG images.

>- License: MIT License
>- Languages: `Python`
>- Tags: dicom, dicom-converter, dicom-images, medical, medical-images, png, python
>- Forks:	51 
>- Issues:	5
>- Watchers:	146
>- Last updated: 2025-10-31

### Png <a name="png"></a>
- [mritopng](https://github.com/danishm/mritopng)
>- A simple python module to make it easy to batch convert DICOM files to PNG images.

>- License: MIT License
>- Languages: `Python`
>- Tags: dicom, dicom-converter, dicom-images, medical, medical-images, png, python
>- Forks:	51 
>- Issues:	5
>- Watchers:	146
>- Last updated: 2025-10-31

### 3D-Reconstruction <a name="3d-reconstruction"></a>
- [NeSVoR](https://github.com/daviddmc/NeSVoR)
>- NeSVoR is a package for GPU-accelerated slice-to-volume reconstruction.

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-reconstruction, 3d-visualization, deep-learning, image-reconstruction, image-registration, implicit-neural-representation, medical-imaging, mri, nerf, neural-network, neural-rendering, pytorch, segmentation, super-resolution, transformers
>- Forks:	21 
>- Issues:	7
>- Watchers:	97
>- Last updated: 2026-01-07

### 3D-Visualization <a name="3d-visualization"></a>
- [NeSVoR](https://github.com/daviddmc/NeSVoR)
>- NeSVoR is a package for GPU-accelerated slice-to-volume reconstruction.

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-reconstruction, 3d-visualization, deep-learning, image-reconstruction, image-registration, implicit-neural-representation, medical-imaging, mri, nerf, neural-network, neural-rendering, pytorch, segmentation, super-resolution, transformers
>- Forks:	21 
>- Issues:	7
>- Watchers:	97
>- Last updated: 2026-01-07

### Implicit-Neural-Representation <a name="implicit-neural-representation"></a>
- [NeSVoR](https://github.com/daviddmc/NeSVoR)
>- NeSVoR is a package for GPU-accelerated slice-to-volume reconstruction.

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-reconstruction, 3d-visualization, deep-learning, image-reconstruction, image-registration, implicit-neural-representation, medical-imaging, mri, nerf, neural-network, neural-rendering, pytorch, segmentation, super-resolution, transformers
>- Forks:	21 
>- Issues:	7
>- Watchers:	97
>- Last updated: 2026-01-07

### Nerf <a name="nerf"></a>
- [NeSVoR](https://github.com/daviddmc/NeSVoR)
>- NeSVoR is a package for GPU-accelerated slice-to-volume reconstruction.

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-reconstruction, 3d-visualization, deep-learning, image-reconstruction, image-registration, implicit-neural-representation, medical-imaging, mri, nerf, neural-network, neural-rendering, pytorch, segmentation, super-resolution, transformers
>- Forks:	21 
>- Issues:	7
>- Watchers:	97
>- Last updated: 2026-01-07

### Neural-Network <a name="neural-network"></a>
- [NeSVoR](https://github.com/daviddmc/NeSVoR)
>- NeSVoR is a package for GPU-accelerated slice-to-volume reconstruction.

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-reconstruction, 3d-visualization, deep-learning, image-reconstruction, image-registration, implicit-neural-representation, medical-imaging, mri, nerf, neural-network, neural-rendering, pytorch, segmentation, super-resolution, transformers
>- Forks:	21 
>- Issues:	7
>- Watchers:	97
>- Last updated: 2026-01-07

### Neural-Rendering <a name="neural-rendering"></a>
- [NeSVoR](https://github.com/daviddmc/NeSVoR)
>- NeSVoR is a package for GPU-accelerated slice-to-volume reconstruction.

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-reconstruction, 3d-visualization, deep-learning, image-reconstruction, image-registration, implicit-neural-representation, medical-imaging, mri, nerf, neural-network, neural-rendering, pytorch, segmentation, super-resolution, transformers
>- Forks:	21 
>- Issues:	7
>- Watchers:	97
>- Last updated: 2026-01-07

### Transformers <a name="transformers"></a>
- [NeSVoR](https://github.com/daviddmc/NeSVoR)
>- NeSVoR is a package for GPU-accelerated slice-to-volume reconstruction.

>- License: MIT License
>- Languages: `Python`
>- Tags: 3d-reconstruction, 3d-visualization, deep-learning, image-reconstruction, image-registration, implicit-neural-representation, medical-imaging, mri, nerf, neural-network, neural-rendering, pytorch, segmentation, super-resolution, transformers
>- Forks:	21 
>- Issues:	7
>- Watchers:	97
>- Last updated: 2026-01-07

### Afni-Brik-Head <a name="afni-brik-head"></a>
- [nibabel](https://github.com/nipy/nibabel)
>- Python package to access a cacophony of neuro-imaging file formats

>- License: Other
>- Languages: `Python`
>- Tags: afni-brik-head, brain-imaging, cifti-2, data-formats, dicom, ecat, gifti, minc, neuroimaging, nifti, python, streamlines, tck, trk
>- Forks:	273 
>- Issues:	145
>- Watchers:	759
>- Last updated: 2026-01-15

### Cifti-2 <a name="cifti-2"></a>
- [nibabel](https://github.com/nipy/nibabel)
>- Python package to access a cacophony of neuro-imaging file formats

>- License: Other
>- Languages: `Python`
>- Tags: afni-brik-head, brain-imaging, cifti-2, data-formats, dicom, ecat, gifti, minc, neuroimaging, nifti, python, streamlines, tck, trk
>- Forks:	273 
>- Issues:	145
>- Watchers:	759
>- Last updated: 2026-01-15

### Data-Formats <a name="data-formats"></a>
- [nibabel](https://github.com/nipy/nibabel)
>- Python package to access a cacophony of neuro-imaging file formats

>- License: Other
>- Languages: `Python`
>- Tags: afni-brik-head, brain-imaging, cifti-2, data-formats, dicom, ecat, gifti, minc, neuroimaging, nifti, python, streamlines, tck, trk
>- Forks:	273 
>- Issues:	145
>- Watchers:	759
>- Last updated: 2026-01-15

### Ecat <a name="ecat"></a>
- [nibabel](https://github.com/nipy/nibabel)
>- Python package to access a cacophony of neuro-imaging file formats

>- License: Other
>- Languages: `Python`
>- Tags: afni-brik-head, brain-imaging, cifti-2, data-formats, dicom, ecat, gifti, minc, neuroimaging, nifti, python, streamlines, tck, trk
>- Forks:	273 
>- Issues:	145
>- Watchers:	759
>- Last updated: 2026-01-15

### Gifti <a name="gifti"></a>
- [nibabel](https://github.com/nipy/nibabel)
>- Python package to access a cacophony of neuro-imaging file formats

>- License: Other
>- Languages: `Python`
>- Tags: afni-brik-head, brain-imaging, cifti-2, data-formats, dicom, ecat, gifti, minc, neuroimaging, nifti, python, streamlines, tck, trk
>- Forks:	273 
>- Issues:	145
>- Watchers:	759
>- Last updated: 2026-01-15

### Minc <a name="minc"></a>
- [nibabel](https://github.com/nipy/nibabel)
>- Python package to access a cacophony of neuro-imaging file formats

>- License: Other
>- Languages: `Python`
>- Tags: afni-brik-head, brain-imaging, cifti-2, data-formats, dicom, ecat, gifti, minc, neuroimaging, nifti, python, streamlines, tck, trk
>- Forks:	273 
>- Issues:	145
>- Watchers:	759
>- Last updated: 2026-01-15

### Streamlines <a name="streamlines"></a>
- [nibabel](https://github.com/nipy/nibabel)
>- Python package to access a cacophony of neuro-imaging file formats

>- License: Other
>- Languages: `Python`
>- Tags: afni-brik-head, brain-imaging, cifti-2, data-formats, dicom, ecat, gifti, minc, neuroimaging, nifti, python, streamlines, tck, trk
>- Forks:	273 
>- Issues:	145
>- Watchers:	759
>- Last updated: 2026-01-15

### Tck <a name="tck"></a>
- [nibabel](https://github.com/nipy/nibabel)
>- Python package to access a cacophony of neuro-imaging file formats

>- License: Other
>- Languages: `Python`
>- Tags: afni-brik-head, brain-imaging, cifti-2, data-formats, dicom, ecat, gifti, minc, neuroimaging, nifti, python, streamlines, tck, trk
>- Forks:	273 
>- Issues:	145
>- Watchers:	759
>- Last updated: 2026-01-15

### Trk <a name="trk"></a>
- [nibabel](https://github.com/nipy/nibabel)
>- Python package to access a cacophony of neuro-imaging file formats

>- License: Other
>- Languages: `Python`
>- Tags: afni-brik-head, brain-imaging, cifti-2, data-formats, dicom, ecat, gifti, minc, neuroimaging, nifti, python, streamlines, tck, trk
>- Forks:	273 
>- Issues:	145
>- Watchers:	759
>- Last updated: 2026-01-15

### Brain-Mri <a name="brain-mri"></a>
- [nilearn](https://github.com/nilearn/nilearn)
>- Machine learning for NeuroImaging in Python

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: brain-connectivity, brain-imaging, brain-mri, decoding, fmri, machine-learning, mvpa, neuroimaging, python
>- Forks:	638 
>- Issues:	272
>- Watchers:	1353
>- Last updated: 2026-01-16

### Decoding <a name="decoding"></a>
- [nilearn](https://github.com/nilearn/nilearn)
>- Machine learning for NeuroImaging in Python

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: brain-connectivity, brain-imaging, brain-mri, decoding, fmri, machine-learning, mvpa, neuroimaging, python
>- Forks:	638 
>- Issues:	272
>- Watchers:	1353
>- Last updated: 2026-01-16

### Mvpa <a name="mvpa"></a>
- [nilearn](https://github.com/nilearn/nilearn)
>- Machine learning for NeuroImaging in Python

>- License: BSD 3-Clause "New" or "Revised" License
>- Languages: `Python`
>- Tags: brain-connectivity, brain-imaging, brain-mri, decoding, fmri, machine-learning, mvpa, neuroimaging, python
>- Forks:	638 
>- Issues:	272
>- Watchers:	1353
>- Last updated: 2026-01-16

### Cardiac <a name="cardiac"></a>
- [KomaMRI.jl](https://github.com/JuliaHealth/KomaMRI.jl)
>- Koma is a Pulseq-compatible framework to efficiently simulate Magnetic Resonance Imaging (MRI) acquisitions. The main focus of this package is to simulate general scenarios that could arise in pulse sequence development.

>- License: MIT License
>- Languages: `Julia`
>- Tags: cardiac, diffusion, diffusion-mri, gpu-acceleration, mri, simulation
>- Forks:	31 
>- Issues:	103
>- Watchers:	176
>- Last updated: 2026-01-06

### Diffusion <a name="diffusion"></a>
- [KomaMRI.jl](https://github.com/JuliaHealth/KomaMRI.jl)
>- Koma is a Pulseq-compatible framework to efficiently simulate Magnetic Resonance Imaging (MRI) acquisitions. The main focus of this package is to simulate general scenarios that could arise in pulse sequence development.

>- License: MIT License
>- Languages: `Julia`
>- Tags: cardiac, diffusion, diffusion-mri, gpu-acceleration, mri, simulation
>- Forks:	31 
>- Issues:	103
>- Watchers:	176
>- Last updated: 2026-01-06

### Gpu-Acceleration <a name="gpu-acceleration"></a>
- [KomaMRI.jl](https://github.com/JuliaHealth/KomaMRI.jl)
>- Koma is a Pulseq-compatible framework to efficiently simulate Magnetic Resonance Imaging (MRI) acquisitions. The main focus of this package is to simulate general scenarios that could arise in pulse sequence development.

>- License: MIT License
>- Languages: `Julia`
>- Tags: cardiac, diffusion, diffusion-mri, gpu-acceleration, mri, simulation
>- Forks:	31 
>- Issues:	103
>- Watchers:	176
>- Last updated: 2026-01-06



## Languages
### Python <a name="python"></a>
### C++ <a name="c++"></a>
### Julia <a name="julia"></a>
### Jupyter Notebook <a name="jupyter-notebook"></a>
### C <a name="c"></a>
### Javascript <a name="javascript"></a>
### R <a name="r"></a>
### Matlab <a name="matlab"></a>
### Swig <a name="swig"></a>
### Typescript <a name="typescript"></a>
