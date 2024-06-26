# MRI Tools
![license](https://img.shields.io/github/license/abdrysdale/mri-tools.svg)

A collection of free and open-source software software tools for use in MRI.
Free is meant as in free beer (gratis) and freedom (libre).

To add a project edit the repos.toml file and submit a pull request.
Repositories are stored in the toml file in the format:

```toml
[repo-name]
languages = ["repo-lang-1", "repo-lang-2"]
link = "repo-link"
license = "repo-license"
description = "A short description about the repo"
tags = ["repo-tag-1", "repo-tag-2"]
```

## Table of Contents
- [stats](#stats)
- [tags](#tags)
	- [analysis](#analysis)
	- [processing](#processing)
	- [reconstruction](#reconstruction)
	- [ml](#ml)
	- [simulation](#simulation)
	- [segmentation](#segmentation)
	- [brain](#brain)
	- [data](#data)
	- [visualisation](#visualisation)
	- [qa](#qa)
	- [fetal](#fetal)
	- [renal](#renal)
	- [spinal](#spinal)
	- [muscle](#muscle)
	- [safety](#safety)
- [languages](#languages)
	- [python](#python)
	- [c++](#c++)
	- [julia](#julia)
	- [c](#c)
	- [javascript](#javascript)
	- [r](#r)
	- [jupyter](#jupyter)

## Stats
- Total repos: 80
- Languages:

| Language | Count |
|---|---|
| python | 62 |
| c++ | 16 |
| julia | 8 |
| c | 6 |
| javascript | 4 |
| r | 2 |
| jupyter | 1 |

- Tags:

| Tag | Count |
|---|---|
| analysis | 21 |
| processing | 18 |
| reconstruction | 17 |
| ml | 14 |
| simulation | 13 |
| segmentation | 13 |
| brain | 11 |
| data | 7 |
| visualisation | 6 |
| qa | 6 |
| fetal | 6 |
| renal | 1 |
| spinal | 1 |
| muscle | 1 |
| safety | 1 |

- Licenses:

| Licence | Count |
|---|---|
| mit | 27 |
| apache | 19 |
| bsd | 14 |
| gplv3 | 8 |
| none | 4 |
| agplv3 | 3 |
| lgplv3 | 2 |
| mpl | 2 |
| gplv2 | 1 |



## Tags
### Analysis <a name="analysis"></a>
- [slicer](https://github.com/Slicer/Slicer)
>- Languages: `Python`, `C++`
>- License: BSD
>- Tags: simulation, analysis, visualisation
>- A open source software package for visualization and image analysis.

- [ukat](https://github.com/UKRIN-MAPS/ukat)
>- Languages: `Python`
>- License: GPLv3
>- Tags: analysis, renal
>- A ukat is a vendor agnostic framework for the analysis of quantitative renal mri data

- [freesurfer](https://github.com/freesurfer/freesurfer)
>- Languages: `C++`, `C`, `Python`
>- License: GPLv3
>- Tags: analysis, visualisation, brain
>- A analysis and visualization of neuroimaging data from cross-sectional and longitudinal studies

- [simple-itk](https://github.com/SimpleITK/SimpleITK)
>- Languages: `C++`, `Python`, `R`
>- License: Apache
>- Tags: segmentation, analysis
>- A image analysis toolkit with a large number of components supporting general filtering operations, image segmentation and registration

- [quit](https://github.com/spinicist/QUIT)
>- Languages: `C++`, `Python`
>- License: MPL
>- Tags: analysis
>- A collection of programs for processing quantitative mri data

- [madym](https://gitlab.com/manchester_qbi/manchester_qbi_public/madym_cxx)
>- Languages: `C++`
>- License: Apache
>- Tags: analysis, processing
>- A c++ toolkit for quantative dce-mri and dwi-mri analysis

- [myoqmri](https://github.com/fsantini/MyoQMRI)
>- Languages: `Python`
>- License: GPLv3
>- Tags: analysis, muscle
>- A quantitative mri of the muscles

- [mrqy](https://github.com/ccipd/MRQy)
>- Languages: `Javascript`, `Python`
>- License: BSD
>- Tags: qa, analysis
>- A generate several tags and noise/information measurements for quality assessment

- [fastmri](https://github.com/facebookresearch/fastMRI)
>- Languages: `Python`
>- License: MIT
>- Tags: ml, analysis, data
>- A research project from facebook ai research (fair) and nyu langone health to investigate the use of ai to make mri scans faster

- [affirm](https://github.com/allard-shi/affirm)
>- Languages: `Python`
>- License: MIT
>- Tags: analysis, fetal
>- A deep recursive fetal motion estimation and correction based on slice and volume affinity fusion

- [mri-research-tools](https://github.com/korbinian90/MriResearchTools.jl)
>- Languages: `Julia`
>- License: MIT
>- Tags: processing, analysis, simulation
>- A specialized tools for mri

- [dcemri](https://github.com/davidssmith/DCEMRI.jl)
>- Languages: `Julia`
>- License: MIT
>- Tags: analysis
>- A open source toolkit for dynamic contrast enhanced mri analysis

- [qslprep](https://github.com/PennLINC/qsiprep)
>- Languages: `Python`
>- License: BSD
>- Tags: processing, reconstruction, analysis
>- A preprocessing and reconstruction of diffusion mri

- [braingraph](https://github.com/cwatson/brainGraph)
>- Languages: `R`
>- License: None
>- Tags: analysis
>- A r package for performing graph theory analyses of brain mri data

- [mriqc](https://github.com/nipreps/mriqc)
>- Languages: `Javascript`, `Python`
>- License: Apache
>- Tags: qa, analysis
>- A extracts no-reference iqms (image quality metrics) from structural (t1w and t2w) and functional mri (magnetic resonance imaging) data

- [nipype](https://github.com/nipy/nipype)
>- Languages: `Python`
>- License: Apache
>- Tags: analysis, brain
>- A python project that provides a uniform interface to existing neuroimaging software and facilitates interaction between these packages within a single workflow

- [nipy](https://github.com/nipy/nipy)
>- Languages: `Python`, `C`
>- License: BSD
>- Tags: analysis, brain
>- A platform-independent python environment for the analysis of functional brain imaging data using an open development model

- [nitime](https://github.com/nipy/nitime)
>- Languages: `Python`
>- License: BSD
>- Tags: analysis, brain
>- A contains a core of numerical algorithms for time-series analysis both in the time and spectral domains, a set of container objects to represent time-series, and auxiliary objects that expose a high level interface to the numerical machinery and make common analysis tasks easy to express with compact and semantically clear code

- [popeye](https://github.com/kdesimone/popeye)
>- Languages: `Python`
>- License: MIT
>- Tags: analysis, brain
>- A python module for estimating population receptive fields from fmri data built on top of scipy

- [nilean](https://github.com/nilearn/nilearn)
>- Languages: `Python`
>- License: BSD
>- Tags: ml, analysis, brain
>- A machine learning for neuroimaging in python

- [pymvpa](https://github.com/PyMVPA/PyMVPA)
>- Languages: `Python`
>- License: MIT
>- Tags: ml, analysis, brain
>- A multivariate pattern analysis in python

### Processing <a name="processing"></a>
- [vespa](https://github.com/vespa-mrs/vespa/)
>- Languages: `Python`
>- License: BSD
>- Tags: simulation, data, processing
>- A integrated, open source, open development platform for magnetic resonance spectroscopy (mrs) research for rf pulse design, spectral simulation and prototyping, creating synthetic mrs data sets and interactive spectral data processing and analysis.

- [scanhub](https://github.com/brain-link/scanhub)
>- Languages: `Python`
>- License: GPLv3
>- Tags: simulation, reconstruction, processing
>- A multi modal acquisition software, which allows individualizable, modular and cloud-based processing of functional and anatomical medical images.

- [spinal-chord-toolbox](https://github.com/spinalcordtoolbox/spinalcordtoolbox)
>- Languages: `Python`
>- License: LGPLv3
>- Tags: segmentation, processing, spinal
>- A comprehensive, free and open-source set of command-line tools dedicated to the processing and analysis of spinal cord mri data

- [madym](https://gitlab.com/manchester_qbi/manchester_qbi_public/madym_cxx)
>- Languages: `C++`
>- License: Apache
>- Tags: analysis, processing
>- A c++ toolkit for quantative dce-mri and dwi-mri analysis

- [decaes](https://github.com/jondeuce/DECAES.jl)
>- Languages: `Julia`
>- License: MIT
>- Tags: processing
>- A julia implementation of the matlab toolbox from the ubc mri research centre for computing voxelwise t2-distributions from multi spin-echo mri images using the extended phase graph algorithm with stimulated echo corrections

- [mrtrix3](https://github.com/MRtrix3/mrtrix3)
>- Languages: `C++`, `Python`
>- License: MPL
>- Tags: processing
>- A set of tools to perform various types of diffusion mri analyses, from various forms of tractography through to next-generation group-level analyses

- [smriprep](https://github.com/nipreps/smriprep)
>- Languages: `Python`
>- License: Apache
>- Tags: processing
>- A structural magnetic resonance imaging (smri) data preprocessing pipeline that is designed to provide an easily accessible, state-of-the-art interface that is robust to variations in scan acquisition protocols and that requires minimal user input, while providing easily interpretable and comprehensive error and output reporting

- [flow4d](https://github.com/saitta-s/flow4D)
>- Languages: `Python`
>- License: MIT
>- Tags: processing
>- A work with 4d flow mri acquisitions for cfd applications

- [mri-research-tools](https://github.com/korbinian90/MriResearchTools.jl)
>- Languages: `Julia`
>- License: MIT
>- Tags: processing, analysis, simulation
>- A specialized tools for mri

- [pydeface](https://github.com/poldracklab/pydeface)
>- Languages: `Python`
>- License: MIT
>- Tags: processing
>- A a tool to remove facial structure from mri images.

- [mritopng](https://github.com/danishm/mritopng)
>- Languages: `Python`
>- License: MIT
>- Tags: processing
>- A a simple python module to make it easy to batch convert dicom files to png images.

- [qslprep](https://github.com/PennLINC/qsiprep)
>- Languages: `Python`
>- License: BSD
>- Tags: processing, reconstruction, analysis
>- A preprocessing and reconstruction of diffusion mri

- [intensity-normalization](https://github.com/jcreinhold/intensity-normalization)
>- Languages: `Python`
>- License: Apache
>- Tags: processing
>- A various methods to normalize the intensity of various modalities of magnetic resonance (mr) images, e.g., t1-weighted (t1-w), t2-weighted (t2-w), fluid-attenuated inversion recovery (flair), and proton density-weighted (pd-w)

- [monai](https://github.com/Project-MONAI/MONAI)
>- Languages: `Python`
>- License: Apache
>- Tags: ml, processing, segmentation
>- A ai toolkit for healthcare imaging

- [torchio](https://github.com/fepegar/torchio)
>- Languages: `Python`
>- License: Apache
>- Tags: ml, processing
>- A medical imaging toolkit for deep learning

- [medical-torch](https://github.com/perone/medicaltorch?tab=readme-ov-file)
>- Languages: `Python`
>- License: Apache
>- Tags: ml, processing
>- A open-source framework for pytorch, implementing an extensive set of loaders, pre-processors and datasets for medical imaging

- [clinicaldl](https://github.com/aramis-lab/clinicadl)
>- Languages: `Python`
>- License: MIT
>- Tags: ml, processing
>- A framework for the reproducible processing of neuroimaging data with deep learning methods

- [brats-toolkit](https://github.com/neuronflow/BraTS-Toolkit)
>- Languages: `Python`
>- License: AGPLv3
>- Tags: ml, segmentation, processing
>- A code to preprocess, segment, and fuse glioma mri scans based on the brats toolkit manuscript

### Reconstruction <a name="reconstruction"></a>
- [scanhub](https://github.com/brain-link/scanhub)
>- Languages: `Python`
>- License: GPLv3
>- Tags: simulation, reconstruction, processing
>- A multi modal acquisition software, which allows individualizable, modular and cloud-based processing of functional and anatomical medical images.

- [eptlib](https://github.com/eptlib/eptlib)
>- Languages: `C++`, `Python`
>- License: MIT
>- Tags: reconstruction
>- A collection of c++ implementations of electric properties tomography (ept) methods.

- [sirf](https://github.com/SyneRBI/SIRF?tab=readme-ov-file)
>- Languages: `C++`, `Python`
>- License: GPLv2
>- Tags: reconstruction
>- A open source toolkit for the reconstruction of pet and mri raw data.

- [hdr-mri](https://github.com/shakes76/sHDR)
>- Languages: `C++`
>- License: Apache
>- Tags: reconstruction
>- A takes as input coregistered mr images (preferrably of different contrasts), non-linearly combines them and outputs a single hdr mr image.

- [gadgetron](https://github.com/gadgetron/gadgetron)
>- Languages: `C++`
>- License: MIT
>- Tags: reconstruction
>- A open source project for medical image reconstruction

- [powergrid](https://github.com/mrfil/PowerGrid)
>- Languages: `C++`
>- License: MIT
>- Tags: reconstruction
>- A cpu and gpu accelerated iterative magnetic resonance imaging reconstruction

- [tensorflow-mri](https://github.com/mrphys/tensorflow-mri)
>- Languages: `Python`
>- License: Apache
>- Tags: reconstruction, ml
>- A library of tensorflow operators for computational mri

- [mri-reco](https://github.com/MagneticResonanceImaging/MRIReco.jl)
>- Languages: `Julia`
>- License: MIT
>- Tags: reconstruction
>- A julia package for magnetic resonance imaging

- [nlsam](https://github.com/samuelstjean/nlsam)
>- Languages: `Python`
>- License: GPLv3
>- Tags: reconstruction
>- A implementation for the non local spatial and angular matching (nlsam) denoising algorithm for diffusion mri

- [bart](http://mrirecon.github.io/bart/)
>- Languages: `C`, `C++`
>- License: BSD
>- Tags: reconstruction
>- A free and open-source image-reconstruction framework for computational magnetic resonance imaging

- [nesvor](https://github.com/daviddmc/NeSVoR)
>- Languages: `Python`
>- License: MIT
>- Tags: reconstruction, fetal
>- A gpu-accelerated slice-to-volume reconstruction (both rigid and deformable)

- [niftymic](https://github.com/gift-surg/NiftyMIC)
>- Languages: `Python`
>- License: BSD
>- Tags: reconstruction, fetal
>- A toolkit for research developed within the gift-surg project to reconstruct an isotropic, high-resolution volume from multiple, possibly motion-corrupted, stacks of low-resolution 2d slices

- [svrtk](https://github.com/SVRTK/SVRTK)
>- Languages: `C++`
>- License: Apache
>- Tags: reconstruction, fetal
>- A mirtk based svr reconstruction

- [mialsuperresolutiontoolkit](https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit)
>- Languages: `C++`, `Python`
>- License: BSD
>- Tags: reconstruction, fetal
>- A c++ and python tools necessary to perform motion-robust super-resolution fetal mri reconstruction

- [qslprep](https://github.com/PennLINC/qsiprep)
>- Languages: `Python`
>- License: BSD
>- Tags: processing, reconstruction, analysis
>- A preprocessing and reconstruction of diffusion mri

- [direct](https://github.com/NKI-AI/direct)
>- Languages: `Python`
>- License: Apache
>- Tags: reconstruction
>- A deep learning framework for mri reconstruction

- [synthseg](https://github.com/BBillot/SynthSeg)
>- Languages: `Python`
>- License: Apache
>- Tags: ml, reconstruction
>- A deep learning tool for segmentation of brain scans of any contrast and resolution

### Ml <a name="ml"></a>
- [tensorflow-mri](https://github.com/mrphys/tensorflow-mri)
>- Languages: `Python`
>- License: Apache
>- Tags: reconstruction, ml
>- A library of tensorflow operators for computational mri

- [fastmri](https://github.com/facebookresearch/fastMRI)
>- Languages: `Python`
>- License: MIT
>- Tags: ml, analysis, data
>- A research project from facebook ai research (fair) and nyu langone health to investigate the use of ai to make mri scans faster

- [synthseg](https://github.com/BBillot/SynthSeg)
>- Languages: `Python`
>- License: Apache
>- Tags: ml, reconstruction
>- A deep learning tool for segmentation of brain scans of any contrast and resolution

- [monai](https://github.com/Project-MONAI/MONAI)
>- Languages: `Python`
>- License: Apache
>- Tags: ml, processing, segmentation
>- A ai toolkit for healthcare imaging

- [medical-detection-toolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit)
>- Languages: `Python`
>- License: Apache
>- Tags: ml, segmentation
>- A contains 2d + 3d implementations of prevalent object detectors such as mask r-cnn, retina net, retina u-net, as well as a training and inference framework focused on dealing with medical images

- [torchio](https://github.com/fepegar/torchio)
>- Languages: `Python`
>- License: Apache
>- Tags: ml, processing
>- A medical imaging toolkit for deep learning

- [deepmedic](https://github.com/deepmedic/deepmedic)
>- Languages: `Python`
>- License: BSD
>- Tags: ml, segmentation
>- A efficient multi-scale 3d convolutional neural network for segmentation of 3d medical scans

- [medical-torch](https://github.com/perone/medicaltorch?tab=readme-ov-file)
>- Languages: `Python`
>- License: Apache
>- Tags: ml, processing
>- A open-source framework for pytorch, implementing an extensive set of loaders, pre-processors and datasets for medical imaging

- [medical-zoo](https://github.com/black0017/MedicalZooPytorch)
>- Languages: `Python`
>- License: MIT
>- Tags: ml, segmentation
>- A pytorch-based deep learning framework for multi-modal 2d/3d medical image segmentation

- [nilean](https://github.com/nilearn/nilearn)
>- Languages: `Python`
>- License: BSD
>- Tags: ml, analysis, brain
>- A machine learning for neuroimaging in python

- [pymvpa](https://github.com/PyMVPA/PyMVPA)
>- Languages: `Python`
>- License: MIT
>- Tags: ml, analysis, brain
>- A multivariate pattern analysis in python

- [tractseg](https://github.com/MIC-DKFZ/TractSeg)
>- Languages: `Python`
>- License: Apache
>- Tags: ml, segmentation, brain
>- A automatic white matter bundle segmentation

- [clinicaldl](https://github.com/aramis-lab/clinicadl)
>- Languages: `Python`
>- License: MIT
>- Tags: ml, processing
>- A framework for the reproducible processing of neuroimaging data with deep learning methods

- [brats-toolkit](https://github.com/neuronflow/BraTS-Toolkit)
>- Languages: `Python`
>- License: AGPLv3
>- Tags: ml, segmentation, processing
>- A code to preprocess, segment, and fuse glioma mri scans based on the brats toolkit manuscript

### Simulation <a name="simulation"></a>
- [pycoilgen](https://github.com/kev-m/pyCoilGen)
>- Languages: `Python`
>- License: GPLv3
>- Tags: simulation
>- A open source tool for generating coil winding layouts, such as gradient field coils, within the mri and nmr environments.

- [virtual-mri-scanner](https://github.com/imr-framework/virtual-scanner)
>- Languages: `Python`
>- License: AGPLv3
>- Tags: simulation
>- A end-to-end hybrid magnetic resonance imaging (mri) simulator/console designed to be zero-footprint, modular, and supported by open-source standards.

- [cosimpy](https://github.com/umbertozanovello/CoSimPy)
>- Languages: `Python`
>- License: MIT
>- Tags: simulation
>- A open source python library aiming to combine results from electromagnetic (em) simulation with circuits analysis through a cosimulation environment.

- [vespa](https://github.com/vespa-mrs/vespa/)
>- Languages: `Python`
>- License: BSD
>- Tags: simulation, data, processing
>- A integrated, open source, open development platform for magnetic resonance spectroscopy (mrs) research for rf pulse design, spectral simulation and prototyping, creating synthetic mrs data sets and interactive spectral data processing and analysis.

- [scanhub](https://github.com/brain-link/scanhub)
>- Languages: `Python`
>- License: GPLv3
>- Tags: simulation, reconstruction, processing
>- A multi modal acquisition software, which allows individualizable, modular and cloud-based processing of functional and anatomical medical images.

- [slicer](https://github.com/Slicer/Slicer)
>- Languages: `Python`, `C++`
>- License: BSD
>- Tags: simulation, analysis, visualisation
>- A open source software package for visualization and image analysis.

- [pypulseq](https://github.com/imr-framework/pypulseq/)
>- Languages: `Python`
>- License: AGPLv3
>- Tags: simulation
>- A enables vendor-neutral pulse sequence design in python [1,2]. the pulse sequences can be exported as a .seq file to be run on siemens/ge/bruker hardware by leveraging their respective pulseq interpreters.

- [koma](https://github.com/JuliaHealth/KomaMRI.jl)
>- Languages: `Julia`
>- License: MIT
>- Tags: simulation
>- A pulseq-compatible framework to efficiently simulate magnetic resonance imaging (mri) acquisitions

- [gropt](https://github.com/mloecher/gropt)
>- Languages: `C`, `Python`
>- License: GPLv3
>- Tags: simulation
>- A  toolbox for mri gradient optimization

- [disimpy](https://github.com/kerkelae/disimpy)
>- Languages: `Python`
>- License: MIT
>- Tags: simulation
>- A python package for generating simulated diffusion-weighted mr signals that can be useful in the development and validation of data acquisition and analysis methods

- [mri-generalized-bloch](https://github.com/JakobAsslaender/MRIgeneralizedBloch.jl)
>- Languages: `Julia`
>- License: MIT
>- Tags: simulation
>- A julia package that implements the generalized bloch equations for modeling the dynamics of the semi-solid spin pool in magnetic resonance imaging (mri), and its exchange with the free spin pool

- [mri-research-tools](https://github.com/korbinian90/MriResearchTools.jl)
>- Languages: `Julia`
>- License: MIT
>- Tags: processing, analysis, simulation
>- A specialized tools for mri

- [mrisafety](https://github.com/felixhorger/MRISafety.jl)
>- Languages: `Julia`
>- License: None
>- Tags: safety, simulation
>- A mri safety checks

### Segmentation <a name="segmentation"></a>
- [dl-direct](https://github.com/SCAN-NRAD/DL-DiReCT)
>- Languages: `Python`
>- License: BSD
>- Tags: segmentation
>- A combines a deep learning-based neuroanatomy segmentation and cortex parcellation with a diffeomorphic registration technique to measure cortical thickness from t1w mri

- [dafne](https://github.com/dafne-imaging/dafne)
>- Languages: `Python`
>- License: GPLv3
>- Tags: segmentation
>- A program for the segmentation of medical images. it relies on a server to provide deep learning models to aid the segmentation, and incremental learning is used to improve the performance

- [spinal-chord-toolbox](https://github.com/spinalcordtoolbox/spinalcordtoolbox)
>- Languages: `Python`
>- License: LGPLv3
>- Tags: segmentation, processing, spinal
>- A comprehensive, free and open-source set of command-line tools dedicated to the processing and analysis of spinal cord mri data

- [dwybss](https://github.com/mmromero/dwybss)
>- Languages: `Python`
>- License: LGPLv3
>- Tags: segmentation, brain
>- A separate microstructure tissue components from the diffusion mri signal, characterize the volume fractions, and t2 maps of these compartments

- [simple-itk](https://github.com/SimpleITK/SimpleITK)
>- Languages: `C++`, `Python`, `R`
>- License: Apache
>- Tags: segmentation, analysis
>- A image analysis toolkit with a large number of components supporting general filtering operations, image segmentation and registration

- [quicknat](https://github.com/ai-med/quickNAT_pytorch)
>- Languages: `Python`
>- License: MIT
>- Tags: segmentation, brain
>- A fully convolutional network for quick and accurate segmentation of neuroanatomy and quality control of structure-wise segmentations

- [brainchop](https://github.com/neuroneural/brainchop)
>- Languages: `Javascript`, `Python`
>- License: MIT
>- Tags: segmentation, visualisation
>- A in-browser 3d mri rendering and segmentation

- [monai](https://github.com/Project-MONAI/MONAI)
>- Languages: `Python`
>- License: Apache
>- Tags: ml, processing, segmentation
>- A ai toolkit for healthcare imaging

- [medical-detection-toolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit)
>- Languages: `Python`
>- License: Apache
>- Tags: ml, segmentation
>- A contains 2d + 3d implementations of prevalent object detectors such as mask r-cnn, retina net, retina u-net, as well as a training and inference framework focused on dealing with medical images

- [deepmedic](https://github.com/deepmedic/deepmedic)
>- Languages: `Python`
>- License: BSD
>- Tags: ml, segmentation
>- A efficient multi-scale 3d convolutional neural network for segmentation of 3d medical scans

- [medical-zoo](https://github.com/black0017/MedicalZooPytorch)
>- Languages: `Python`
>- License: MIT
>- Tags: ml, segmentation
>- A pytorch-based deep learning framework for multi-modal 2d/3d medical image segmentation

- [tractseg](https://github.com/MIC-DKFZ/TractSeg)
>- Languages: `Python`
>- License: Apache
>- Tags: ml, segmentation, brain
>- A automatic white matter bundle segmentation

- [brats-toolkit](https://github.com/neuronflow/BraTS-Toolkit)
>- Languages: `Python`
>- License: AGPLv3
>- Tags: ml, segmentation, processing
>- A code to preprocess, segment, and fuse glioma mri scans based on the brats toolkit manuscript

### Brain <a name="brain"></a>
- [ukf-tractography](https://github.com/pnlbwh/ukftractography)
>- Languages: `C`, `C++`
>- License: BSD
>- Tags: visualisation, brain
>- A framework which uses an unscented kalman filter for performing tractography

- [dwybss](https://github.com/mmromero/dwybss)
>- Languages: `Python`
>- License: LGPLv3
>- Tags: segmentation, brain
>- A separate microstructure tissue components from the diffusion mri signal, characterize the volume fractions, and t2 maps of these compartments

- [freesurfer](https://github.com/freesurfer/freesurfer)
>- Languages: `C++`, `C`, `Python`
>- License: GPLv3
>- Tags: analysis, visualisation, brain
>- A analysis and visualization of neuroimaging data from cross-sectional and longitudinal studies

- [quicknat](https://github.com/ai-med/quickNAT_pytorch)
>- Languages: `Python`
>- License: MIT
>- Tags: segmentation, brain
>- A fully convolutional network for quick and accurate segmentation of neuroanatomy and quality control of structure-wise segmentations

- [nipype](https://github.com/nipy/nipype)
>- Languages: `Python`
>- License: Apache
>- Tags: analysis, brain
>- A python project that provides a uniform interface to existing neuroimaging software and facilitates interaction between these packages within a single workflow

- [nipy](https://github.com/nipy/nipy)
>- Languages: `Python`, `C`
>- License: BSD
>- Tags: analysis, brain
>- A platform-independent python environment for the analysis of functional brain imaging data using an open development model

- [nitime](https://github.com/nipy/nitime)
>- Languages: `Python`
>- License: BSD
>- Tags: analysis, brain
>- A contains a core of numerical algorithms for time-series analysis both in the time and spectral domains, a set of container objects to represent time-series, and auxiliary objects that expose a high level interface to the numerical machinery and make common analysis tasks easy to express with compact and semantically clear code

- [popeye](https://github.com/kdesimone/popeye)
>- Languages: `Python`
>- License: MIT
>- Tags: analysis, brain
>- A python module for estimating population receptive fields from fmri data built on top of scipy

- [nilean](https://github.com/nilearn/nilearn)
>- Languages: `Python`
>- License: BSD
>- Tags: ml, analysis, brain
>- A machine learning for neuroimaging in python

- [pymvpa](https://github.com/PyMVPA/PyMVPA)
>- Languages: `Python`
>- License: MIT
>- Tags: ml, analysis, brain
>- A multivariate pattern analysis in python

- [tractseg](https://github.com/MIC-DKFZ/TractSeg)
>- Languages: `Python`
>- License: Apache
>- Tags: ml, segmentation, brain
>- A automatic white matter bundle segmentation

### Data <a name="data"></a>
- [vespa](https://github.com/vespa-mrs/vespa/)
>- Languages: `Python`
>- License: BSD
>- Tags: simulation, data, processing
>- A integrated, open source, open development platform for magnetic resonance spectroscopy (mrs) research for rf pulse design, spectral simulation and prototyping, creating synthetic mrs data sets and interactive spectral data processing and analysis.

- [ismrm-raw-data-format](https://ismrmrd.github.io/apidocs/1.5.0/)
>- Languages: `C`, `C++`, `Python`
>- License: MIT
>- Tags: data
>- A  common raw data format, which attempts to capture the data fields that are required to describe the magnetic resonance experiment with enough detail to reconstruct images

- [fastmri](https://github.com/facebookresearch/fastMRI)
>- Languages: `Python`
>- License: MIT
>- Tags: ml, analysis, data
>- A research project from facebook ai research (fair) and nyu langone health to investigate the use of ai to make mri scans faster

- [nlft](https://github.com/JuliaNeuroscience/NIfTI.jl)
>- Languages: `Julia`
>- License: MIT
>- Tags: data
>- A julia module for reading/writing nifti mri files

- [openmorph](https://github.com/cMadan/openMorph)
>- Languages: `Jupyter`
>- License: None
>- Tags: data
>- A curated list of open-access databases with human structural mri data

- [nibabel](https://github.com/nipy/nibabel)
>- Languages: `Python`
>- License: MIT
>- Tags: data
>- A read and write access to common neuroimaging file formats, including: analyze (plain, spm99, spm2 and later), gifti, nifti1, nifti2, cifti-2, minc1, minc2, afni brik/head, ecat and philips par/rec. in addition, nibabel also supports freesurfer's mgh, geometry, annotation and morphometry files, and provides some limited support for dicom

- [mrdqed](https://github.com/EGates1/MRDQED)
>- Languages: `Python`
>- License: None
>- Tags: qa, data
>- A magnetic resonance data quality evaluation dashboard

### Visualisation <a name="visualisation"></a>
- [slicer](https://github.com/Slicer/Slicer)
>- Languages: `Python`, `C++`
>- License: BSD
>- Tags: simulation, analysis, visualisation
>- A open source software package for visualization and image analysis.

- [ukf-tractography](https://github.com/pnlbwh/ukftractography)
>- Languages: `C`, `C++`
>- License: BSD
>- Tags: visualisation, brain
>- A framework which uses an unscented kalman filter for performing tractography

- [freesurfer](https://github.com/freesurfer/freesurfer)
>- Languages: `C++`, `C`, `Python`
>- License: GPLv3
>- Tags: analysis, visualisation, brain
>- A analysis and visualization of neuroimaging data from cross-sectional and longitudinal studies

- [gif_your_nifti](https://github.com/miykael/gif_your_nifti)
>- Languages: `Python`
>- License: BSD
>- Tags: visualisation
>- A create nice looking gifs from your nifti (.nii or .nii.gz) files with a simple command

- [brainchop](https://github.com/neuroneural/brainchop)
>- Languages: `Javascript`, `Python`
>- License: MIT
>- Tags: segmentation, visualisation
>- A in-browser 3d mri rendering and segmentation

- [mri-viewer](https://github.com/epam/mriviewer)
>- Languages: `Javascript`
>- License: Apache
>- Tags: visualisation
>- A high performance web tool for advanced visualization (both in 2d and 3d modes) medical volumetric data, provided in popular file formats: dicom, nifti, ktx, hdr

### Qa <a name="qa"></a>
- [mrqy](https://github.com/ccipd/MRQy)
>- Languages: `Javascript`, `Python`
>- License: BSD
>- Tags: qa, analysis
>- A generate several tags and noise/information measurements for quality assessment

- [fetal-iqa](https://github.com/daviddmc/fetal-IQA)
>- Languages: `Python`
>- License: MIT
>- Tags: qa, fetal
>- A  image quality assessment (iqa) method for fetal mri

- [mriqc](https://github.com/nipreps/mriqc)
>- Languages: `Javascript`, `Python`
>- License: Apache
>- Tags: qa, analysis
>- A extracts no-reference iqms (image quality metrics) from structural (t1w and t2w) and functional mri (magnetic resonance imaging) data

- [mrqa](https://github.com/Open-Minds-Lab/mrQA)
>- Languages: `Python`
>- License: Apache
>- Tags: qa
>- A mrqa: tools for quality assurance in medical imaging datasets, including protocol compliance

- [hazen](https://github.com/GSTT-CSC/hazen)
>- Languages: `Python`
>- License: Apache
>- Tags: qa
>- A quality assurance framework for magnetic resonance imaging

- [mrdqed](https://github.com/EGates1/MRDQED)
>- Languages: `Python`
>- License: None
>- Tags: qa, data
>- A magnetic resonance data quality evaluation dashboard

### Fetal <a name="fetal"></a>
- [nesvor](https://github.com/daviddmc/NeSVoR)
>- Languages: `Python`
>- License: MIT
>- Tags: reconstruction, fetal
>- A gpu-accelerated slice-to-volume reconstruction (both rigid and deformable)

- [affirm](https://github.com/allard-shi/affirm)
>- Languages: `Python`
>- License: MIT
>- Tags: analysis, fetal
>- A deep recursive fetal motion estimation and correction based on slice and volume affinity fusion

- [niftymic](https://github.com/gift-surg/NiftyMIC)
>- Languages: `Python`
>- License: BSD
>- Tags: reconstruction, fetal
>- A toolkit for research developed within the gift-surg project to reconstruct an isotropic, high-resolution volume from multiple, possibly motion-corrupted, stacks of low-resolution 2d slices

- [svrtk](https://github.com/SVRTK/SVRTK)
>- Languages: `C++`
>- License: Apache
>- Tags: reconstruction, fetal
>- A mirtk based svr reconstruction

- [mialsuperresolutiontoolkit](https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit)
>- Languages: `C++`, `Python`
>- License: BSD
>- Tags: reconstruction, fetal
>- A c++ and python tools necessary to perform motion-robust super-resolution fetal mri reconstruction

- [fetal-iqa](https://github.com/daviddmc/fetal-IQA)
>- Languages: `Python`
>- License: MIT
>- Tags: qa, fetal
>- A  image quality assessment (iqa) method for fetal mri

### Renal <a name="renal"></a>
- [ukat](https://github.com/UKRIN-MAPS/ukat)
>- Languages: `Python`
>- License: GPLv3
>- Tags: analysis, renal
>- A ukat is a vendor agnostic framework for the analysis of quantitative renal mri data

### Spinal <a name="spinal"></a>
- [spinal-chord-toolbox](https://github.com/spinalcordtoolbox/spinalcordtoolbox)
>- Languages: `Python`
>- License: LGPLv3
>- Tags: segmentation, processing, spinal
>- A comprehensive, free and open-source set of command-line tools dedicated to the processing and analysis of spinal cord mri data

### Muscle <a name="muscle"></a>
- [myoqmri](https://github.com/fsantini/MyoQMRI)
>- Languages: `Python`
>- License: GPLv3
>- Tags: analysis, muscle
>- A quantitative mri of the muscles

### Safety <a name="safety"></a>
- [mrisafety](https://github.com/felixhorger/MRISafety.jl)
>- Languages: `Julia`
>- License: None
>- Tags: safety, simulation
>- A mri safety checks



## Languages
### Python <a name="python"></a>
- [pycoilgen](https://github.com/kev-m/pyCoilGen)
>- Languages: `Python`
>- License: GPLv3
>- Tags: simulation
>- A open source tool for generating coil winding layouts, such as gradient field coils, within the mri and nmr environments.

- [virtual-mri-scanner](https://github.com/imr-framework/virtual-scanner)
>- Languages: `Python`
>- License: AGPLv3
>- Tags: simulation
>- A end-to-end hybrid magnetic resonance imaging (mri) simulator/console designed to be zero-footprint, modular, and supported by open-source standards.

- [cosimpy](https://github.com/umbertozanovello/CoSimPy)
>- Languages: `Python`
>- License: MIT
>- Tags: simulation
>- A open source python library aiming to combine results from electromagnetic (em) simulation with circuits analysis through a cosimulation environment.

- [vespa](https://github.com/vespa-mrs/vespa/)
>- Languages: `Python`
>- License: BSD
>- Tags: simulation, data, processing
>- A integrated, open source, open development platform for magnetic resonance spectroscopy (mrs) research for rf pulse design, spectral simulation and prototyping, creating synthetic mrs data sets and interactive spectral data processing and analysis.

- [scanhub](https://github.com/brain-link/scanhub)
>- Languages: `Python`
>- License: GPLv3
>- Tags: simulation, reconstruction, processing
>- A multi modal acquisition software, which allows individualizable, modular and cloud-based processing of functional and anatomical medical images.

- [slicer](https://github.com/Slicer/Slicer)
>- Languages: `Python`, `C++`
>- License: BSD
>- Tags: simulation, analysis, visualisation
>- A open source software package for visualization and image analysis.

- [eptlib](https://github.com/eptlib/eptlib)
>- Languages: `C++`, `Python`
>- License: MIT
>- Tags: reconstruction
>- A collection of c++ implementations of electric properties tomography (ept) methods.

- [sirf](https://github.com/SyneRBI/SIRF?tab=readme-ov-file)
>- Languages: `C++`, `Python`
>- License: GPLv2
>- Tags: reconstruction
>- A open source toolkit for the reconstruction of pet and mri raw data.

- [dl-direct](https://github.com/SCAN-NRAD/DL-DiReCT)
>- Languages: `Python`
>- License: BSD
>- Tags: segmentation
>- A combines a deep learning-based neuroanatomy segmentation and cortex parcellation with a diffeomorphic registration technique to measure cortical thickness from t1w mri

- [dafne](https://github.com/dafne-imaging/dafne)
>- Languages: `Python`
>- License: GPLv3
>- Tags: segmentation
>- A program for the segmentation of medical images. it relies on a server to provide deep learning models to aid the segmentation, and incremental learning is used to improve the performance

- [ukat](https://github.com/UKRIN-MAPS/ukat)
>- Languages: `Python`
>- License: GPLv3
>- Tags: analysis, renal
>- A ukat is a vendor agnostic framework for the analysis of quantitative renal mri data

- [pypulseq](https://github.com/imr-framework/pypulseq/)
>- Languages: `Python`
>- License: AGPLv3
>- Tags: simulation
>- A enables vendor-neutral pulse sequence design in python [1,2]. the pulse sequences can be exported as a .seq file to be run on siemens/ge/bruker hardware by leveraging their respective pulseq interpreters.

- [spinal-chord-toolbox](https://github.com/spinalcordtoolbox/spinalcordtoolbox)
>- Languages: `Python`
>- License: LGPLv3
>- Tags: segmentation, processing, spinal
>- A comprehensive, free and open-source set of command-line tools dedicated to the processing and analysis of spinal cord mri data

- [dwybss](https://github.com/mmromero/dwybss)
>- Languages: `Python`
>- License: LGPLv3
>- Tags: segmentation, brain
>- A separate microstructure tissue components from the diffusion mri signal, characterize the volume fractions, and t2 maps of these compartments

- [freesurfer](https://github.com/freesurfer/freesurfer)
>- Languages: `C++`, `C`, `Python`
>- License: GPLv3
>- Tags: analysis, visualisation, brain
>- A analysis and visualization of neuroimaging data from cross-sectional and longitudinal studies

- [simple-itk](https://github.com/SimpleITK/SimpleITK)
>- Languages: `C++`, `Python`, `R`
>- License: Apache
>- Tags: segmentation, analysis
>- A image analysis toolkit with a large number of components supporting general filtering operations, image segmentation and registration

- [ismrm-raw-data-format](https://ismrmrd.github.io/apidocs/1.5.0/)
>- Languages: `C`, `C++`, `Python`
>- License: MIT
>- Tags: data
>- A  common raw data format, which attempts to capture the data fields that are required to describe the magnetic resonance experiment with enough detail to reconstruct images

- [tensorflow-mri](https://github.com/mrphys/tensorflow-mri)
>- Languages: `Python`
>- License: Apache
>- Tags: reconstruction, ml
>- A library of tensorflow operators for computational mri

- [quit](https://github.com/spinicist/QUIT)
>- Languages: `C++`, `Python`
>- License: MPL
>- Tags: analysis
>- A collection of programs for processing quantitative mri data

- [gropt](https://github.com/mloecher/gropt)
>- Languages: `C`, `Python`
>- License: GPLv3
>- Tags: simulation
>- A  toolbox for mri gradient optimization

- [disimpy](https://github.com/kerkelae/disimpy)
>- Languages: `Python`
>- License: MIT
>- Tags: simulation
>- A python package for generating simulated diffusion-weighted mr signals that can be useful in the development and validation of data acquisition and analysis methods

- [myoqmri](https://github.com/fsantini/MyoQMRI)
>- Languages: `Python`
>- License: GPLv3
>- Tags: analysis, muscle
>- A quantitative mri of the muscles

- [mrqy](https://github.com/ccipd/MRQy)
>- Languages: `Javascript`, `Python`
>- License: BSD
>- Tags: qa, analysis
>- A generate several tags and noise/information measurements for quality assessment

- [nlsam](https://github.com/samuelstjean/nlsam)
>- Languages: `Python`
>- License: GPLv3
>- Tags: reconstruction
>- A implementation for the non local spatial and angular matching (nlsam) denoising algorithm for diffusion mri

- [mrtrix3](https://github.com/MRtrix3/mrtrix3)
>- Languages: `C++`, `Python`
>- License: MPL
>- Tags: processing
>- A set of tools to perform various types of diffusion mri analyses, from various forms of tractography through to next-generation group-level analyses

- [smriprep](https://github.com/nipreps/smriprep)
>- Languages: `Python`
>- License: Apache
>- Tags: processing
>- A structural magnetic resonance imaging (smri) data preprocessing pipeline that is designed to provide an easily accessible, state-of-the-art interface that is robust to variations in scan acquisition protocols and that requires minimal user input, while providing easily interpretable and comprehensive error and output reporting

- [fastmri](https://github.com/facebookresearch/fastMRI)
>- Languages: `Python`
>- License: MIT
>- Tags: ml, analysis, data
>- A research project from facebook ai research (fair) and nyu langone health to investigate the use of ai to make mri scans faster

- [flow4d](https://github.com/saitta-s/flow4D)
>- Languages: `Python`
>- License: MIT
>- Tags: processing
>- A work with 4d flow mri acquisitions for cfd applications

- [nesvor](https://github.com/daviddmc/NeSVoR)
>- Languages: `Python`
>- License: MIT
>- Tags: reconstruction, fetal
>- A gpu-accelerated slice-to-volume reconstruction (both rigid and deformable)

- [affirm](https://github.com/allard-shi/affirm)
>- Languages: `Python`
>- License: MIT
>- Tags: analysis, fetal
>- A deep recursive fetal motion estimation and correction based on slice and volume affinity fusion

- [niftymic](https://github.com/gift-surg/NiftyMIC)
>- Languages: `Python`
>- License: BSD
>- Tags: reconstruction, fetal
>- A toolkit for research developed within the gift-surg project to reconstruct an isotropic, high-resolution volume from multiple, possibly motion-corrupted, stacks of low-resolution 2d slices

- [mialsuperresolutiontoolkit](https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit)
>- Languages: `C++`, `Python`
>- License: BSD
>- Tags: reconstruction, fetal
>- A c++ and python tools necessary to perform motion-robust super-resolution fetal mri reconstruction

- [fetal-iqa](https://github.com/daviddmc/fetal-IQA)
>- Languages: `Python`
>- License: MIT
>- Tags: qa, fetal
>- A  image quality assessment (iqa) method for fetal mri

- [quicknat](https://github.com/ai-med/quickNAT_pytorch)
>- Languages: `Python`
>- License: MIT
>- Tags: segmentation, brain
>- A fully convolutional network for quick and accurate segmentation of neuroanatomy and quality control of structure-wise segmentations

- [pydeface](https://github.com/poldracklab/pydeface)
>- Languages: `Python`
>- License: MIT
>- Tags: processing
>- A a tool to remove facial structure from mri images.

- [mritopng](https://github.com/danishm/mritopng)
>- Languages: `Python`
>- License: MIT
>- Tags: processing
>- A a simple python module to make it easy to batch convert dicom files to png images.

- [qslprep](https://github.com/PennLINC/qsiprep)
>- Languages: `Python`
>- License: BSD
>- Tags: processing, reconstruction, analysis
>- A preprocessing and reconstruction of diffusion mri

- [gif_your_nifti](https://github.com/miykael/gif_your_nifti)
>- Languages: `Python`
>- License: BSD
>- Tags: visualisation
>- A create nice looking gifs from your nifti (.nii or .nii.gz) files with a simple command

- [direct](https://github.com/NKI-AI/direct)
>- Languages: `Python`
>- License: Apache
>- Tags: reconstruction
>- A deep learning framework for mri reconstruction

- [brainchop](https://github.com/neuroneural/brainchop)
>- Languages: `Javascript`, `Python`
>- License: MIT
>- Tags: segmentation, visualisation
>- A in-browser 3d mri rendering and segmentation

- [mriqc](https://github.com/nipreps/mriqc)
>- Languages: `Javascript`, `Python`
>- License: Apache
>- Tags: qa, analysis
>- A extracts no-reference iqms (image quality metrics) from structural (t1w and t2w) and functional mri (magnetic resonance imaging) data

- [synthseg](https://github.com/BBillot/SynthSeg)
>- Languages: `Python`
>- License: Apache
>- Tags: ml, reconstruction
>- A deep learning tool for segmentation of brain scans of any contrast and resolution

- [intensity-normalization](https://github.com/jcreinhold/intensity-normalization)
>- Languages: `Python`
>- License: Apache
>- Tags: processing
>- A various methods to normalize the intensity of various modalities of magnetic resonance (mr) images, e.g., t1-weighted (t1-w), t2-weighted (t2-w), fluid-attenuated inversion recovery (flair), and proton density-weighted (pd-w)

- [monai](https://github.com/Project-MONAI/MONAI)
>- Languages: `Python`
>- License: Apache
>- Tags: ml, processing, segmentation
>- A ai toolkit for healthcare imaging

- [medical-detection-toolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit)
>- Languages: `Python`
>- License: Apache
>- Tags: ml, segmentation
>- A contains 2d + 3d implementations of prevalent object detectors such as mask r-cnn, retina net, retina u-net, as well as a training and inference framework focused on dealing with medical images

- [torchio](https://github.com/fepegar/torchio)
>- Languages: `Python`
>- License: Apache
>- Tags: ml, processing
>- A medical imaging toolkit for deep learning

- [deepmedic](https://github.com/deepmedic/deepmedic)
>- Languages: `Python`
>- License: BSD
>- Tags: ml, segmentation
>- A efficient multi-scale 3d convolutional neural network for segmentation of 3d medical scans

- [medical-torch](https://github.com/perone/medicaltorch?tab=readme-ov-file)
>- Languages: `Python`
>- License: Apache
>- Tags: ml, processing
>- A open-source framework for pytorch, implementing an extensive set of loaders, pre-processors and datasets for medical imaging

- [medical-zoo](https://github.com/black0017/MedicalZooPytorch)
>- Languages: `Python`
>- License: MIT
>- Tags: ml, segmentation
>- A pytorch-based deep learning framework for multi-modal 2d/3d medical image segmentation

- [nipype](https://github.com/nipy/nipype)
>- Languages: `Python`
>- License: Apache
>- Tags: analysis, brain
>- A python project that provides a uniform interface to existing neuroimaging software and facilitates interaction between these packages within a single workflow

- [nibabel](https://github.com/nipy/nibabel)
>- Languages: `Python`
>- License: MIT
>- Tags: data
>- A read and write access to common neuroimaging file formats, including: analyze (plain, spm99, spm2 and later), gifti, nifti1, nifti2, cifti-2, minc1, minc2, afni brik/head, ecat and philips par/rec. in addition, nibabel also supports freesurfer's mgh, geometry, annotation and morphometry files, and provides some limited support for dicom

- [nipy](https://github.com/nipy/nipy)
>- Languages: `Python`, `C`
>- License: BSD
>- Tags: analysis, brain
>- A platform-independent python environment for the analysis of functional brain imaging data using an open development model

- [nitime](https://github.com/nipy/nitime)
>- Languages: `Python`
>- License: BSD
>- Tags: analysis, brain
>- A contains a core of numerical algorithms for time-series analysis both in the time and spectral domains, a set of container objects to represent time-series, and auxiliary objects that expose a high level interface to the numerical machinery and make common analysis tasks easy to express with compact and semantically clear code

- [popeye](https://github.com/kdesimone/popeye)
>- Languages: `Python`
>- License: MIT
>- Tags: analysis, brain
>- A python module for estimating population receptive fields from fmri data built on top of scipy

- [nilean](https://github.com/nilearn/nilearn)
>- Languages: `Python`
>- License: BSD
>- Tags: ml, analysis, brain
>- A machine learning for neuroimaging in python

- [pymvpa](https://github.com/PyMVPA/PyMVPA)
>- Languages: `Python`
>- License: MIT
>- Tags: ml, analysis, brain
>- A multivariate pattern analysis in python

- [tractseg](https://github.com/MIC-DKFZ/TractSeg)
>- Languages: `Python`
>- License: Apache
>- Tags: ml, segmentation, brain
>- A automatic white matter bundle segmentation

- [clinicaldl](https://github.com/aramis-lab/clinicadl)
>- Languages: `Python`
>- License: MIT
>- Tags: ml, processing
>- A framework for the reproducible processing of neuroimaging data with deep learning methods

- [brats-toolkit](https://github.com/neuronflow/BraTS-Toolkit)
>- Languages: `Python`
>- License: AGPLv3
>- Tags: ml, segmentation, processing
>- A code to preprocess, segment, and fuse glioma mri scans based on the brats toolkit manuscript

- [mrqa](https://github.com/Open-Minds-Lab/mrQA)
>- Languages: `Python`
>- License: Apache
>- Tags: qa
>- A mrqa: tools for quality assurance in medical imaging datasets, including protocol compliance

- [hazen](https://github.com/GSTT-CSC/hazen)
>- Languages: `Python`
>- License: Apache
>- Tags: qa
>- A quality assurance framework for magnetic resonance imaging

- [mrdqed](https://github.com/EGates1/MRDQED)
>- Languages: `Python`
>- License: None
>- Tags: qa, data
>- A magnetic resonance data quality evaluation dashboard

### C++ <a name="c++"></a>
- [slicer](https://github.com/Slicer/Slicer)
>- Languages: `Python`, `C++`
>- License: BSD
>- Tags: simulation, analysis, visualisation
>- A open source software package for visualization and image analysis.

- [eptlib](https://github.com/eptlib/eptlib)
>- Languages: `C++`, `Python`
>- License: MIT
>- Tags: reconstruction
>- A collection of c++ implementations of electric properties tomography (ept) methods.

- [sirf](https://github.com/SyneRBI/SIRF?tab=readme-ov-file)
>- Languages: `C++`, `Python`
>- License: GPLv2
>- Tags: reconstruction
>- A open source toolkit for the reconstruction of pet and mri raw data.

- [hdr-mri](https://github.com/shakes76/sHDR)
>- Languages: `C++`
>- License: Apache
>- Tags: reconstruction
>- A takes as input coregistered mr images (preferrably of different contrasts), non-linearly combines them and outputs a single hdr mr image.

- [gadgetron](https://github.com/gadgetron/gadgetron)
>- Languages: `C++`
>- License: MIT
>- Tags: reconstruction
>- A open source project for medical image reconstruction

- [powergrid](https://github.com/mrfil/PowerGrid)
>- Languages: `C++`
>- License: MIT
>- Tags: reconstruction
>- A cpu and gpu accelerated iterative magnetic resonance imaging reconstruction

- [ukf-tractography](https://github.com/pnlbwh/ukftractography)
>- Languages: `C`, `C++`
>- License: BSD
>- Tags: visualisation, brain
>- A framework which uses an unscented kalman filter for performing tractography

- [freesurfer](https://github.com/freesurfer/freesurfer)
>- Languages: `C++`, `C`, `Python`
>- License: GPLv3
>- Tags: analysis, visualisation, brain
>- A analysis and visualization of neuroimaging data from cross-sectional and longitudinal studies

- [simple-itk](https://github.com/SimpleITK/SimpleITK)
>- Languages: `C++`, `Python`, `R`
>- License: Apache
>- Tags: segmentation, analysis
>- A image analysis toolkit with a large number of components supporting general filtering operations, image segmentation and registration

- [ismrm-raw-data-format](https://ismrmrd.github.io/apidocs/1.5.0/)
>- Languages: `C`, `C++`, `Python`
>- License: MIT
>- Tags: data
>- A  common raw data format, which attempts to capture the data fields that are required to describe the magnetic resonance experiment with enough detail to reconstruct images

- [quit](https://github.com/spinicist/QUIT)
>- Languages: `C++`, `Python`
>- License: MPL
>- Tags: analysis
>- A collection of programs for processing quantitative mri data

- [madym](https://gitlab.com/manchester_qbi/manchester_qbi_public/madym_cxx)
>- Languages: `C++`
>- License: Apache
>- Tags: analysis, processing
>- A c++ toolkit for quantative dce-mri and dwi-mri analysis

- [bart](http://mrirecon.github.io/bart/)
>- Languages: `C`, `C++`
>- License: BSD
>- Tags: reconstruction
>- A free and open-source image-reconstruction framework for computational magnetic resonance imaging

- [mrtrix3](https://github.com/MRtrix3/mrtrix3)
>- Languages: `C++`, `Python`
>- License: MPL
>- Tags: processing
>- A set of tools to perform various types of diffusion mri analyses, from various forms of tractography through to next-generation group-level analyses

- [svrtk](https://github.com/SVRTK/SVRTK)
>- Languages: `C++`
>- License: Apache
>- Tags: reconstruction, fetal
>- A mirtk based svr reconstruction

- [mialsuperresolutiontoolkit](https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit)
>- Languages: `C++`, `Python`
>- License: BSD
>- Tags: reconstruction, fetal
>- A c++ and python tools necessary to perform motion-robust super-resolution fetal mri reconstruction

### Julia <a name="julia"></a>
- [koma](https://github.com/JuliaHealth/KomaMRI.jl)
>- Languages: `Julia`
>- License: MIT
>- Tags: simulation
>- A pulseq-compatible framework to efficiently simulate magnetic resonance imaging (mri) acquisitions

- [mri-generalized-bloch](https://github.com/JakobAsslaender/MRIgeneralizedBloch.jl)
>- Languages: `Julia`
>- License: MIT
>- Tags: simulation
>- A julia package that implements the generalized bloch equations for modeling the dynamics of the semi-solid spin pool in magnetic resonance imaging (mri), and its exchange with the free spin pool

- [decaes](https://github.com/jondeuce/DECAES.jl)
>- Languages: `Julia`
>- License: MIT
>- Tags: processing
>- A julia implementation of the matlab toolbox from the ubc mri research centre for computing voxelwise t2-distributions from multi spin-echo mri images using the extended phase graph algorithm with stimulated echo corrections

- [mri-reco](https://github.com/MagneticResonanceImaging/MRIReco.jl)
>- Languages: `Julia`
>- License: MIT
>- Tags: reconstruction
>- A julia package for magnetic resonance imaging

- [mri-research-tools](https://github.com/korbinian90/MriResearchTools.jl)
>- Languages: `Julia`
>- License: MIT
>- Tags: processing, analysis, simulation
>- A specialized tools for mri

- [nlft](https://github.com/JuliaNeuroscience/NIfTI.jl)
>- Languages: `Julia`
>- License: MIT
>- Tags: data
>- A julia module for reading/writing nifti mri files

- [dcemri](https://github.com/davidssmith/DCEMRI.jl)
>- Languages: `Julia`
>- License: MIT
>- Tags: analysis
>- A open source toolkit for dynamic contrast enhanced mri analysis

- [mrisafety](https://github.com/felixhorger/MRISafety.jl)
>- Languages: `Julia`
>- License: None
>- Tags: safety, simulation
>- A mri safety checks

### C <a name="c"></a>
- [ukf-tractography](https://github.com/pnlbwh/ukftractography)
>- Languages: `C`, `C++`
>- License: BSD
>- Tags: visualisation, brain
>- A framework which uses an unscented kalman filter for performing tractography

- [freesurfer](https://github.com/freesurfer/freesurfer)
>- Languages: `C++`, `C`, `Python`
>- License: GPLv3
>- Tags: analysis, visualisation, brain
>- A analysis and visualization of neuroimaging data from cross-sectional and longitudinal studies

- [ismrm-raw-data-format](https://ismrmrd.github.io/apidocs/1.5.0/)
>- Languages: `C`, `C++`, `Python`
>- License: MIT
>- Tags: data
>- A  common raw data format, which attempts to capture the data fields that are required to describe the magnetic resonance experiment with enough detail to reconstruct images

- [gropt](https://github.com/mloecher/gropt)
>- Languages: `C`, `Python`
>- License: GPLv3
>- Tags: simulation
>- A  toolbox for mri gradient optimization

- [bart](http://mrirecon.github.io/bart/)
>- Languages: `C`, `C++`
>- License: BSD
>- Tags: reconstruction
>- A free and open-source image-reconstruction framework for computational magnetic resonance imaging

- [nipy](https://github.com/nipy/nipy)
>- Languages: `Python`, `C`
>- License: BSD
>- Tags: analysis, brain
>- A platform-independent python environment for the analysis of functional brain imaging data using an open development model

### Javascript <a name="javascript"></a>
- [mrqy](https://github.com/ccipd/MRQy)
>- Languages: `Javascript`, `Python`
>- License: BSD
>- Tags: qa, analysis
>- A generate several tags and noise/information measurements for quality assessment

- [brainchop](https://github.com/neuroneural/brainchop)
>- Languages: `Javascript`, `Python`
>- License: MIT
>- Tags: segmentation, visualisation
>- A in-browser 3d mri rendering and segmentation

- [mriqc](https://github.com/nipreps/mriqc)
>- Languages: `Javascript`, `Python`
>- License: Apache
>- Tags: qa, analysis
>- A extracts no-reference iqms (image quality metrics) from structural (t1w and t2w) and functional mri (magnetic resonance imaging) data

- [mri-viewer](https://github.com/epam/mriviewer)
>- Languages: `Javascript`
>- License: Apache
>- Tags: visualisation
>- A high performance web tool for advanced visualization (both in 2d and 3d modes) medical volumetric data, provided in popular file formats: dicom, nifti, ktx, hdr

### R <a name="r"></a>
- [braingraph](https://github.com/cwatson/brainGraph)
>- Languages: `R`
>- License: None
>- Tags: analysis
>- A r package for performing graph theory analyses of brain mri data

### Jupyter <a name="jupyter"></a>
- [openmorph](https://github.com/cMadan/openMorph)
>- Languages: `Jupyter`
>- License: None
>- Tags: data
>- A curated list of open-access databases with human structural mri data

