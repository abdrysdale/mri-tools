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
	- [reconstruction](#reconstruction)
	- [analysis](#analysis)
	- [processing](#processing)
	- [simulation](#simulation)
	- [segmentation](#segmentation)
	- [visualisation](#visualisation)
	- [fetal](#fetal)
	- [data](#data)
	- [brain](#brain)
	- [ml](#ml)
	- [qa](#qa)
	- [renal](#renal)
	- [spinal](#spinal)
	- [muscle](#muscle)
- [languages](#languages)
	- [python](#python)
	- [c++](#c++)
	- [julia](#julia)
	- [c](#c)
	- [javascript](#javascript)
	- [r](#r)
	- [jupyter](#jupyter)

## Stats
- Total repos: 60
- Languages:

| Language | Count |
|---|---|
| python | 43 |
| c++ | 16 |
| julia | 7 |
| c | 5 |
| javascript | 4 |
| r | 2 |
| jupyter | 1 |

- Tags:

| Tag | Count |
|---|---|
| reconstruction | 17 |
| analysis | 15 |
| processing | 13 |
| simulation | 12 |
| segmentation | 7 |
| visualisation | 6 |
| fetal | 6 |
| data | 5 |
| brain | 4 |
| ml | 3 |
| qa | 3 |
| renal | 1 |
| spinal | 1 |
| muscle | 1 |

- Licenses:

| Licence | Count |
|---|---|
| mit | 22 |
| apache | 11 |
| bsd | 10 |
| gplv3 | 8 |
| agplv3 | 2 |
| lgplv3 | 2 |
| mpl | 2 |
| none | 2 |
| gplv2 | 1 |



## Tags
### Reconstruction <a name="reconstruction"></a>
- [scanhub](https://github.com/brain-link/scanhub)
>- Languages: python
>- License: GPLv3
>- Tags: simulation, reconstruction, processing
>- Multi Modal Acquisition Software, Which Allows Individualizable, Modular And Cloud-Based Processing Of Functional And Anatomical Medical Images.

- [eptlib](https://github.com/eptlib/eptlib)
>- Languages: c++, python
>- License: MIT
>- Tags: reconstruction
>- Collection Of C++ Implementations Of Electric Properties Tomography (Ept) Methods.

- [sirf](https://github.com/SyneRBI/SIRF?tab=readme-ov-file)
>- Languages: c++, python
>- License: GPLv2
>- Tags: reconstruction
>- Open Source Toolkit For The Reconstruction Of Pet And Mri Raw Data.

- [hdr-mri](https://github.com/shakes76/sHDR)
>- Languages: c++
>- License: Apache
>- Tags: reconstruction
>- Takes As Input Coregistered Mr Images (Preferrably Of Different Contrasts), Non-Linearly Combines Them And Outputs A Single Hdr Mr Image.

- [gadgetron](https://github.com/gadgetron/gadgetron)
>- Languages: c++
>- License: MIT
>- Tags: reconstruction
>- Open Source Project For Medical Image Reconstruction

- [powergrid](https://github.com/mrfil/PowerGrid)
>- Languages: c++
>- License: MIT
>- Tags: reconstruction
>- Cpu And Gpu Accelerated Iterative Magnetic Resonance Imaging Reconstruction

- [tensorflow-mri](https://github.com/mrphys/tensorflow-mri)
>- Languages: python
>- License: Apache
>- Tags: reconstruction, ml
>- Library Of Tensorflow Operators For Computational Mri

- [mri-reco](https://github.com/MagneticResonanceImaging/MRIReco.jl)
>- Languages: julia
>- License: MIT
>- Tags: reconstruction
>- Julia Package For Magnetic Resonance Imaging

- [nlsam](https://github.com/samuelstjean/nlsam)
>- Languages: python
>- License: GPLv3
>- Tags: reconstruction
>- Implementation For The Non Local Spatial And Angular Matching (Nlsam) Denoising Algorithm For Diffusion Mri

- [bart](http://mrirecon.github.io/bart/)
>- Languages: c, c++
>- License: BSD
>- Tags: reconstruction
>- Free And Open-Source Image-Reconstruction Framework For Computational Magnetic Resonance Imaging

- [nesvor](https://github.com/daviddmc/NeSVoR)
>- Languages: python
>- License: MIT
>- Tags: reconstruction, fetal
>- Gpu-Accelerated Slice-To-Volume Reconstruction (Both Rigid And Deformable)

- [niftymic](https://github.com/gift-surg/NiftyMIC)
>- Languages: python
>- License: BSD
>- Tags: reconstruction, fetal
>- Toolkit For Research Developed Within The Gift-Surg Project To Reconstruct An Isotropic, High-Resolution Volume From Multiple, Possibly Motion-Corrupted, Stacks Of Low-Resolution 2D Slices

- [svrtk](https://github.com/SVRTK/SVRTK)
>- Languages: c++
>- License: Apache
>- Tags: reconstruction, fetal
>- Mirtk Based Svr Reconstruction

- [mialsuperresolutiontoolkit](https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit)
>- Languages: c++, python
>- License: BSD
>- Tags: reconstruction, fetal
>- C++ And Python Tools Necessary To Perform Motion-Robust Super-Resolution Fetal Mri Reconstruction

- [qslprep](https://github.com/PennLINC/qsiprep)
>- Languages: python
>- License: BSD
>- Tags: processing, reconstruction, analysis
>- Preprocessing And Reconstruction Of Diffusion Mri

- [direct](https://github.com/NKI-AI/direct)
>- Languages: python
>- License: Apache
>- Tags: reconstruction
>- Deep Learning Framework For Mri Reconstruction

- [synthseg](https://github.com/BBillot/SynthSeg)
>- Languages: python
>- License: Apache
>- Tags: ml, reconstruction
>- Deep Learning Tool For Segmentation Of Brain Scans Of Any Contrast And Resolution

### Analysis <a name="analysis"></a>
- [slicer](https://github.com/Slicer/Slicer)
>- Languages: python, c++
>- License: BSD
>- Tags: simulation, analysis, visualisation
>- Open Source Software Package For Visualization And Image Analysis.

- [ukat](https://github.com/UKRIN-MAPS/ukat)
>- Languages: python
>- License: GPLv3
>- Tags: analysis, renal
>- Ukat Is A Vendor Agnostic Framework For The Analysis Of Quantitative Renal Mri Data

- [freesurfer](https://github.com/freesurfer/freesurfer)
>- Languages: c++, c, python
>- License: GPLv3
>- Tags: analysis, visualisation, brain
>- Analysis And Visualization Of Neuroimaging Data From Cross-Sectional And Longitudinal Studies

- [simple-itk](https://github.com/SimpleITK/SimpleITK)
>- Languages: c++, python, R
>- License: Apache
>- Tags: segmentation, analysis
>- Image Analysis Toolkit With A Large Number Of Components Supporting General Filtering Operations, Image Segmentation And Registration

- [quit](https://github.com/spinicist/QUIT)
>- Languages: c++, python
>- License: MPL
>- Tags: analysis
>- Collection Of Programs For Processing Quantitative Mri Data

- [madym](https://gitlab.com/manchester_qbi/manchester_qbi_public/madym_cxx)
>- Languages: c++
>- License: Apache
>- Tags: analysis, processing
>- C++ Toolkit For Quantative Dce-Mri And Dwi-Mri Analysis

- [myoqmri](https://github.com/fsantini/MyoQMRI)
>- Languages: python
>- License: GPLv3
>- Tags: analysis, muscle
>- Quantitative Mri Of The Muscles

- [mrqy](https://github.com/ccipd/MRQy)
>- Languages: javascript, python
>- License: BSD
>- Tags: qa, analysis
>- Generate Several Tags And Noise/Information Measurements For Quality Assessment

- [fastmri](https://github.com/facebookresearch/fastMRI)
>- Languages: python
>- License: MIT
>- Tags: ml, analysis, data
>- Research Project From Facebook Ai Research (Fair) And Nyu Langone Health To Investigate The Use Of Ai To Make Mri Scans Faster

- [affirm](https://github.com/allard-shi/affirm)
>- Languages: python
>- License: MIT
>- Tags: analysis, fetal
>- Deep Recursive Fetal Motion Estimation And Correction Based On Slice And Volume Affinity Fusion

- [mri-research-tools](https://github.com/korbinian90/MriResearchTools.jl)
>- Languages: julia
>- License: MIT
>- Tags: processing, analysis, simulation
>- Specialized Tools For Mri

- [dcemri](https://github.com/davidssmith/DCEMRI.jl)
>- Languages: julia
>- License: MIT
>- Tags: analysis
>- Open Source Toolkit For Dynamic Contrast Enhanced Mri Analysis

- [qslprep](https://github.com/PennLINC/qsiprep)
>- Languages: python
>- License: BSD
>- Tags: processing, reconstruction, analysis
>- Preprocessing And Reconstruction Of Diffusion Mri

- [braingraph](https://github.com/cwatson/brainGraph)
>- Languages: r
>- License: None
>- Tags: analysis
>- R Package For Performing Graph Theory Analyses Of Brain Mri Data

- [mriqc](https://github.com/nipreps/mriqc)
>- Languages: javascript, python
>- License: Apache
>- Tags: qa, analysis
>- Extracts No-Reference Iqms (Image Quality Metrics) From Structural (T1W And T2W) And Functional Mri (Magnetic Resonance Imaging) Data

### Processing <a name="processing"></a>
- [vespa](https://github.com/vespa-mrs/vespa/)
>- Languages: python
>- License: BSD
>- Tags: simulation, data, processing
>- Integrated, Open Source, Open Development Platform For Magnetic Resonance Spectroscopy (Mrs) Research For Rf Pulse Design, Spectral Simulation And Prototyping, Creating Synthetic Mrs Data Sets And Interactive Spectral Data Processing And Analysis.

- [scanhub](https://github.com/brain-link/scanhub)
>- Languages: python
>- License: GPLv3
>- Tags: simulation, reconstruction, processing
>- Multi Modal Acquisition Software, Which Allows Individualizable, Modular And Cloud-Based Processing Of Functional And Anatomical Medical Images.

- [spinal-chord-toolbox](https://github.com/spinalcordtoolbox/spinalcordtoolbox)
>- Languages: python
>- License: LGPLv3
>- Tags: segmentation, processing, spinal
>- Comprehensive, Free And Open-Source Set Of Command-Line Tools Dedicated To The Processing And Analysis Of Spinal Cord Mri Data

- [madym](https://gitlab.com/manchester_qbi/manchester_qbi_public/madym_cxx)
>- Languages: c++
>- License: Apache
>- Tags: analysis, processing
>- C++ Toolkit For Quantative Dce-Mri And Dwi-Mri Analysis

- [decaes](https://github.com/jondeuce/DECAES.jl)
>- Languages: julia
>- License: MIT
>- Tags: processing
>- Julia Implementation Of The Matlab Toolbox From The Ubc Mri Research Centre For Computing Voxelwise T2-Distributions From Multi Spin-Echo Mri Images Using The Extended Phase Graph Algorithm With Stimulated Echo Corrections

- [mrtrix3](https://github.com/MRtrix3/mrtrix3)
>- Languages: c++, python
>- License: MPL
>- Tags: processing
>- Set Of Tools To Perform Various Types Of Diffusion Mri Analyses, From Various Forms Of Tractography Through To Next-Generation Group-Level Analyses

- [smriprep](https://github.com/nipreps/smriprep)
>- Languages: python
>- License: Apache
>- Tags: processing
>- Structural Magnetic Resonance Imaging (Smri) Data Preprocessing Pipeline That Is Designed To Provide An Easily Accessible, State-Of-The-Art Interface That Is Robust To Variations In Scan Acquisition Protocols And That Requires Minimal User Input, While Providing Easily Interpretable And Comprehensive Error And Output Reporting

- [flow4d](https://github.com/saitta-s/flow4D)
>- Languages: python
>- License: MIT
>- Tags: processing
>- Work With 4D Flow Mri Acquisitions For Cfd Applications

- [mri-research-tools](https://github.com/korbinian90/MriResearchTools.jl)
>- Languages: julia
>- License: MIT
>- Tags: processing, analysis, simulation
>- Specialized Tools For Mri

- [pydeface](https://github.com/poldracklab/pydeface)
>- Languages: python
>- License: MIT
>- Tags: processing
>- A Tool To Remove Facial Structure From Mri Images.

- [mritopng](https://github.com/danishm/mritopng)
>- Languages: python
>- License: MIT
>- Tags: processing
>- A Simple Python Module To Make It Easy To Batch Convert Dicom Files To Png Images.

- [qslprep](https://github.com/PennLINC/qsiprep)
>- Languages: python
>- License: BSD
>- Tags: processing, reconstruction, analysis
>- Preprocessing And Reconstruction Of Diffusion Mri

- [intensity-normalization](https://github.com/jcreinhold/intensity-normalization)
>- Languages: python
>- License: Apache
>- Tags: processing
>- Various Methods To Normalize The Intensity Of Various Modalities Of Magnetic Resonance (Mr) Images, E.G., T1-Weighted (T1-W), T2-Weighted (T2-W), Fluid-Attenuated Inversion Recovery (Flair), And Proton Density-Weighted (Pd-W)

### Simulation <a name="simulation"></a>
- [pycoilgen](https://github.com/kev-m/pyCoilGen)
>- Languages: python
>- License: GPLv3
>- Tags: simulation
>- Open Source Tool For Generating Coil Winding Layouts, Such As Gradient Field Coils, Within The Mri And Nmr Environments.

- [virtual-mri-scanner](https://github.com/imr-framework/virtual-scanner)
>- Languages: python
>- License: AGPLv3
>- Tags: simulation
>- End-To-End Hybrid Magnetic Resonance Imaging (Mri) Simulator/Console Designed To Be Zero-Footprint, Modular, And Supported By Open-Source Standards.

- [cosimpy](https://github.com/umbertozanovello/CoSimPy)
>- Languages: python
>- License: MIT
>- Tags: simulation
>- Open Source Python Library Aiming To Combine Results From Electromagnetic (Em) Simulation With Circuits Analysis Through A Cosimulation Environment.

- [vespa](https://github.com/vespa-mrs/vespa/)
>- Languages: python
>- License: BSD
>- Tags: simulation, data, processing
>- Integrated, Open Source, Open Development Platform For Magnetic Resonance Spectroscopy (Mrs) Research For Rf Pulse Design, Spectral Simulation And Prototyping, Creating Synthetic Mrs Data Sets And Interactive Spectral Data Processing And Analysis.

- [scanhub](https://github.com/brain-link/scanhub)
>- Languages: python
>- License: GPLv3
>- Tags: simulation, reconstruction, processing
>- Multi Modal Acquisition Software, Which Allows Individualizable, Modular And Cloud-Based Processing Of Functional And Anatomical Medical Images.

- [slicer](https://github.com/Slicer/Slicer)
>- Languages: python, c++
>- License: BSD
>- Tags: simulation, analysis, visualisation
>- Open Source Software Package For Visualization And Image Analysis.

- [pypulseq](https://github.com/imr-framework/pypulseq/)
>- Languages: python
>- License: AGPLv3
>- Tags: simulation
>- Enables Vendor-Neutral Pulse Sequence Design In Python [1,2]. The Pulse Sequences Can Be Exported As A .Seq File To Be Run On Siemens/Ge/Bruker Hardware By Leveraging Their Respective Pulseq Interpreters.

- [koma](https://github.com/JuliaHealth/KomaMRI.jl)
>- Languages: julia
>- License: MIT
>- Tags: simulation
>- Pulseq-Compatible Framework To Efficiently Simulate Magnetic Resonance Imaging (Mri) Acquisitions

- [gropt](https://github.com/mloecher/gropt)
>- Languages: c, python
>- License: GPLv3
>- Tags: simulation
>-  Toolbox For Mri Gradient Optimization

- [disimpy](https://github.com/kerkelae/disimpy)
>- Languages: python
>- License: MIT
>- Tags: simulation
>- Python Package For Generating Simulated Diffusion-Weighted Mr Signals That Can Be Useful In The Development And Validation Of Data Acquisition And Analysis Methods

- [mri-generalized-bloch](https://github.com/JakobAsslaender/MRIgeneralizedBloch.jl)
>- Languages: julia
>- License: MIT
>- Tags: simulation
>- Julia Package That Implements The Generalized Bloch Equations For Modeling The Dynamics Of The Semi-Solid Spin Pool In Magnetic Resonance Imaging (Mri), And Its Exchange With The Free Spin Pool

- [mri-research-tools](https://github.com/korbinian90/MriResearchTools.jl)
>- Languages: julia
>- License: MIT
>- Tags: processing, analysis, simulation
>- Specialized Tools For Mri

### Segmentation <a name="segmentation"></a>
- [dl-direct](https://github.com/SCAN-NRAD/DL-DiReCT)
>- Languages: python
>- License: BSD
>- Tags: segmentation
>- Combines A Deep Learning-Based Neuroanatomy Segmentation And Cortex Parcellation With A Diffeomorphic Registration Technique To Measure Cortical Thickness From T1W Mri

- [dafne](https://github.com/dafne-imaging/dafne)
>- Languages: python
>- License: GPLv3
>- Tags: segmentation
>- Program For The Segmentation Of Medical Images. It Relies On A Server To Provide Deep Learning Models To Aid The Segmentation, And Incremental Learning Is Used To Improve The Performance

- [spinal-chord-toolbox](https://github.com/spinalcordtoolbox/spinalcordtoolbox)
>- Languages: python
>- License: LGPLv3
>- Tags: segmentation, processing, spinal
>- Comprehensive, Free And Open-Source Set Of Command-Line Tools Dedicated To The Processing And Analysis Of Spinal Cord Mri Data

- [dwybss](https://github.com/mmromero/dwybss)
>- Languages: python
>- License: LGPLv3
>- Tags: segmentation, brain
>- Separate Microstructure Tissue Components From The Diffusion Mri Signal, Characterize The Volume Fractions, And T2 Maps Of These Compartments

- [simple-itk](https://github.com/SimpleITK/SimpleITK)
>- Languages: c++, python, R
>- License: Apache
>- Tags: segmentation, analysis
>- Image Analysis Toolkit With A Large Number Of Components Supporting General Filtering Operations, Image Segmentation And Registration

- [quicknat](https://github.com/ai-med/quickNAT_pytorch)
>- Languages: python
>- License: MIT
>- Tags: segmentation, brain
>- Fully Convolutional Network For Quick And Accurate Segmentation Of Neuroanatomy And Quality Control Of Structure-Wise Segmentations

- [brainchop](https://github.com/neuroneural/brainchop)
>- Languages: javascript, python
>- License: MIT
>- Tags: segmentation, visualisation
>- In-Browser 3D Mri Rendering And Segmentation

### Visualisation <a name="visualisation"></a>
- [slicer](https://github.com/Slicer/Slicer)
>- Languages: python, c++
>- License: BSD
>- Tags: simulation, analysis, visualisation
>- Open Source Software Package For Visualization And Image Analysis.

- [ukf-tractography](https://github.com/pnlbwh/ukftractography)
>- Languages: c, c++
>- License: BSD
>- Tags: visualisation, brain
>- Framework Which Uses An Unscented Kalman Filter For Performing Tractography

- [freesurfer](https://github.com/freesurfer/freesurfer)
>- Languages: c++, c, python
>- License: GPLv3
>- Tags: analysis, visualisation, brain
>- Analysis And Visualization Of Neuroimaging Data From Cross-Sectional And Longitudinal Studies

- [gif_your_nifti](https://github.com/miykael/gif_your_nifti)
>- Languages: python
>- License: BSD
>- Tags: visualisation
>- Create Nice Looking Gifs From Your Nifti (.Nii Or .Nii.Gz) Files With A Simple Command

- [brainchop](https://github.com/neuroneural/brainchop)
>- Languages: javascript, python
>- License: MIT
>- Tags: segmentation, visualisation
>- In-Browser 3D Mri Rendering And Segmentation

- [mri-viewer](https://github.com/epam/mriviewer)
>- Languages: javascript
>- License: Apache
>- Tags: visualisation
>- High Performance Web Tool For Advanced Visualization (Both In 2D And 3D Modes) Medical Volumetric Data, Provided In Popular File Formats: Dicom, Nifti, Ktx, Hdr

### Fetal <a name="fetal"></a>
- [nesvor](https://github.com/daviddmc/NeSVoR)
>- Languages: python
>- License: MIT
>- Tags: reconstruction, fetal
>- Gpu-Accelerated Slice-To-Volume Reconstruction (Both Rigid And Deformable)

- [affirm](https://github.com/allard-shi/affirm)
>- Languages: python
>- License: MIT
>- Tags: analysis, fetal
>- Deep Recursive Fetal Motion Estimation And Correction Based On Slice And Volume Affinity Fusion

- [niftymic](https://github.com/gift-surg/NiftyMIC)
>- Languages: python
>- License: BSD
>- Tags: reconstruction, fetal
>- Toolkit For Research Developed Within The Gift-Surg Project To Reconstruct An Isotropic, High-Resolution Volume From Multiple, Possibly Motion-Corrupted, Stacks Of Low-Resolution 2D Slices

- [svrtk](https://github.com/SVRTK/SVRTK)
>- Languages: c++
>- License: Apache
>- Tags: reconstruction, fetal
>- Mirtk Based Svr Reconstruction

- [mialsuperresolutiontoolkit](https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit)
>- Languages: c++, python
>- License: BSD
>- Tags: reconstruction, fetal
>- C++ And Python Tools Necessary To Perform Motion-Robust Super-Resolution Fetal Mri Reconstruction

- [fetal-iqa](https://github.com/daviddmc/fetal-IQA)
>- Languages: python
>- License: MIT
>- Tags: qa, fetal
>-  Image Quality Assessment (Iqa) Method For Fetal Mri

### Data <a name="data"></a>
- [vespa](https://github.com/vespa-mrs/vespa/)
>- Languages: python
>- License: BSD
>- Tags: simulation, data, processing
>- Integrated, Open Source, Open Development Platform For Magnetic Resonance Spectroscopy (Mrs) Research For Rf Pulse Design, Spectral Simulation And Prototyping, Creating Synthetic Mrs Data Sets And Interactive Spectral Data Processing And Analysis.

- [ismrm-raw-data-format](https://ismrmrd.github.io/apidocs/1.5.0/)
>- Languages: c, c++, python
>- License: MIT
>- Tags: data
>-  Common Raw Data Format, Which Attempts To Capture The Data Fields That Are Required To Describe The Magnetic Resonance Experiment With Enough Detail To Reconstruct Images

- [fastmri](https://github.com/facebookresearch/fastMRI)
>- Languages: python
>- License: MIT
>- Tags: ml, analysis, data
>- Research Project From Facebook Ai Research (Fair) And Nyu Langone Health To Investigate The Use Of Ai To Make Mri Scans Faster

- [nlft](https://github.com/JuliaNeuroscience/NIfTI.jl)
>- Languages: julia
>- License: MIT
>- Tags: data
>- Julia Module For Reading/Writing Nifti Mri Files

- [openmorph](https://github.com/cMadan/openMorph)
>- Languages: jupyter
>- License: None
>- Tags: data
>- Curated List Of Open-Access Databases With Human Structural Mri Data

### Brain <a name="brain"></a>
- [ukf-tractography](https://github.com/pnlbwh/ukftractography)
>- Languages: c, c++
>- License: BSD
>- Tags: visualisation, brain
>- Framework Which Uses An Unscented Kalman Filter For Performing Tractography

- [dwybss](https://github.com/mmromero/dwybss)
>- Languages: python
>- License: LGPLv3
>- Tags: segmentation, brain
>- Separate Microstructure Tissue Components From The Diffusion Mri Signal, Characterize The Volume Fractions, And T2 Maps Of These Compartments

- [freesurfer](https://github.com/freesurfer/freesurfer)
>- Languages: c++, c, python
>- License: GPLv3
>- Tags: analysis, visualisation, brain
>- Analysis And Visualization Of Neuroimaging Data From Cross-Sectional And Longitudinal Studies

- [quicknat](https://github.com/ai-med/quickNAT_pytorch)
>- Languages: python
>- License: MIT
>- Tags: segmentation, brain
>- Fully Convolutional Network For Quick And Accurate Segmentation Of Neuroanatomy And Quality Control Of Structure-Wise Segmentations

### Ml <a name="ml"></a>
- [tensorflow-mri](https://github.com/mrphys/tensorflow-mri)
>- Languages: python
>- License: Apache
>- Tags: reconstruction, ml
>- Library Of Tensorflow Operators For Computational Mri

- [fastmri](https://github.com/facebookresearch/fastMRI)
>- Languages: python
>- License: MIT
>- Tags: ml, analysis, data
>- Research Project From Facebook Ai Research (Fair) And Nyu Langone Health To Investigate The Use Of Ai To Make Mri Scans Faster

- [synthseg](https://github.com/BBillot/SynthSeg)
>- Languages: python
>- License: Apache
>- Tags: ml, reconstruction
>- Deep Learning Tool For Segmentation Of Brain Scans Of Any Contrast And Resolution

### Qa <a name="qa"></a>
- [mrqy](https://github.com/ccipd/MRQy)
>- Languages: javascript, python
>- License: BSD
>- Tags: qa, analysis
>- Generate Several Tags And Noise/Information Measurements For Quality Assessment

- [fetal-iqa](https://github.com/daviddmc/fetal-IQA)
>- Languages: python
>- License: MIT
>- Tags: qa, fetal
>-  Image Quality Assessment (Iqa) Method For Fetal Mri

- [mriqc](https://github.com/nipreps/mriqc)
>- Languages: javascript, python
>- License: Apache
>- Tags: qa, analysis
>- Extracts No-Reference Iqms (Image Quality Metrics) From Structural (T1W And T2W) And Functional Mri (Magnetic Resonance Imaging) Data

### Renal <a name="renal"></a>
- [ukat](https://github.com/UKRIN-MAPS/ukat)
>- Languages: python
>- License: GPLv3
>- Tags: analysis, renal
>- Ukat Is A Vendor Agnostic Framework For The Analysis Of Quantitative Renal Mri Data

### Spinal <a name="spinal"></a>
- [spinal-chord-toolbox](https://github.com/spinalcordtoolbox/spinalcordtoolbox)
>- Languages: python
>- License: LGPLv3
>- Tags: segmentation, processing, spinal
>- Comprehensive, Free And Open-Source Set Of Command-Line Tools Dedicated To The Processing And Analysis Of Spinal Cord Mri Data

### Muscle <a name="muscle"></a>
- [myoqmri](https://github.com/fsantini/MyoQMRI)
>- Languages: python
>- License: GPLv3
>- Tags: analysis, muscle
>- Quantitative Mri Of The Muscles



## Languages
### Python <a name="python"></a>
- [pycoilgen](https://github.com/kev-m/pyCoilGen)
>- Languages: python
>- License: GPLv3
>- Tags: simulation
>- Open Source Tool For Generating Coil Winding Layouts, Such As Gradient Field Coils, Within The Mri And Nmr Environments.

- [virtual-mri-scanner](https://github.com/imr-framework/virtual-scanner)
>- Languages: python
>- License: AGPLv3
>- Tags: simulation
>- End-To-End Hybrid Magnetic Resonance Imaging (Mri) Simulator/Console Designed To Be Zero-Footprint, Modular, And Supported By Open-Source Standards.

- [cosimpy](https://github.com/umbertozanovello/CoSimPy)
>- Languages: python
>- License: MIT
>- Tags: simulation
>- Open Source Python Library Aiming To Combine Results From Electromagnetic (Em) Simulation With Circuits Analysis Through A Cosimulation Environment.

- [vespa](https://github.com/vespa-mrs/vespa/)
>- Languages: python
>- License: BSD
>- Tags: simulation, data, processing
>- Integrated, Open Source, Open Development Platform For Magnetic Resonance Spectroscopy (Mrs) Research For Rf Pulse Design, Spectral Simulation And Prototyping, Creating Synthetic Mrs Data Sets And Interactive Spectral Data Processing And Analysis.

- [scanhub](https://github.com/brain-link/scanhub)
>- Languages: python
>- License: GPLv3
>- Tags: simulation, reconstruction, processing
>- Multi Modal Acquisition Software, Which Allows Individualizable, Modular And Cloud-Based Processing Of Functional And Anatomical Medical Images.

- [slicer](https://github.com/Slicer/Slicer)
>- Languages: python, c++
>- License: BSD
>- Tags: simulation, analysis, visualisation
>- Open Source Software Package For Visualization And Image Analysis.

- [eptlib](https://github.com/eptlib/eptlib)
>- Languages: c++, python
>- License: MIT
>- Tags: reconstruction
>- Collection Of C++ Implementations Of Electric Properties Tomography (Ept) Methods.

- [sirf](https://github.com/SyneRBI/SIRF?tab=readme-ov-file)
>- Languages: c++, python
>- License: GPLv2
>- Tags: reconstruction
>- Open Source Toolkit For The Reconstruction Of Pet And Mri Raw Data.

- [dl-direct](https://github.com/SCAN-NRAD/DL-DiReCT)
>- Languages: python
>- License: BSD
>- Tags: segmentation
>- Combines A Deep Learning-Based Neuroanatomy Segmentation And Cortex Parcellation With A Diffeomorphic Registration Technique To Measure Cortical Thickness From T1W Mri

- [dafne](https://github.com/dafne-imaging/dafne)
>- Languages: python
>- License: GPLv3
>- Tags: segmentation
>- Program For The Segmentation Of Medical Images. It Relies On A Server To Provide Deep Learning Models To Aid The Segmentation, And Incremental Learning Is Used To Improve The Performance

- [ukat](https://github.com/UKRIN-MAPS/ukat)
>- Languages: python
>- License: GPLv3
>- Tags: analysis, renal
>- Ukat Is A Vendor Agnostic Framework For The Analysis Of Quantitative Renal Mri Data

- [pypulseq](https://github.com/imr-framework/pypulseq/)
>- Languages: python
>- License: AGPLv3
>- Tags: simulation
>- Enables Vendor-Neutral Pulse Sequence Design In Python [1,2]. The Pulse Sequences Can Be Exported As A .Seq File To Be Run On Siemens/Ge/Bruker Hardware By Leveraging Their Respective Pulseq Interpreters.

- [spinal-chord-toolbox](https://github.com/spinalcordtoolbox/spinalcordtoolbox)
>- Languages: python
>- License: LGPLv3
>- Tags: segmentation, processing, spinal
>- Comprehensive, Free And Open-Source Set Of Command-Line Tools Dedicated To The Processing And Analysis Of Spinal Cord Mri Data

- [dwybss](https://github.com/mmromero/dwybss)
>- Languages: python
>- License: LGPLv3
>- Tags: segmentation, brain
>- Separate Microstructure Tissue Components From The Diffusion Mri Signal, Characterize The Volume Fractions, And T2 Maps Of These Compartments

- [freesurfer](https://github.com/freesurfer/freesurfer)
>- Languages: c++, c, python
>- License: GPLv3
>- Tags: analysis, visualisation, brain
>- Analysis And Visualization Of Neuroimaging Data From Cross-Sectional And Longitudinal Studies

- [simple-itk](https://github.com/SimpleITK/SimpleITK)
>- Languages: c++, python, R
>- License: Apache
>- Tags: segmentation, analysis
>- Image Analysis Toolkit With A Large Number Of Components Supporting General Filtering Operations, Image Segmentation And Registration

- [ismrm-raw-data-format](https://ismrmrd.github.io/apidocs/1.5.0/)
>- Languages: c, c++, python
>- License: MIT
>- Tags: data
>-  Common Raw Data Format, Which Attempts To Capture The Data Fields That Are Required To Describe The Magnetic Resonance Experiment With Enough Detail To Reconstruct Images

- [tensorflow-mri](https://github.com/mrphys/tensorflow-mri)
>- Languages: python
>- License: Apache
>- Tags: reconstruction, ml
>- Library Of Tensorflow Operators For Computational Mri

- [quit](https://github.com/spinicist/QUIT)
>- Languages: c++, python
>- License: MPL
>- Tags: analysis
>- Collection Of Programs For Processing Quantitative Mri Data

- [gropt](https://github.com/mloecher/gropt)
>- Languages: c, python
>- License: GPLv3
>- Tags: simulation
>-  Toolbox For Mri Gradient Optimization

- [disimpy](https://github.com/kerkelae/disimpy)
>- Languages: python
>- License: MIT
>- Tags: simulation
>- Python Package For Generating Simulated Diffusion-Weighted Mr Signals That Can Be Useful In The Development And Validation Of Data Acquisition And Analysis Methods

- [myoqmri](https://github.com/fsantini/MyoQMRI)
>- Languages: python
>- License: GPLv3
>- Tags: analysis, muscle
>- Quantitative Mri Of The Muscles

- [mrqy](https://github.com/ccipd/MRQy)
>- Languages: javascript, python
>- License: BSD
>- Tags: qa, analysis
>- Generate Several Tags And Noise/Information Measurements For Quality Assessment

- [nlsam](https://github.com/samuelstjean/nlsam)
>- Languages: python
>- License: GPLv3
>- Tags: reconstruction
>- Implementation For The Non Local Spatial And Angular Matching (Nlsam) Denoising Algorithm For Diffusion Mri

- [mrtrix3](https://github.com/MRtrix3/mrtrix3)
>- Languages: c++, python
>- License: MPL
>- Tags: processing
>- Set Of Tools To Perform Various Types Of Diffusion Mri Analyses, From Various Forms Of Tractography Through To Next-Generation Group-Level Analyses

- [smriprep](https://github.com/nipreps/smriprep)
>- Languages: python
>- License: Apache
>- Tags: processing
>- Structural Magnetic Resonance Imaging (Smri) Data Preprocessing Pipeline That Is Designed To Provide An Easily Accessible, State-Of-The-Art Interface That Is Robust To Variations In Scan Acquisition Protocols And That Requires Minimal User Input, While Providing Easily Interpretable And Comprehensive Error And Output Reporting

- [fastmri](https://github.com/facebookresearch/fastMRI)
>- Languages: python
>- License: MIT
>- Tags: ml, analysis, data
>- Research Project From Facebook Ai Research (Fair) And Nyu Langone Health To Investigate The Use Of Ai To Make Mri Scans Faster

- [flow4d](https://github.com/saitta-s/flow4D)
>- Languages: python
>- License: MIT
>- Tags: processing
>- Work With 4D Flow Mri Acquisitions For Cfd Applications

- [nesvor](https://github.com/daviddmc/NeSVoR)
>- Languages: python
>- License: MIT
>- Tags: reconstruction, fetal
>- Gpu-Accelerated Slice-To-Volume Reconstruction (Both Rigid And Deformable)

- [affirm](https://github.com/allard-shi/affirm)
>- Languages: python
>- License: MIT
>- Tags: analysis, fetal
>- Deep Recursive Fetal Motion Estimation And Correction Based On Slice And Volume Affinity Fusion

- [niftymic](https://github.com/gift-surg/NiftyMIC)
>- Languages: python
>- License: BSD
>- Tags: reconstruction, fetal
>- Toolkit For Research Developed Within The Gift-Surg Project To Reconstruct An Isotropic, High-Resolution Volume From Multiple, Possibly Motion-Corrupted, Stacks Of Low-Resolution 2D Slices

- [mialsuperresolutiontoolkit](https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit)
>- Languages: c++, python
>- License: BSD
>- Tags: reconstruction, fetal
>- C++ And Python Tools Necessary To Perform Motion-Robust Super-Resolution Fetal Mri Reconstruction

- [fetal-iqa](https://github.com/daviddmc/fetal-IQA)
>- Languages: python
>- License: MIT
>- Tags: qa, fetal
>-  Image Quality Assessment (Iqa) Method For Fetal Mri

- [quicknat](https://github.com/ai-med/quickNAT_pytorch)
>- Languages: python
>- License: MIT
>- Tags: segmentation, brain
>- Fully Convolutional Network For Quick And Accurate Segmentation Of Neuroanatomy And Quality Control Of Structure-Wise Segmentations

- [pydeface](https://github.com/poldracklab/pydeface)
>- Languages: python
>- License: MIT
>- Tags: processing
>- A Tool To Remove Facial Structure From Mri Images.

- [mritopng](https://github.com/danishm/mritopng)
>- Languages: python
>- License: MIT
>- Tags: processing
>- A Simple Python Module To Make It Easy To Batch Convert Dicom Files To Png Images.

- [qslprep](https://github.com/PennLINC/qsiprep)
>- Languages: python
>- License: BSD
>- Tags: processing, reconstruction, analysis
>- Preprocessing And Reconstruction Of Diffusion Mri

- [gif_your_nifti](https://github.com/miykael/gif_your_nifti)
>- Languages: python
>- License: BSD
>- Tags: visualisation
>- Create Nice Looking Gifs From Your Nifti (.Nii Or .Nii.Gz) Files With A Simple Command

- [direct](https://github.com/NKI-AI/direct)
>- Languages: python
>- License: Apache
>- Tags: reconstruction
>- Deep Learning Framework For Mri Reconstruction

- [brainchop](https://github.com/neuroneural/brainchop)
>- Languages: javascript, python
>- License: MIT
>- Tags: segmentation, visualisation
>- In-Browser 3D Mri Rendering And Segmentation

- [mriqc](https://github.com/nipreps/mriqc)
>- Languages: javascript, python
>- License: Apache
>- Tags: qa, analysis
>- Extracts No-Reference Iqms (Image Quality Metrics) From Structural (T1W And T2W) And Functional Mri (Magnetic Resonance Imaging) Data

- [synthseg](https://github.com/BBillot/SynthSeg)
>- Languages: python
>- License: Apache
>- Tags: ml, reconstruction
>- Deep Learning Tool For Segmentation Of Brain Scans Of Any Contrast And Resolution

- [intensity-normalization](https://github.com/jcreinhold/intensity-normalization)
>- Languages: python
>- License: Apache
>- Tags: processing
>- Various Methods To Normalize The Intensity Of Various Modalities Of Magnetic Resonance (Mr) Images, E.G., T1-Weighted (T1-W), T2-Weighted (T2-W), Fluid-Attenuated Inversion Recovery (Flair), And Proton Density-Weighted (Pd-W)

### C++ <a name="c++"></a>
- [slicer](https://github.com/Slicer/Slicer)
>- Languages: python, c++
>- License: BSD
>- Tags: simulation, analysis, visualisation
>- Open Source Software Package For Visualization And Image Analysis.

- [eptlib](https://github.com/eptlib/eptlib)
>- Languages: c++, python
>- License: MIT
>- Tags: reconstruction
>- Collection Of C++ Implementations Of Electric Properties Tomography (Ept) Methods.

- [sirf](https://github.com/SyneRBI/SIRF?tab=readme-ov-file)
>- Languages: c++, python
>- License: GPLv2
>- Tags: reconstruction
>- Open Source Toolkit For The Reconstruction Of Pet And Mri Raw Data.

- [hdr-mri](https://github.com/shakes76/sHDR)
>- Languages: c++
>- License: Apache
>- Tags: reconstruction
>- Takes As Input Coregistered Mr Images (Preferrably Of Different Contrasts), Non-Linearly Combines Them And Outputs A Single Hdr Mr Image.

- [gadgetron](https://github.com/gadgetron/gadgetron)
>- Languages: c++
>- License: MIT
>- Tags: reconstruction
>- Open Source Project For Medical Image Reconstruction

- [powergrid](https://github.com/mrfil/PowerGrid)
>- Languages: c++
>- License: MIT
>- Tags: reconstruction
>- Cpu And Gpu Accelerated Iterative Magnetic Resonance Imaging Reconstruction

- [ukf-tractography](https://github.com/pnlbwh/ukftractography)
>- Languages: c, c++
>- License: BSD
>- Tags: visualisation, brain
>- Framework Which Uses An Unscented Kalman Filter For Performing Tractography

- [freesurfer](https://github.com/freesurfer/freesurfer)
>- Languages: c++, c, python
>- License: GPLv3
>- Tags: analysis, visualisation, brain
>- Analysis And Visualization Of Neuroimaging Data From Cross-Sectional And Longitudinal Studies

- [simple-itk](https://github.com/SimpleITK/SimpleITK)
>- Languages: c++, python, R
>- License: Apache
>- Tags: segmentation, analysis
>- Image Analysis Toolkit With A Large Number Of Components Supporting General Filtering Operations, Image Segmentation And Registration

- [ismrm-raw-data-format](https://ismrmrd.github.io/apidocs/1.5.0/)
>- Languages: c, c++, python
>- License: MIT
>- Tags: data
>-  Common Raw Data Format, Which Attempts To Capture The Data Fields That Are Required To Describe The Magnetic Resonance Experiment With Enough Detail To Reconstruct Images

- [quit](https://github.com/spinicist/QUIT)
>- Languages: c++, python
>- License: MPL
>- Tags: analysis
>- Collection Of Programs For Processing Quantitative Mri Data

- [madym](https://gitlab.com/manchester_qbi/manchester_qbi_public/madym_cxx)
>- Languages: c++
>- License: Apache
>- Tags: analysis, processing
>- C++ Toolkit For Quantative Dce-Mri And Dwi-Mri Analysis

- [bart](http://mrirecon.github.io/bart/)
>- Languages: c, c++
>- License: BSD
>- Tags: reconstruction
>- Free And Open-Source Image-Reconstruction Framework For Computational Magnetic Resonance Imaging

- [mrtrix3](https://github.com/MRtrix3/mrtrix3)
>- Languages: c++, python
>- License: MPL
>- Tags: processing
>- Set Of Tools To Perform Various Types Of Diffusion Mri Analyses, From Various Forms Of Tractography Through To Next-Generation Group-Level Analyses

- [svrtk](https://github.com/SVRTK/SVRTK)
>- Languages: c++
>- License: Apache
>- Tags: reconstruction, fetal
>- Mirtk Based Svr Reconstruction

- [mialsuperresolutiontoolkit](https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit)
>- Languages: c++, python
>- License: BSD
>- Tags: reconstruction, fetal
>- C++ And Python Tools Necessary To Perform Motion-Robust Super-Resolution Fetal Mri Reconstruction

### Julia <a name="julia"></a>
- [koma](https://github.com/JuliaHealth/KomaMRI.jl)
>- Languages: julia
>- License: MIT
>- Tags: simulation
>- Pulseq-Compatible Framework To Efficiently Simulate Magnetic Resonance Imaging (Mri) Acquisitions

- [mri-generalized-bloch](https://github.com/JakobAsslaender/MRIgeneralizedBloch.jl)
>- Languages: julia
>- License: MIT
>- Tags: simulation
>- Julia Package That Implements The Generalized Bloch Equations For Modeling The Dynamics Of The Semi-Solid Spin Pool In Magnetic Resonance Imaging (Mri), And Its Exchange With The Free Spin Pool

- [decaes](https://github.com/jondeuce/DECAES.jl)
>- Languages: julia
>- License: MIT
>- Tags: processing
>- Julia Implementation Of The Matlab Toolbox From The Ubc Mri Research Centre For Computing Voxelwise T2-Distributions From Multi Spin-Echo Mri Images Using The Extended Phase Graph Algorithm With Stimulated Echo Corrections

- [mri-reco](https://github.com/MagneticResonanceImaging/MRIReco.jl)
>- Languages: julia
>- License: MIT
>- Tags: reconstruction
>- Julia Package For Magnetic Resonance Imaging

- [mri-research-tools](https://github.com/korbinian90/MriResearchTools.jl)
>- Languages: julia
>- License: MIT
>- Tags: processing, analysis, simulation
>- Specialized Tools For Mri

- [nlft](https://github.com/JuliaNeuroscience/NIfTI.jl)
>- Languages: julia
>- License: MIT
>- Tags: data
>- Julia Module For Reading/Writing Nifti Mri Files

- [dcemri](https://github.com/davidssmith/DCEMRI.jl)
>- Languages: julia
>- License: MIT
>- Tags: analysis
>- Open Source Toolkit For Dynamic Contrast Enhanced Mri Analysis

### C <a name="c"></a>
- [ukf-tractography](https://github.com/pnlbwh/ukftractography)
>- Languages: c, c++
>- License: BSD
>- Tags: visualisation, brain
>- Framework Which Uses An Unscented Kalman Filter For Performing Tractography

- [freesurfer](https://github.com/freesurfer/freesurfer)
>- Languages: c++, c, python
>- License: GPLv3
>- Tags: analysis, visualisation, brain
>- Analysis And Visualization Of Neuroimaging Data From Cross-Sectional And Longitudinal Studies

- [ismrm-raw-data-format](https://ismrmrd.github.io/apidocs/1.5.0/)
>- Languages: c, c++, python
>- License: MIT
>- Tags: data
>-  Common Raw Data Format, Which Attempts To Capture The Data Fields That Are Required To Describe The Magnetic Resonance Experiment With Enough Detail To Reconstruct Images

- [gropt](https://github.com/mloecher/gropt)
>- Languages: c, python
>- License: GPLv3
>- Tags: simulation
>-  Toolbox For Mri Gradient Optimization

- [bart](http://mrirecon.github.io/bart/)
>- Languages: c, c++
>- License: BSD
>- Tags: reconstruction
>- Free And Open-Source Image-Reconstruction Framework For Computational Magnetic Resonance Imaging

### Javascript <a name="javascript"></a>
- [mrqy](https://github.com/ccipd/MRQy)
>- Languages: javascript, python
>- License: BSD
>- Tags: qa, analysis
>- Generate Several Tags And Noise/Information Measurements For Quality Assessment

- [brainchop](https://github.com/neuroneural/brainchop)
>- Languages: javascript, python
>- License: MIT
>- Tags: segmentation, visualisation
>- In-Browser 3D Mri Rendering And Segmentation

- [mriqc](https://github.com/nipreps/mriqc)
>- Languages: javascript, python
>- License: Apache
>- Tags: qa, analysis
>- Extracts No-Reference Iqms (Image Quality Metrics) From Structural (T1W And T2W) And Functional Mri (Magnetic Resonance Imaging) Data

- [mri-viewer](https://github.com/epam/mriviewer)
>- Languages: javascript
>- License: Apache
>- Tags: visualisation
>- High Performance Web Tool For Advanced Visualization (Both In 2D And 3D Modes) Medical Volumetric Data, Provided In Popular File Formats: Dicom, Nifti, Ktx, Hdr

### R <a name="r"></a>
- [braingraph](https://github.com/cwatson/brainGraph)
>- Languages: r
>- License: None
>- Tags: analysis
>- R Package For Performing Graph Theory Analyses Of Brain Mri Data

### Jupyter <a name="jupyter"></a>
- [openmorph](https://github.com/cMadan/openMorph)
>- Languages: jupyter
>- License: None
>- Tags: data
>- Curated List Of Open-Access Databases With Human Structural Mri Data

