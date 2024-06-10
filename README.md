# MRI Tools

A collection of free and open-source software software tools for use in MRI.
Free is meant as in free beer (gratis) and freedom (libre).

To add a project edit the repos.toml file and submit a pull request.
## Stats
- Total repos: 61
- Languages:

| Language | Count |
|---|---|
| python | 43 |
| c++ | 16 |
| julia | 7 |
| c | 5 |
| javascript | 4 |
| mathematica | 1 |
| R | 1 |
| jupyter | 1 |
| r | 1 |

- Tags:

| Tag | Count |
|---|---|
| reconstruction | 17 |
| analysis | 15 |
| processing | 14 |
| simulation | 13 |
| visualisation | 7 |
| segmentation | 7 |
| fetal | 6 |
| data | 5 |
| brain | 4 |
| ML | 3 |
| qa | 3 |
| renal | 1 |
| spinal | 1 |
| muscle | 1 |

- Licenses:

| Licence | Count |
|---|---|
| MIT | 22 |
| BSD | 11 |
| Apache | 11 |
| GPLv3 | 8 |
| AGPLv3 | 2 |
| LGPLv3 | 2 |
| MPL | 2 |
| None | 2 |
| GPLv2 | 1 |



## Languages
### Python
- [pycoilgen](https://github.com/kev-m/pyCoilGen)
>- languages: ['python']
>- license: GPLv3
>- Tags: ['simulation']
Open Source Tool For Generating Coil Winding Layouts, Such As Gradient Field Coils, Within The Mri And Nmr Environments.

- [virtual-mri-scanner](https://github.com/imr-framework/virtual-scanner)
>- languages: ['python']
>- license: AGPLv3
>- Tags: ['simulation']
End-To-End Hybrid Magnetic Resonance Imaging (Mri) Simulator/Console Designed To Be Zero-Footprint, Modular, And Supported By Open-Source Standards.

- [cosimpy](https://github.com/umbertozanovello/CoSimPy)
>- languages: ['python']
>- license: MIT
>- Tags: ['simulation']
Open Source Python Library Aiming To Combine Results From Electromagnetic (Em) Simulation With Circuits Analysis Through A Cosimulation Environment.

- [vespa](https://github.com/vespa-mrs/vespa/)
>- languages: ['python']
>- license: BSD
>- Tags: ['simulation', 'data', 'processing']
Integrated, Open Source, Open Development Platform For Magnetic Resonance Spectroscopy (Mrs) Research For Rf Pulse Design, Spectral Simulation And Prototyping, Creating Synthetic Mrs Data Sets And Interactive Spectral Data Processing And Analysis.

- [scanhub](https://github.com/brain-link/scanhub)
>- languages: ['python']
>- license: GPLv3
>- Tags: ['simulation', 'reconstruction', 'processing']
Multi Modal Acquisition Software, Which Allows Individualizable, Modular And Cloud-Based Processing Of Functional And Anatomical Medical Images.

- [slicer](https://github.com/Slicer/Slicer)
>- languages: ['python', 'c++']
>- license: BSD
>- Tags: ['simulation', 'analysis', 'visualisation']
Open Source Software Package For Visualization And Image Analysis.

- [eptlib](https://github.com/eptlib/eptlib)
>- languages: ['c++', 'python']
>- license: MIT
>- Tags: ['reconstruction']
Collection Of C++ Implementations Of Electric Properties Tomography (Ept) Methods.

- [sirf](https://github.com/SyneRBI/SIRF?tab=readme-ov-file)
>- languages: ['c++', 'python']
>- license: GPLv2
>- Tags: ['reconstruction']
Open Source Toolkit For The Reconstruction Of Pet And Mri Raw Data.

- [dl-direct](https://github.com/SCAN-NRAD/DL-DiReCT)
>- languages: ['python']
>- license: BSD
>- Tags: ['segmentation']
Combines A Deep Learning-Based Neuroanatomy Segmentation And Cortex Parcellation With A Diffeomorphic Registration Technique To Measure Cortical Thickness From T1W Mri

- [dafne](https://github.com/dafne-imaging/dafne)
>- languages: ['python']
>- license: GPLv3
>- Tags: ['segmentation']
Program For The Segmentation Of Medical Images. It Relies On A Server To Provide Deep Learning Models To Aid The Segmentation, And Incremental Learning Is Used To Improve The Performance

- [ukat](https://github.com/UKRIN-MAPS/ukat)
>- languages: ['python']
>- license: GPLv3
>- Tags: ['analysis', 'renal']
Ukat Is A Vendor Agnostic Framework For The Analysis Of Quantitative Renal Mri Data

- [pypulseq](https://github.com/imr-framework/pypulseq/)
>- languages: ['python']
>- license: AGPLv3
>- Tags: ['simulation']
Enables Vendor-Neutral Pulse Sequence Design In Python [1,2]. The Pulse Sequences Can Be Exported As A .Seq File To Be Run On Siemens/Ge/Bruker Hardware By Leveraging Their Respective Pulseq Interpreters.

- [spinal-chord-toolbox](https://github.com/spinalcordtoolbox/spinalcordtoolbox)
>- languages: ['python']
>- license: LGPLv3
>- Tags: ['segmentation', 'processing', 'spinal']
Comprehensive, Free And Open-Source Set Of Command-Line Tools Dedicated To The Processing And Analysis Of Spinal Cord Mri Data

- [dwybss](https://github.com/mmromero/dwybss)
>- languages: ['python']
>- license: LGPLv3
>- Tags: ['segmentation', 'brain']
Separate Microstructure Tissue Components From The Diffusion Mri Signal, Characterize The Volume Fractions, And T2 Maps Of These Compartments

- [freesurfer](https://github.com/freesurfer/freesurfer)
>- languages: ['c++', 'c', 'python']
>- license: GPLv3
>- Tags: ['analysis', 'visualisation', 'brain']
Analysis And Visualization Of Neuroimaging Data From Cross-Sectional And Longitudinal Studies

- [simple-itk](https://github.com/SimpleITK/SimpleITK)
>- languages: ['c++', 'python', 'R']
>- license: Apache
>- Tags: ['segmentation', 'analysis']
Image Analysis Toolkit With A Large Number Of Components Supporting General Filtering Operations, Image Segmentation And Registration

- [ismrm-raw-data-format](https://ismrmrd.github.io/apidocs/1.5.0/)
>- languages: ['c', 'c++', 'python']
>- license: MIT
>- Tags: ['data']
 Common Raw Data Format, Which Attempts To Capture The Data Fields That Are Required To Describe The Magnetic Resonance Experiment With Enough Detail To Reconstruct Images

- [tensorflow-mri](https://github.com/mrphys/tensorflow-mri)
>- languages: ['python']
>- license: Apache
>- Tags: ['reconstruction', 'ML']
Library Of Tensorflow Operators For Computational Mri

- [quit](https://github.com/spinicist/QUIT)
>- languages: ['c++', 'python']
>- license: MPL
>- Tags: ['analysis']
Collection Of Programs For Processing Quantitative Mri Data

- [gropt](https://github.com/mloecher/gropt)
>- languages: ['c', 'python']
>- license: GPLv3
>- Tags: ['simulation']
 Toolbox For Mri Gradient Optimization

- [disimpy](https://github.com/kerkelae/disimpy)
>- languages: ['python']
>- license: MIT
>- Tags: ['simulation']
Python Package For Generating Simulated Diffusion-Weighted Mr Signals That Can Be Useful In The Development And Validation Of Data Acquisition And Analysis Methods

- [myoqmri](https://github.com/fsantini/MyoQMRI)
>- languages: ['python']
>- license: GPLv3
>- Tags: ['analysis', 'muscle']
Quantitative Mri Of The Muscles

- [mrqy](https://github.com/ccipd/MRQy)
>- languages: ['javascript', 'python']
>- license: BSD
>- Tags: ['qa', 'analysis']
Generate Several Tags And Noise/Information Measurements For Quality Assessment

- [nlsam](https://github.com/samuelstjean/nlsam)
>- languages: ['python']
>- license: GPLv3
>- Tags: ['reconstruction']
Implementation For The Non Local Spatial And Angular Matching (Nlsam) Denoising Algorithm For Diffusion Mri

- [mrtrix3](https://github.com/MRtrix3/mrtrix3)
>- languages: ['c++', 'python']
>- license: MPL
>- Tags: ['processing']
Set Of Tools To Perform Various Types Of Diffusion Mri Analyses, From Various Forms Of Tractography Through To Next-Generation Group-Level Analyses

- [smriprep](https://github.com/nipreps/smriprep)
>- languages: ['python']
>- license: Apache
>- Tags: ['processing']
Structural Magnetic Resonance Imaging (Smri) Data Preprocessing Pipeline That Is Designed To Provide An Easily Accessible, State-Of-The-Art Interface That Is Robust To Variations In Scan Acquisition Protocols And That Requires Minimal User Input, While Providing Easily Interpretable And Comprehensive Error And Output Reporting

- [fastmri](https://github.com/facebookresearch/fastMRI)
>- languages: ['python']
>- license: MIT
>- Tags: ['ML', 'analysis', 'data']
Research Project From Facebook Ai Research (Fair) And Nyu Langone Health To Investigate The Use Of Ai To Make Mri Scans Faster

- [flow4d](https://github.com/saitta-s/flow4D)
>- languages: ['python']
>- license: MIT
>- Tags: ['processing']
Work With 4D Flow Mri Acquisitions For Cfd Applications

- [nesvor](https://github.com/daviddmc/NeSVoR)
>- languages: ['python']
>- license: MIT
>- Tags: ['reconstruction', 'fetal']
Gpu-Accelerated Slice-To-Volume Reconstruction (Both Rigid And Deformable)

- [affirm](https://github.com/allard-shi/affirm)
>- languages: ['python']
>- license: MIT
>- Tags: ['analysis', 'fetal']
Deep Recursive Fetal Motion Estimation And Correction Based On Slice And Volume Affinity Fusion

- [niftymic](https://github.com/gift-surg/NiftyMIC)
>- languages: ['python']
>- license: BSD
>- Tags: ['reconstruction', 'fetal']
Toolkit For Research Developed Within The Gift-Surg Project To Reconstruct An Isotropic, High-Resolution Volume From Multiple, Possibly Motion-Corrupted, Stacks Of Low-Resolution 2D Slices

- [mialsuperresolutiontoolkit](https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit)
>- languages: ['c++', 'python']
>- license: BSD
>- Tags: ['reconstruction', 'fetal']
C++ And Python Tools Necessary To Perform Motion-Robust Super-Resolution Fetal Mri Reconstruction

- [fetal-iqa](https://github.com/daviddmc/fetal-IQA)
>- languages: ['python']
>- license: MIT
>- Tags: ['qa', 'fetal']
 Image Quality Assessment (Iqa) Method For Fetal Mri

- [quicknat](https://github.com/ai-med/quickNAT_pytorch)
>- languages: ['python']
>- license: MIT
>- Tags: ['segmentation', 'brain']
Fully Convolutional Network For Quick And Accurate Segmentation Of Neuroanatomy And Quality Control Of Structure-Wise Segmentations

- [pydeface](https://github.com/poldracklab/pydeface)
>- languages: ['python']
>- license: MIT
>- Tags: ['processing']
A Tool To Remove Facial Structure From Mri Images.

- [mritopng](https://github.com/danishm/mritopng)
>- languages: ['python']
>- license: MIT
>- Tags: ['processing']
A Simple Python Module To Make It Easy To Batch Convert Dicom Files To Png Images.

- [qslprep](https://github.com/PennLINC/qsiprep)
>- languages: ['python']
>- license: BSD
>- Tags: ['processing', 'reconstruction', 'analysis']
Preprocessing And Reconstruction Of Diffusion Mri

- [gif_your_nifti](https://github.com/miykael/gif_your_nifti)
>- languages: ['python']
>- license: BSD
>- Tags: ['visualisation']
Create Nice Looking Gifs From Your Nifti (.Nii Or .Nii.Gz) Files With A Simple Command

- [direct](https://github.com/NKI-AI/direct)
>- languages: ['python']
>- license: Apache
>- Tags: ['reconstruction']
Deep Learning Framework For Mri Reconstruction

- [brainchop](https://github.com/neuroneural/brainchop)
>- languages: ['javascript', 'python']
>- license: MIT
>- Tags: ['segmentation', 'visualisation']
In-Browser 3D Mri Rendering And Segmentation

- [mriqc](https://github.com/nipreps/mriqc)
>- languages: ['javascript', 'python']
>- license: Apache
>- Tags: ['qa', 'analysis']
Extracts No-Reference Iqms (Image Quality Metrics) From Structural (T1W And T2W) And Functional Mri (Magnetic Resonance Imaging) Data

- [synthseg](https://github.com/BBillot/SynthSeg)
>- languages: ['python']
>- license: Apache
>- Tags: ['ML', 'reconstruction']
Deep Learning Tool For Segmentation Of Brain Scans Of Any Contrast And Resolution

- [intensity-normalization](https://github.com/jcreinhold/intensity-normalization)
>- languages: ['python']
>- license: Apache
>- Tags: ['processing']
Various Methods To Normalize The Intensity Of Various Modalities Of Magnetic Resonance (Mr) Images, E.G., T1-Weighted (T1-W), T2-Weighted (T2-W), Fluid-Attenuated Inversion Recovery (Flair), And Proton Density-Weighted (Pd-W)

### C++
- [slicer](https://github.com/Slicer/Slicer)
>- languages: ['python', 'c++']
>- license: BSD
>- Tags: ['simulation', 'analysis', 'visualisation']
Open Source Software Package For Visualization And Image Analysis.

- [eptlib](https://github.com/eptlib/eptlib)
>- languages: ['c++', 'python']
>- license: MIT
>- Tags: ['reconstruction']
Collection Of C++ Implementations Of Electric Properties Tomography (Ept) Methods.

- [sirf](https://github.com/SyneRBI/SIRF?tab=readme-ov-file)
>- languages: ['c++', 'python']
>- license: GPLv2
>- Tags: ['reconstruction']
Open Source Toolkit For The Reconstruction Of Pet And Mri Raw Data.

- [hdr-mri](https://github.com/shakes76/sHDR)
>- languages: ['c++']
>- license: Apache
>- Tags: ['reconstruction']
Takes As Input Coregistered Mr Images (Preferrably Of Different Contrasts), Non-Linearly Combines Them And Outputs A Single Hdr Mr Image.

- [gadgetron](https://github.com/gadgetron/gadgetron)
>- languages: ['c++']
>- license: MIT
>- Tags: ['reconstruction']
Open Source Project For Medical Image Reconstruction

- [powergrid](https://github.com/mrfil/PowerGrid)
>- languages: ['c++']
>- license: MIT
>- Tags: ['reconstruction']
Cpu And Gpu Accelerated Iterative Magnetic Resonance Imaging Reconstruction

- [ukf-tractography](https://github.com/pnlbwh/ukftractography)
>- languages: ['c', 'c++']
>- license: BSD
>- Tags: ['visualisation', 'brain']
Framework Which Uses An Unscented Kalman Filter For Performing Tractography

- [freesurfer](https://github.com/freesurfer/freesurfer)
>- languages: ['c++', 'c', 'python']
>- license: GPLv3
>- Tags: ['analysis', 'visualisation', 'brain']
Analysis And Visualization Of Neuroimaging Data From Cross-Sectional And Longitudinal Studies

- [simple-itk](https://github.com/SimpleITK/SimpleITK)
>- languages: ['c++', 'python', 'R']
>- license: Apache
>- Tags: ['segmentation', 'analysis']
Image Analysis Toolkit With A Large Number Of Components Supporting General Filtering Operations, Image Segmentation And Registration

- [ismrm-raw-data-format](https://ismrmrd.github.io/apidocs/1.5.0/)
>- languages: ['c', 'c++', 'python']
>- license: MIT
>- Tags: ['data']
 Common Raw Data Format, Which Attempts To Capture The Data Fields That Are Required To Describe The Magnetic Resonance Experiment With Enough Detail To Reconstruct Images

- [quit](https://github.com/spinicist/QUIT)
>- languages: ['c++', 'python']
>- license: MPL
>- Tags: ['analysis']
Collection Of Programs For Processing Quantitative Mri Data

- [madym](https://gitlab.com/manchester_qbi/manchester_qbi_public/madym_cxx)
>- languages: ['c++']
>- license: Apache
>- Tags: ['analysis', 'processing']
C++ Toolkit For Quantative Dce-Mri And Dwi-Mri Analysis

- [bart](http://mrirecon.github.io/bart/)
>- languages: ['c', 'c++']
>- license: BSD
>- Tags: ['reconstruction']
Free And Open-Source Image-Reconstruction Framework For Computational Magnetic Resonance Imaging

- [mrtrix3](https://github.com/MRtrix3/mrtrix3)
>- languages: ['c++', 'python']
>- license: MPL
>- Tags: ['processing']
Set Of Tools To Perform Various Types Of Diffusion Mri Analyses, From Various Forms Of Tractography Through To Next-Generation Group-Level Analyses

- [svrtk](https://github.com/SVRTK/SVRTK)
>- languages: ['c++']
>- license: Apache
>- Tags: ['reconstruction', 'fetal']
Mirtk Based Svr Reconstruction

- [mialsuperresolutiontoolkit](https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit)
>- languages: ['c++', 'python']
>- license: BSD
>- Tags: ['reconstruction', 'fetal']
C++ And Python Tools Necessary To Perform Motion-Robust Super-Resolution Fetal Mri Reconstruction

### Julia
- [koma](https://github.com/JuliaHealth/KomaMRI.jl)
>- languages: ['julia']
>- license: MIT
>- Tags: ['simulation']
Pulseq-Compatible Framework To Efficiently Simulate Magnetic Resonance Imaging (Mri) Acquisitions

- [mri-generalized-bloch](https://github.com/JakobAsslaender/MRIgeneralizedBloch.jl)
>- languages: ['julia']
>- license: MIT
>- Tags: ['simulation']
Julia Package That Implements The Generalized Bloch Equations For Modeling The Dynamics Of The Semi-Solid Spin Pool In Magnetic Resonance Imaging (Mri), And Its Exchange With The Free Spin Pool

- [decaes](https://github.com/jondeuce/DECAES.jl)
>- languages: ['julia']
>- license: MIT
>- Tags: ['processing']
Julia Implementation Of The Matlab Toolbox From The Ubc Mri Research Centre For Computing Voxelwise T2-Distributions From Multi Spin-Echo Mri Images Using The Extended Phase Graph Algorithm With Stimulated Echo Corrections

- [mri-reco](https://github.com/MagneticResonanceImaging/MRIReco.jl)
>- languages: ['julia']
>- license: MIT
>- Tags: ['reconstruction']
Julia Package For Magnetic Resonance Imaging

- [mri-research-tools](https://github.com/korbinian90/MriResearchTools.jl)
>- languages: ['julia']
>- license: MIT
>- Tags: ['processing', 'analysis', 'simulation']
Specialized Tools For Mri

- [nlft](https://github.com/JuliaNeuroscience/NIfTI.jl)
>- languages: ['julia']
>- license: MIT
>- Tags: ['data']
Julia Module For Reading/Writing Nifti Mri Files

- [dcemri](https://github.com/davidssmith/DCEMRI.jl)
>- languages: ['julia']
>- license: MIT
>- Tags: ['analysis']
Open Source Toolkit For Dynamic Contrast Enhanced Mri Analysis

### C
- [ukf-tractography](https://github.com/pnlbwh/ukftractography)
>- languages: ['c', 'c++']
>- license: BSD
>- Tags: ['visualisation', 'brain']
Framework Which Uses An Unscented Kalman Filter For Performing Tractography

- [freesurfer](https://github.com/freesurfer/freesurfer)
>- languages: ['c++', 'c', 'python']
>- license: GPLv3
>- Tags: ['analysis', 'visualisation', 'brain']
Analysis And Visualization Of Neuroimaging Data From Cross-Sectional And Longitudinal Studies

- [ismrm-raw-data-format](https://ismrmrd.github.io/apidocs/1.5.0/)
>- languages: ['c', 'c++', 'python']
>- license: MIT
>- Tags: ['data']
 Common Raw Data Format, Which Attempts To Capture The Data Fields That Are Required To Describe The Magnetic Resonance Experiment With Enough Detail To Reconstruct Images

- [gropt](https://github.com/mloecher/gropt)
>- languages: ['c', 'python']
>- license: GPLv3
>- Tags: ['simulation']
 Toolbox For Mri Gradient Optimization

- [bart](http://mrirecon.github.io/bart/)
>- languages: ['c', 'c++']
>- license: BSD
>- Tags: ['reconstruction']
Free And Open-Source Image-Reconstruction Framework For Computational Magnetic Resonance Imaging

### Javascript
- [mrqy](https://github.com/ccipd/MRQy)
>- languages: ['javascript', 'python']
>- license: BSD
>- Tags: ['qa', 'analysis']
Generate Several Tags And Noise/Information Measurements For Quality Assessment

- [brainchop](https://github.com/neuroneural/brainchop)
>- languages: ['javascript', 'python']
>- license: MIT
>- Tags: ['segmentation', 'visualisation']
In-Browser 3D Mri Rendering And Segmentation

- [mriqc](https://github.com/nipreps/mriqc)
>- languages: ['javascript', 'python']
>- license: Apache
>- Tags: ['qa', 'analysis']
Extracts No-Reference Iqms (Image Quality Metrics) From Structural (T1W And T2W) And Functional Mri (Magnetic Resonance Imaging) Data

- [mri-viewer](https://github.com/epam/mriviewer)
>- languages: ['javascript']
>- license: Apache
>- Tags: ['visualisation']
High Performance Web Tool For Advanced Visualization (Both In 2D And 3D Modes) Medical Volumetric Data, Provided In Popular File Formats: Dicom, Nifti, Ktx, Hdr

### Mathematica
- [qmritools](https://github.com/mfroeling/QMRITools)
>- languages: ['mathematica']
>- license: BSD
>- Tags: ['simulation', 'processing', 'visualisation']
Collection Of Tools And Functions For Processing Quantitative Mri Data.

### R
- [simple-itk](https://github.com/SimpleITK/SimpleITK)
>- languages: ['c++', 'python', 'R']
>- license: Apache
>- Tags: ['segmentation', 'analysis']
Image Analysis Toolkit With A Large Number Of Components Supporting General Filtering Operations, Image Segmentation And Registration

### Jupyter
- [openmorph](https://github.com/cMadan/openMorph)
>- languages: ['jupyter']
>- license: None
>- Tags: ['data']
Curated List Of Open-Access Databases With Human Structural Mri Data

### R
- [braingraph](https://github.com/cwatson/brainGraph)
>- languages: ['r']
>- license: None
>- Tags: ['analysis']
R Package For Performing Graph Theory Analyses Of Brain Mri Data



## Tags
### Reconstruction
- [scanhub](https://github.com/brain-link/scanhub)
>- languages: ['python']
>- license: GPLv3
>- Tags: ['simulation', 'reconstruction', 'processing']
Multi Modal Acquisition Software, Which Allows Individualizable, Modular And Cloud-Based Processing Of Functional And Anatomical Medical Images.

- [eptlib](https://github.com/eptlib/eptlib)
>- languages: ['c++', 'python']
>- license: MIT
>- Tags: ['reconstruction']
Collection Of C++ Implementations Of Electric Properties Tomography (Ept) Methods.

- [sirf](https://github.com/SyneRBI/SIRF?tab=readme-ov-file)
>- languages: ['c++', 'python']
>- license: GPLv2
>- Tags: ['reconstruction']
Open Source Toolkit For The Reconstruction Of Pet And Mri Raw Data.

- [hdr-mri](https://github.com/shakes76/sHDR)
>- languages: ['c++']
>- license: Apache
>- Tags: ['reconstruction']
Takes As Input Coregistered Mr Images (Preferrably Of Different Contrasts), Non-Linearly Combines Them And Outputs A Single Hdr Mr Image.

- [gadgetron](https://github.com/gadgetron/gadgetron)
>- languages: ['c++']
>- license: MIT
>- Tags: ['reconstruction']
Open Source Project For Medical Image Reconstruction

- [powergrid](https://github.com/mrfil/PowerGrid)
>- languages: ['c++']
>- license: MIT
>- Tags: ['reconstruction']
Cpu And Gpu Accelerated Iterative Magnetic Resonance Imaging Reconstruction

- [tensorflow-mri](https://github.com/mrphys/tensorflow-mri)
>- languages: ['python']
>- license: Apache
>- Tags: ['reconstruction', 'ML']
Library Of Tensorflow Operators For Computational Mri

- [mri-reco](https://github.com/MagneticResonanceImaging/MRIReco.jl)
>- languages: ['julia']
>- license: MIT
>- Tags: ['reconstruction']
Julia Package For Magnetic Resonance Imaging

- [nlsam](https://github.com/samuelstjean/nlsam)
>- languages: ['python']
>- license: GPLv3
>- Tags: ['reconstruction']
Implementation For The Non Local Spatial And Angular Matching (Nlsam) Denoising Algorithm For Diffusion Mri

- [bart](http://mrirecon.github.io/bart/)
>- languages: ['c', 'c++']
>- license: BSD
>- Tags: ['reconstruction']
Free And Open-Source Image-Reconstruction Framework For Computational Magnetic Resonance Imaging

- [nesvor](https://github.com/daviddmc/NeSVoR)
>- languages: ['python']
>- license: MIT
>- Tags: ['reconstruction', 'fetal']
Gpu-Accelerated Slice-To-Volume Reconstruction (Both Rigid And Deformable)

- [niftymic](https://github.com/gift-surg/NiftyMIC)
>- languages: ['python']
>- license: BSD
>- Tags: ['reconstruction', 'fetal']
Toolkit For Research Developed Within The Gift-Surg Project To Reconstruct An Isotropic, High-Resolution Volume From Multiple, Possibly Motion-Corrupted, Stacks Of Low-Resolution 2D Slices

- [svrtk](https://github.com/SVRTK/SVRTK)
>- languages: ['c++']
>- license: Apache
>- Tags: ['reconstruction', 'fetal']
Mirtk Based Svr Reconstruction

- [mialsuperresolutiontoolkit](https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit)
>- languages: ['c++', 'python']
>- license: BSD
>- Tags: ['reconstruction', 'fetal']
C++ And Python Tools Necessary To Perform Motion-Robust Super-Resolution Fetal Mri Reconstruction

- [qslprep](https://github.com/PennLINC/qsiprep)
>- languages: ['python']
>- license: BSD
>- Tags: ['processing', 'reconstruction', 'analysis']
Preprocessing And Reconstruction Of Diffusion Mri

- [direct](https://github.com/NKI-AI/direct)
>- languages: ['python']
>- license: Apache
>- Tags: ['reconstruction']
Deep Learning Framework For Mri Reconstruction

- [synthseg](https://github.com/BBillot/SynthSeg)
>- languages: ['python']
>- license: Apache
>- Tags: ['ML', 'reconstruction']
Deep Learning Tool For Segmentation Of Brain Scans Of Any Contrast And Resolution

### Analysis
- [slicer](https://github.com/Slicer/Slicer)
>- languages: ['python', 'c++']
>- license: BSD
>- Tags: ['simulation', 'analysis', 'visualisation']
Open Source Software Package For Visualization And Image Analysis.

- [ukat](https://github.com/UKRIN-MAPS/ukat)
>- languages: ['python']
>- license: GPLv3
>- Tags: ['analysis', 'renal']
Ukat Is A Vendor Agnostic Framework For The Analysis Of Quantitative Renal Mri Data

- [freesurfer](https://github.com/freesurfer/freesurfer)
>- languages: ['c++', 'c', 'python']
>- license: GPLv3
>- Tags: ['analysis', 'visualisation', 'brain']
Analysis And Visualization Of Neuroimaging Data From Cross-Sectional And Longitudinal Studies

- [simple-itk](https://github.com/SimpleITK/SimpleITK)
>- languages: ['c++', 'python', 'R']
>- license: Apache
>- Tags: ['segmentation', 'analysis']
Image Analysis Toolkit With A Large Number Of Components Supporting General Filtering Operations, Image Segmentation And Registration

- [quit](https://github.com/spinicist/QUIT)
>- languages: ['c++', 'python']
>- license: MPL
>- Tags: ['analysis']
Collection Of Programs For Processing Quantitative Mri Data

- [madym](https://gitlab.com/manchester_qbi/manchester_qbi_public/madym_cxx)
>- languages: ['c++']
>- license: Apache
>- Tags: ['analysis', 'processing']
C++ Toolkit For Quantative Dce-Mri And Dwi-Mri Analysis

- [myoqmri](https://github.com/fsantini/MyoQMRI)
>- languages: ['python']
>- license: GPLv3
>- Tags: ['analysis', 'muscle']
Quantitative Mri Of The Muscles

- [mrqy](https://github.com/ccipd/MRQy)
>- languages: ['javascript', 'python']
>- license: BSD
>- Tags: ['qa', 'analysis']
Generate Several Tags And Noise/Information Measurements For Quality Assessment

- [fastmri](https://github.com/facebookresearch/fastMRI)
>- languages: ['python']
>- license: MIT
>- Tags: ['ML', 'analysis', 'data']
Research Project From Facebook Ai Research (Fair) And Nyu Langone Health To Investigate The Use Of Ai To Make Mri Scans Faster

- [affirm](https://github.com/allard-shi/affirm)
>- languages: ['python']
>- license: MIT
>- Tags: ['analysis', 'fetal']
Deep Recursive Fetal Motion Estimation And Correction Based On Slice And Volume Affinity Fusion

- [mri-research-tools](https://github.com/korbinian90/MriResearchTools.jl)
>- languages: ['julia']
>- license: MIT
>- Tags: ['processing', 'analysis', 'simulation']
Specialized Tools For Mri

- [dcemri](https://github.com/davidssmith/DCEMRI.jl)
>- languages: ['julia']
>- license: MIT
>- Tags: ['analysis']
Open Source Toolkit For Dynamic Contrast Enhanced Mri Analysis

- [qslprep](https://github.com/PennLINC/qsiprep)
>- languages: ['python']
>- license: BSD
>- Tags: ['processing', 'reconstruction', 'analysis']
Preprocessing And Reconstruction Of Diffusion Mri

- [braingraph](https://github.com/cwatson/brainGraph)
>- languages: ['r']
>- license: None
>- Tags: ['analysis']
R Package For Performing Graph Theory Analyses Of Brain Mri Data

- [mriqc](https://github.com/nipreps/mriqc)
>- languages: ['javascript', 'python']
>- license: Apache
>- Tags: ['qa', 'analysis']
Extracts No-Reference Iqms (Image Quality Metrics) From Structural (T1W And T2W) And Functional Mri (Magnetic Resonance Imaging) Data

### Processing
- [vespa](https://github.com/vespa-mrs/vespa/)
>- languages: ['python']
>- license: BSD
>- Tags: ['simulation', 'data', 'processing']
Integrated, Open Source, Open Development Platform For Magnetic Resonance Spectroscopy (Mrs) Research For Rf Pulse Design, Spectral Simulation And Prototyping, Creating Synthetic Mrs Data Sets And Interactive Spectral Data Processing And Analysis.

- [scanhub](https://github.com/brain-link/scanhub)
>- languages: ['python']
>- license: GPLv3
>- Tags: ['simulation', 'reconstruction', 'processing']
Multi Modal Acquisition Software, Which Allows Individualizable, Modular And Cloud-Based Processing Of Functional And Anatomical Medical Images.

- [qmritools](https://github.com/mfroeling/QMRITools)
>- languages: ['mathematica']
>- license: BSD
>- Tags: ['simulation', 'processing', 'visualisation']
Collection Of Tools And Functions For Processing Quantitative Mri Data.

- [spinal-chord-toolbox](https://github.com/spinalcordtoolbox/spinalcordtoolbox)
>- languages: ['python']
>- license: LGPLv3
>- Tags: ['segmentation', 'processing', 'spinal']
Comprehensive, Free And Open-Source Set Of Command-Line Tools Dedicated To The Processing And Analysis Of Spinal Cord Mri Data

- [madym](https://gitlab.com/manchester_qbi/manchester_qbi_public/madym_cxx)
>- languages: ['c++']
>- license: Apache
>- Tags: ['analysis', 'processing']
C++ Toolkit For Quantative Dce-Mri And Dwi-Mri Analysis

- [decaes](https://github.com/jondeuce/DECAES.jl)
>- languages: ['julia']
>- license: MIT
>- Tags: ['processing']
Julia Implementation Of The Matlab Toolbox From The Ubc Mri Research Centre For Computing Voxelwise T2-Distributions From Multi Spin-Echo Mri Images Using The Extended Phase Graph Algorithm With Stimulated Echo Corrections

- [mrtrix3](https://github.com/MRtrix3/mrtrix3)
>- languages: ['c++', 'python']
>- license: MPL
>- Tags: ['processing']
Set Of Tools To Perform Various Types Of Diffusion Mri Analyses, From Various Forms Of Tractography Through To Next-Generation Group-Level Analyses

- [smriprep](https://github.com/nipreps/smriprep)
>- languages: ['python']
>- license: Apache
>- Tags: ['processing']
Structural Magnetic Resonance Imaging (Smri) Data Preprocessing Pipeline That Is Designed To Provide An Easily Accessible, State-Of-The-Art Interface That Is Robust To Variations In Scan Acquisition Protocols And That Requires Minimal User Input, While Providing Easily Interpretable And Comprehensive Error And Output Reporting

- [flow4d](https://github.com/saitta-s/flow4D)
>- languages: ['python']
>- license: MIT
>- Tags: ['processing']
Work With 4D Flow Mri Acquisitions For Cfd Applications

- [mri-research-tools](https://github.com/korbinian90/MriResearchTools.jl)
>- languages: ['julia']
>- license: MIT
>- Tags: ['processing', 'analysis', 'simulation']
Specialized Tools For Mri

- [pydeface](https://github.com/poldracklab/pydeface)
>- languages: ['python']
>- license: MIT
>- Tags: ['processing']
A Tool To Remove Facial Structure From Mri Images.

- [mritopng](https://github.com/danishm/mritopng)
>- languages: ['python']
>- license: MIT
>- Tags: ['processing']
A Simple Python Module To Make It Easy To Batch Convert Dicom Files To Png Images.

- [qslprep](https://github.com/PennLINC/qsiprep)
>- languages: ['python']
>- license: BSD
>- Tags: ['processing', 'reconstruction', 'analysis']
Preprocessing And Reconstruction Of Diffusion Mri

- [intensity-normalization](https://github.com/jcreinhold/intensity-normalization)
>- languages: ['python']
>- license: Apache
>- Tags: ['processing']
Various Methods To Normalize The Intensity Of Various Modalities Of Magnetic Resonance (Mr) Images, E.G., T1-Weighted (T1-W), T2-Weighted (T2-W), Fluid-Attenuated Inversion Recovery (Flair), And Proton Density-Weighted (Pd-W)

### Simulation
- [pycoilgen](https://github.com/kev-m/pyCoilGen)
>- languages: ['python']
>- license: GPLv3
>- Tags: ['simulation']
Open Source Tool For Generating Coil Winding Layouts, Such As Gradient Field Coils, Within The Mri And Nmr Environments.

- [virtual-mri-scanner](https://github.com/imr-framework/virtual-scanner)
>- languages: ['python']
>- license: AGPLv3
>- Tags: ['simulation']
End-To-End Hybrid Magnetic Resonance Imaging (Mri) Simulator/Console Designed To Be Zero-Footprint, Modular, And Supported By Open-Source Standards.

- [cosimpy](https://github.com/umbertozanovello/CoSimPy)
>- languages: ['python']
>- license: MIT
>- Tags: ['simulation']
Open Source Python Library Aiming To Combine Results From Electromagnetic (Em) Simulation With Circuits Analysis Through A Cosimulation Environment.

- [vespa](https://github.com/vespa-mrs/vespa/)
>- languages: ['python']
>- license: BSD
>- Tags: ['simulation', 'data', 'processing']
Integrated, Open Source, Open Development Platform For Magnetic Resonance Spectroscopy (Mrs) Research For Rf Pulse Design, Spectral Simulation And Prototyping, Creating Synthetic Mrs Data Sets And Interactive Spectral Data Processing And Analysis.

- [scanhub](https://github.com/brain-link/scanhub)
>- languages: ['python']
>- license: GPLv3
>- Tags: ['simulation', 'reconstruction', 'processing']
Multi Modal Acquisition Software, Which Allows Individualizable, Modular And Cloud-Based Processing Of Functional And Anatomical Medical Images.

- [slicer](https://github.com/Slicer/Slicer)
>- languages: ['python', 'c++']
>- license: BSD
>- Tags: ['simulation', 'analysis', 'visualisation']
Open Source Software Package For Visualization And Image Analysis.

- [qmritools](https://github.com/mfroeling/QMRITools)
>- languages: ['mathematica']
>- license: BSD
>- Tags: ['simulation', 'processing', 'visualisation']
Collection Of Tools And Functions For Processing Quantitative Mri Data.

- [pypulseq](https://github.com/imr-framework/pypulseq/)
>- languages: ['python']
>- license: AGPLv3
>- Tags: ['simulation']
Enables Vendor-Neutral Pulse Sequence Design In Python [1,2]. The Pulse Sequences Can Be Exported As A .Seq File To Be Run On Siemens/Ge/Bruker Hardware By Leveraging Their Respective Pulseq Interpreters.

- [koma](https://github.com/JuliaHealth/KomaMRI.jl)
>- languages: ['julia']
>- license: MIT
>- Tags: ['simulation']
Pulseq-Compatible Framework To Efficiently Simulate Magnetic Resonance Imaging (Mri) Acquisitions

- [gropt](https://github.com/mloecher/gropt)
>- languages: ['c', 'python']
>- license: GPLv3
>- Tags: ['simulation']
 Toolbox For Mri Gradient Optimization

- [disimpy](https://github.com/kerkelae/disimpy)
>- languages: ['python']
>- license: MIT
>- Tags: ['simulation']
Python Package For Generating Simulated Diffusion-Weighted Mr Signals That Can Be Useful In The Development And Validation Of Data Acquisition And Analysis Methods

- [mri-generalized-bloch](https://github.com/JakobAsslaender/MRIgeneralizedBloch.jl)
>- languages: ['julia']
>- license: MIT
>- Tags: ['simulation']
Julia Package That Implements The Generalized Bloch Equations For Modeling The Dynamics Of The Semi-Solid Spin Pool In Magnetic Resonance Imaging (Mri), And Its Exchange With The Free Spin Pool

- [mri-research-tools](https://github.com/korbinian90/MriResearchTools.jl)
>- languages: ['julia']
>- license: MIT
>- Tags: ['processing', 'analysis', 'simulation']
Specialized Tools For Mri

### Visualisation
- [slicer](https://github.com/Slicer/Slicer)
>- languages: ['python', 'c++']
>- license: BSD
>- Tags: ['simulation', 'analysis', 'visualisation']
Open Source Software Package For Visualization And Image Analysis.

- [qmritools](https://github.com/mfroeling/QMRITools)
>- languages: ['mathematica']
>- license: BSD
>- Tags: ['simulation', 'processing', 'visualisation']
Collection Of Tools And Functions For Processing Quantitative Mri Data.

- [ukf-tractography](https://github.com/pnlbwh/ukftractography)
>- languages: ['c', 'c++']
>- license: BSD
>- Tags: ['visualisation', 'brain']
Framework Which Uses An Unscented Kalman Filter For Performing Tractography

- [freesurfer](https://github.com/freesurfer/freesurfer)
>- languages: ['c++', 'c', 'python']
>- license: GPLv3
>- Tags: ['analysis', 'visualisation', 'brain']
Analysis And Visualization Of Neuroimaging Data From Cross-Sectional And Longitudinal Studies

- [gif_your_nifti](https://github.com/miykael/gif_your_nifti)
>- languages: ['python']
>- license: BSD
>- Tags: ['visualisation']
Create Nice Looking Gifs From Your Nifti (.Nii Or .Nii.Gz) Files With A Simple Command

- [brainchop](https://github.com/neuroneural/brainchop)
>- languages: ['javascript', 'python']
>- license: MIT
>- Tags: ['segmentation', 'visualisation']
In-Browser 3D Mri Rendering And Segmentation

- [mri-viewer](https://github.com/epam/mriviewer)
>- languages: ['javascript']
>- license: Apache
>- Tags: ['visualisation']
High Performance Web Tool For Advanced Visualization (Both In 2D And 3D Modes) Medical Volumetric Data, Provided In Popular File Formats: Dicom, Nifti, Ktx, Hdr

### Segmentation
- [dl-direct](https://github.com/SCAN-NRAD/DL-DiReCT)
>- languages: ['python']
>- license: BSD
>- Tags: ['segmentation']
Combines A Deep Learning-Based Neuroanatomy Segmentation And Cortex Parcellation With A Diffeomorphic Registration Technique To Measure Cortical Thickness From T1W Mri

- [dafne](https://github.com/dafne-imaging/dafne)
>- languages: ['python']
>- license: GPLv3
>- Tags: ['segmentation']
Program For The Segmentation Of Medical Images. It Relies On A Server To Provide Deep Learning Models To Aid The Segmentation, And Incremental Learning Is Used To Improve The Performance

- [spinal-chord-toolbox](https://github.com/spinalcordtoolbox/spinalcordtoolbox)
>- languages: ['python']
>- license: LGPLv3
>- Tags: ['segmentation', 'processing', 'spinal']
Comprehensive, Free And Open-Source Set Of Command-Line Tools Dedicated To The Processing And Analysis Of Spinal Cord Mri Data

- [dwybss](https://github.com/mmromero/dwybss)
>- languages: ['python']
>- license: LGPLv3
>- Tags: ['segmentation', 'brain']
Separate Microstructure Tissue Components From The Diffusion Mri Signal, Characterize The Volume Fractions, And T2 Maps Of These Compartments

- [simple-itk](https://github.com/SimpleITK/SimpleITK)
>- languages: ['c++', 'python', 'R']
>- license: Apache
>- Tags: ['segmentation', 'analysis']
Image Analysis Toolkit With A Large Number Of Components Supporting General Filtering Operations, Image Segmentation And Registration

- [quicknat](https://github.com/ai-med/quickNAT_pytorch)
>- languages: ['python']
>- license: MIT
>- Tags: ['segmentation', 'brain']
Fully Convolutional Network For Quick And Accurate Segmentation Of Neuroanatomy And Quality Control Of Structure-Wise Segmentations

- [brainchop](https://github.com/neuroneural/brainchop)
>- languages: ['javascript', 'python']
>- license: MIT
>- Tags: ['segmentation', 'visualisation']
In-Browser 3D Mri Rendering And Segmentation

### Fetal
- [nesvor](https://github.com/daviddmc/NeSVoR)
>- languages: ['python']
>- license: MIT
>- Tags: ['reconstruction', 'fetal']
Gpu-Accelerated Slice-To-Volume Reconstruction (Both Rigid And Deformable)

- [affirm](https://github.com/allard-shi/affirm)
>- languages: ['python']
>- license: MIT
>- Tags: ['analysis', 'fetal']
Deep Recursive Fetal Motion Estimation And Correction Based On Slice And Volume Affinity Fusion

- [niftymic](https://github.com/gift-surg/NiftyMIC)
>- languages: ['python']
>- license: BSD
>- Tags: ['reconstruction', 'fetal']
Toolkit For Research Developed Within The Gift-Surg Project To Reconstruct An Isotropic, High-Resolution Volume From Multiple, Possibly Motion-Corrupted, Stacks Of Low-Resolution 2D Slices

- [svrtk](https://github.com/SVRTK/SVRTK)
>- languages: ['c++']
>- license: Apache
>- Tags: ['reconstruction', 'fetal']
Mirtk Based Svr Reconstruction

- [mialsuperresolutiontoolkit](https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit)
>- languages: ['c++', 'python']
>- license: BSD
>- Tags: ['reconstruction', 'fetal']
C++ And Python Tools Necessary To Perform Motion-Robust Super-Resolution Fetal Mri Reconstruction

- [fetal-iqa](https://github.com/daviddmc/fetal-IQA)
>- languages: ['python']
>- license: MIT
>- Tags: ['qa', 'fetal']
 Image Quality Assessment (Iqa) Method For Fetal Mri

### Data
- [vespa](https://github.com/vespa-mrs/vespa/)
>- languages: ['python']
>- license: BSD
>- Tags: ['simulation', 'data', 'processing']
Integrated, Open Source, Open Development Platform For Magnetic Resonance Spectroscopy (Mrs) Research For Rf Pulse Design, Spectral Simulation And Prototyping, Creating Synthetic Mrs Data Sets And Interactive Spectral Data Processing And Analysis.

- [ismrm-raw-data-format](https://ismrmrd.github.io/apidocs/1.5.0/)
>- languages: ['c', 'c++', 'python']
>- license: MIT
>- Tags: ['data']
 Common Raw Data Format, Which Attempts To Capture The Data Fields That Are Required To Describe The Magnetic Resonance Experiment With Enough Detail To Reconstruct Images

- [fastmri](https://github.com/facebookresearch/fastMRI)
>- languages: ['python']
>- license: MIT
>- Tags: ['ML', 'analysis', 'data']
Research Project From Facebook Ai Research (Fair) And Nyu Langone Health To Investigate The Use Of Ai To Make Mri Scans Faster

- [nlft](https://github.com/JuliaNeuroscience/NIfTI.jl)
>- languages: ['julia']
>- license: MIT
>- Tags: ['data']
Julia Module For Reading/Writing Nifti Mri Files

- [openmorph](https://github.com/cMadan/openMorph)
>- languages: ['jupyter']
>- license: None
>- Tags: ['data']
Curated List Of Open-Access Databases With Human Structural Mri Data

### Brain
- [ukf-tractography](https://github.com/pnlbwh/ukftractography)
>- languages: ['c', 'c++']
>- license: BSD
>- Tags: ['visualisation', 'brain']
Framework Which Uses An Unscented Kalman Filter For Performing Tractography

- [dwybss](https://github.com/mmromero/dwybss)
>- languages: ['python']
>- license: LGPLv3
>- Tags: ['segmentation', 'brain']
Separate Microstructure Tissue Components From The Diffusion Mri Signal, Characterize The Volume Fractions, And T2 Maps Of These Compartments

- [freesurfer](https://github.com/freesurfer/freesurfer)
>- languages: ['c++', 'c', 'python']
>- license: GPLv3
>- Tags: ['analysis', 'visualisation', 'brain']
Analysis And Visualization Of Neuroimaging Data From Cross-Sectional And Longitudinal Studies

- [quicknat](https://github.com/ai-med/quickNAT_pytorch)
>- languages: ['python']
>- license: MIT
>- Tags: ['segmentation', 'brain']
Fully Convolutional Network For Quick And Accurate Segmentation Of Neuroanatomy And Quality Control Of Structure-Wise Segmentations

### Ml
- [tensorflow-mri](https://github.com/mrphys/tensorflow-mri)
>- languages: ['python']
>- license: Apache
>- Tags: ['reconstruction', 'ML']
Library Of Tensorflow Operators For Computational Mri

- [fastmri](https://github.com/facebookresearch/fastMRI)
>- languages: ['python']
>- license: MIT
>- Tags: ['ML', 'analysis', 'data']
Research Project From Facebook Ai Research (Fair) And Nyu Langone Health To Investigate The Use Of Ai To Make Mri Scans Faster

- [synthseg](https://github.com/BBillot/SynthSeg)
>- languages: ['python']
>- license: Apache
>- Tags: ['ML', 'reconstruction']
Deep Learning Tool For Segmentation Of Brain Scans Of Any Contrast And Resolution

### Qa
- [mrqy](https://github.com/ccipd/MRQy)
>- languages: ['javascript', 'python']
>- license: BSD
>- Tags: ['qa', 'analysis']
Generate Several Tags And Noise/Information Measurements For Quality Assessment

- [fetal-iqa](https://github.com/daviddmc/fetal-IQA)
>- languages: ['python']
>- license: MIT
>- Tags: ['qa', 'fetal']
 Image Quality Assessment (Iqa) Method For Fetal Mri

- [mriqc](https://github.com/nipreps/mriqc)
>- languages: ['javascript', 'python']
>- license: Apache
>- Tags: ['qa', 'analysis']
Extracts No-Reference Iqms (Image Quality Metrics) From Structural (T1W And T2W) And Functional Mri (Magnetic Resonance Imaging) Data

### Renal
- [ukat](https://github.com/UKRIN-MAPS/ukat)
>- languages: ['python']
>- license: GPLv3
>- Tags: ['analysis', 'renal']
Ukat Is A Vendor Agnostic Framework For The Analysis Of Quantitative Renal Mri Data

### Spinal
- [spinal-chord-toolbox](https://github.com/spinalcordtoolbox/spinalcordtoolbox)
>- languages: ['python']
>- license: LGPLv3
>- Tags: ['segmentation', 'processing', 'spinal']
Comprehensive, Free And Open-Source Set Of Command-Line Tools Dedicated To The Processing And Analysis Of Spinal Cord Mri Data

### Muscle
- [myoqmri](https://github.com/fsantini/MyoQMRI)
>- languages: ['python']
>- license: GPLv3
>- Tags: ['analysis', 'muscle']
Quantitative Mri Of The Muscles

