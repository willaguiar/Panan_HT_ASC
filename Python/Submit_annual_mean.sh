#!/bin/bash

script_dir=/home/156/wf4500/v45_wf4500/Project_panan/GH/Panan_HT_ASC/Python/
cd $script_dir

# load conda
module use /g/data/hh5/public/modules
module unload conda
module load conda/analysis3-22.04
module load nco
module load cdo

module list


# THis will be submitted in parts for each year, always ending in the output of december, which would be either "011","23","35","47","59","71","83","95","107","119"


script_dir=/home/156/wf4500/v45_wf4500/Project_panan/GH/Panan_HT_ASC/Python/
cd $script_dir


qsub -v indir=/g/data/ik11/outputs/mom6-panan/panant-0025-zstar-ACCESSyr2/output024 separate_UVT_panan0025.sh
qsub -v indir=/g/data/ik11/outputs/mom6-panan/panant-0025-zstar-ACCESSyr2/output025 separate_UVT_panan0025.sh
qsub -v indir=/g/data/ik11/outputs/mom6-panan/panant-0025-zstar-ACCESSyr2/output026 separate_UVT_panan0025.sh
qsub -v indir=/g/data/ik11/outputs/mom6-panan/panant-0025-zstar-ACCESSyr2/output027 separate_UVT_panan0025.sh
qsub -v indir=/g/data/ik11/outputs/mom6-panan/panant-0025-zstar-ACCESSyr2/output028 separate_UVT_panan0025.sh
qsub -v indir=/g/data/ik11/outputs/mom6-panan/panant-0025-zstar-ACCESSyr2/output029 separate_UVT_panan0025.sh
qsub -v indir=/g/data/ik11/outputs/mom6-panan/panant-0025-zstar-ACCESSyr2/output030 separate_UVT_panan0025.sh
qsub -v indir=/g/data/ik11/outputs/mom6-panan/panant-0025-zstar-ACCESSyr2/output031 separate_UVT_panan0025.sh
qsub -v indir=/g/data/ik11/outputs/mom6-panan/panant-0025-zstar-ACCESSyr2/output032 separate_UVT_panan0025.sh
qsub -v indir=/g/data/ik11/outputs/mom6-panan/panant-0025-zstar-ACCESSyr2/output033 separate_UVT_panan0025.sh
qsub -v indir=/g/data/ik11/outputs/mom6-panan/panant-0025-zstar-ACCESSyr2/output034 separate_UVT_panan0025.sh
qsub -v indir=/g/data/ik11/outputs/mom6-panan/panant-0025-zstar-ACCESSyr2/output035 separate_UVT_panan0025.sh
