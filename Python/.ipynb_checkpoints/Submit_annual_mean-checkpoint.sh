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


qsub -v indir=/g/data/ik11/outputs/mom6-panan/panant-0025-zstar-ACCESSyr2/output000 separate_UVT_panan0025.sh
qsub -v indir=/g/data/ik11/outputs/mom6-panan/panant-0025-zstar-ACCESSyr2/output001 separate_UVT_panan0025.sh
qsub -v indir=/g/data/ik11/outputs/mom6-panan/panant-0025-zstar-ACCESSyr2/output002 separate_UVT_panan0025.sh
qsub -v indir=/g/data/ik11/outputs/mom6-panan/panant-0025-zstar-ACCESSyr2/output003 separate_UVT_panan0025.sh
qsub -v indir=/g/data/ik11/outputs/mom6-panan/panant-0025-zstar-ACCESSyr2/output004 separate_UVT_panan0025.sh
qsub -v indir=/g/data/ik11/outputs/mom6-panan/panant-0025-zstar-ACCESSyr2/output005 separate_UVT_panan0025.sh
qsub -v indir=/g/data/ik11/outputs/mom6-panan/panant-0025-zstar-ACCESSyr2/output006 separate_UVT_panan0025.sh
qsub -v indir=/g/data/ik11/outputs/mom6-panan/panant-0025-zstar-ACCESSyr2/output007 separate_UVT_panan0025.sh
qsub -v indir=/g/data/ik11/outputs/mom6-panan/panant-0025-zstar-ACCESSyr2/output008 separate_UVT_panan0025.sh
qsub -v indir=/g/data/ik11/outputs/mom6-panan/panant-0025-zstar-ACCESSyr2/output009 separate_UVT_panan0025.sh
qsub -v indir=/g/data/ik11/outputs/mom6-panan/panant-0025-zstar-ACCESSyr2/output010 separate_UVT_panan0025.sh
qsub -v indir=/g/data/ik11/outputs/mom6-panan/panant-0025-zstar-ACCESSyr2/output011 separate_UVT_panan0025.sh