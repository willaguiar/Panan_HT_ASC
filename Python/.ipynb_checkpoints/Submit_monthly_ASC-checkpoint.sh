#!/bin/bash
#PBS -N ASC_panan0025
#PBS -P ol01
#PBS -q hugemem
#PBS -l walltime=1:00:00
#PBS -l mem=1470GB
#PBS -l software=netcdf
#PBS -l ncpus=48
#PBS -l storage=gdata/v45+gdata/hh5+gdata/cj50+gdata/ik11+scratch/x77+scratch/v45+gdata/x77+gdata/nm03+scratch/nm03+scratch/oz91+gdata/oz91+scratch/ol01+gdata/ol01+scratch/e14+gdata/e14+gdata/xp65+scratch/xp65+gdata/vk83+gdata/cj50
#PBS -v month,year

#Calculates the monthly Along and cross isobath (1000m) speed monthly, for the specified year
## Note, run this with:
## qsub -v month=2,year=2000 Submit_monthly_ASC.sh
## For running the along and cross slope transports, you will need more memory than 256 GB, Id say at least 1000GB

script_dir=/g/data/ik11/users/wf4500/Project_panan/GH/Panan_HT_ASC/Python/
cd $script_dir

# load conda
module unload conda
module use /g/data/xp65/public/modules/
module load conda/analysis3-25.09

module list

# run
python3 Panan_ASC_along_isobath_coordinate_system.py $month $year &>> ASC_${year}_${month}.out



