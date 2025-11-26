#!/bin/bash
#PBS -N ztosigma2
#PBS -P g40
#PBS -q normalbw
#PBS -l walltime=4:30:00
#PBS -l mem=256GB
#PBS -l software=netcdf
#PBS -l ncpus=56
#PBS -l storage=gdata/v45+gdata/hh5+gdata/cj50+gdata/ik11+scratch/x77+scratch/v45+gdata/x77+gdata/nm03+scratch/nm03+gdata/e14+scratch/e14
#PBS -v month,year

#Calculates the monthly Along and cross isobath (1000m) speed monthly, for the specified year
## Note, run this with:
## qsub -v month=1,year=2000 ztorho_monthly.sh
## For running the along and cross slope transports, you will need more memory than 256 GB, Id say at least 1000GB

script_dir=/home/156/wf4500/v45_wf4500/Project_panan/GH/Panan_HT_ASC/Python/
cd $script_dir

# load conda
module use /g/data/hh5/public/modules
module unload conda
module load conda/analysis3-unstable

module list

# run
python3 HT_ztosigma.py $month $year &>> ztosigma_${year}_${month}.out



