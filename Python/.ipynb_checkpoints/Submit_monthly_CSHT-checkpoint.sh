#!/bin/bash
#PBS -N ST_panan01
#PBS -P e14
#PBS -q normal
#PBS -l walltime=6:30:00
#PBS -l mem=190GB
#PBS -l software=netcdf
#PBS -l ncpus=48
#PBS -l storage=gdata/v45+gdata/hh5+gdata/cj50+gdata/ik11+scratch/x77+scratch/v45+gdata/x77+gdata/nm03+scratch/nm03+scratch/oz91+gdata/oz91+scratch/ol01+gdata/ol01+scratch/e14+gdata/e14
#PBS -v month,year

#Calculates the monthly Along and cross isobath (1000m) speed monthly, for the specified year
## Note, run this with:
## qsub -v month=8,year=2000 Submit_monthly.sh
## For running the along and cross slope transports, you will need more memory than 256 GB, Id say at least 1000GB

script_dir=/g/data/ik11/users/wf4500/Project_panan/GH/Panan_HT_ASC/Python/
cd $script_dir

# load conda
module use /g/data/hh5/public/modules
module unload conda
module load conda/analysis3-unstable

module list

# run
python3 Panan0025_CSHT.py $month $year &>> CSHT_${year}_${month}.out



