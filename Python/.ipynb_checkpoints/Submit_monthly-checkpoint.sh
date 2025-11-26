#!/bin/bash
#PBS -N HT_panan0025
#PBS -P oz91
#PBS -q hugemem
#PBS -l walltime=4:30:00
#PBS -l mem=1000GB
#PBS -l software=netcdf
#PBS -l ncpus=28
#PBS -l storage=gdata/v45+gdata/hh5+gdata/cj50+gdata/ik11+scratch/x77+scratch/v45+gdata/x77+gdata/nm03+scratch/nm03+gdata/ol01+scratch/oz91+gdata/g40+scratch/g40
#PBS -v month,year

#Calculates the monthly Along and cross isobath (1000m) speed monthly, for the specified year
## Note, run this with:
## qsub -v month=8,year=2000 Submit_monthly.sh
## For running the along and cross slope transports, you will need more memory than 256 GB, Id say at least 1000GB
### panan005 ran in a normalbw queue, mem=250GB and ncpus=28
### panan0025 ran in a hugemembw queue, mem=1000GB and ncpus=28
script_dir=/g/data/ik11/users/wf4500/Project_panan/GH/Panan_HT_ASC/Python/
cd $script_dir

# load conda
module use /g/data/hh5/public/modules
module unload conda
#module load conda/analysis3-unstable
module load conda/analysis3-24.04

module list

# run
# python3 Monthly_cross_slope_heat_transport.py $month $year &>> ASC_job${month}.out


python3 Panan_CSHTsigma.py $month $year &>> CSHT_${year}_${month}.out

#python3 Binning_ASC_speed.py $month $year &>> BinASC_${year}_${month}.out



