#!/bin/bash
#PBS -N HT_panan01
#PBS -P ol01
#PBS -q normalbw
#PBS -l walltime=1:30:00
#PBS -l mem=256GB
#PBS -l software=netcdf
#PBS -l ncpus=28
#PBS -l storage=gdata/v45+gdata/hh5+gdata/cj50+gdata/ik11+scratch/x77+scratch/v45+gdata/x77+gdata/nm03+scratch/nm03
#PBS -v month,year

#Calculates the monthly Along and cross isobath (1000m) speed monthly, for the specified year
## Note, run this with:
## qsub -v month=8,year=2000 Submit_monthly.sh
## For running the along and cross slope transports, you will need more memory than 256 GB, Id say at least 1000GB

script_dir=/home/156/wf4500/v45_wf4500/Project_panan/GH/Panan_HT_ASC/Python/
cd $script_dir

# load conda
module use /g/data/hh5/public/modules
module unload conda
module load conda/analysis3-22.04

module list

# run
# python3 Monthly_cross_slope_heat_transport.py $month $year &>> ASC_job${month}.out


python3 Panan_CSHT.py $month $year &>> CSHTiaf_${year}_${month}.out

#python3 Binning_ASC_speed.py $month $year &>> BinASC_${year}_${month}.out



