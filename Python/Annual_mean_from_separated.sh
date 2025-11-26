#!/bin/bash
#PBS -N Pm0025_clim
#PBS -P g40
#PBS -q normalbw
#PBS -l walltime=5:00:00
#PBS -l mem=256GB
#PBS -l software=netcdf
#PBS -l ncpus=28
#PBS -l storage=gdata/v45+gdata/hh5+gdata/cj50+gdata/ik11+scratch/x77+scratch/v45+gdata/x77+gdata/nm03+scratch/nm03


#Calculates the monthly Along and cross isobath (1000m) speed monthly, for the specified year 
## Note, run this with:
## qsub -v indir=/g/data/ik11/outputs/mom6-panan/panant-0025-zstar-ACCESSyr2/output011 separate_UVT_panan0025.sh
## If flag=0, the annual mean isnt calcualted, if flag=1, annual mean is calcualted. annual mean should be calculated only every december

script_dir=/home/156/wf4500/v45_wf4500/Project_panan/GH/Panan_HT_ASC/Python/
cd $script_dir

# load conda
module use /g/data/hh5/public/modules
module unload conda
module load conda/analysis3-22.04
module load nco
module load cdo

module list

sleep 480

#if calc_year=1, then the model calculates the yearly mean after separating the data
indir=/home/156/wf4500/v45_wf4500/Project_panan/GH/Panan_HT_ASC/Processed_data/panan0025/CSHT_mean_CPrhoUhT/
infiles=/home/156/wf4500/v45_wf4500/Project_panan/GH/Panan_HT_ASC/Processed_data/panan0025/CSHT_mean_CPrhoUhT/*.ocean_uvt_z.nc


#lets check how many files we have. there should be 12. this is a simple condition for the code to run
infilesnumber=$(ls $infiles >~/infliesline.txt; wc -l ~/infliesline.txt)
infilesnumber=${infilesnumber:0:2}

if [ "$infilesnumber" = "12" ]; then
    echo "12 months test passed. Lets calculate the annual mean"
    cd $script_dir
    
else
    echo "ERROR!  There aren't 12 months of data to calculate the annual mean. Quitting this job."
    exit 125
fi

rm ~/infliesline.txt

ofile=$(ls $infiles > ~/infliesline.txt; head -n 1 ~/infliesline.txt)
ofile=${ofile:0:106}.amean_uvt_z.nc

cdo ensmean $infiles $ofile



#rm $infiles
exit