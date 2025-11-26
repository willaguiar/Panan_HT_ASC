#!/bin/bash
#PBS -N Pm0025_clim
#PBS -P g40
#PBS -q normalbw
#PBS -l walltime=5:00:00
#PBS -l mem=128GB
#PBS -l software=netcdf
#PBS -l ncpus=14
#PBS -l storage=gdata/v45+gdata/hh5+gdata/cj50+gdata/ik11+scratch/x77+scratch/v45+gdata/x77+gdata/nm03+scratch/nm03


#Calculates the monthly Along and cross isobath (1000m) speed monthly, for the specified year 
## Note, run this with:
## If flag=0, the annual mean isnt calcualted, if flag=1, annual mean is calcualted. annual mean should be calculated only every december

script_dir=/home/156/wf4500/v45_wf4500/Project_panan/GH/Panan_HT_ASC/Python/
cd $script_dir

# load conda
module use /g/data/hh5/public/modules
module unload conda
module load conda/analysis3-unstable
module load nco
module load cdo

module list


#if calc_year=1, then the model calculates the yearly mean after separating the data
indir=/home/156/wf4500/v45_wf4500/Project_panan/GH/Panan_HT_ASC/Processed_data/panan0025/CSHT_mean_CPrhoUhT/
infiles=/home/156/wf4500/v45_wf4500/Project_panan/GH/Panan_HT_ASC/Processed_data/panan0025/CSHT_mean_CPrhoUhT/*.amean_uvt_z.nc


#lets check how many files we have. there should be 12. this is a simple condition for the code to run
infilesnumber=$(ls $infiles >~/infliesline.txt; wc -l ~/infliesline.txt)
infilesnumber=${infilesnumber:0:2}

# if [ "$infilesnumber" = "10" ]; then
#     echo "10 years of data. Lets calculate the annual mean"
#     cd $script_dir
    
# else
#     echo "ERROR!  There aren't 10 of data to calculate the annual mean. Quitting this job."
#     exit 125
# fi

rm ~/infliesline.txt

#ofile=$(ls $infiles > ~/infliesline.txt; head -n 1 ~/infliesline.txt)
ofile=/home/156/wf4500/v45_wf4500/Project_panan/GH/Panan_HT_ASC/Processed_data/panan0025/CSHT_mean_CPrhoUhT/clim.amean_uvt_z.nc

cdo ensmean $infiles $ofile



#rm $infiles
exit