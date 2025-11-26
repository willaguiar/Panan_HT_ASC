#!/bin/bash
#PBS -N P0025_clim
#PBS -P g40
#PBS -q normalbw
#PBS -l walltime=2:00:00
#PBS -l mem=256GB
#PBS -l software=netcdf
#PBS -l ncpus=28
#PBS -l storage=gdata/v45+gdata/hh5+gdata/cj50+gdata/ik11+scratch/x77+scratch/v45+gdata/x77+gdata/nm03+scratch/nm03
#PBS -v indir

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

#if calc_year=1, then the model calculates the yearly mean after separating the data
calc_year=["011","23","35","47","59","71","83","95","107","119"]
file0="/*.ocean_month_z.nc"
file_in=$(ls $indir$file0)
file_out=/home/156/wf4500/v45_wf4500/Project_panan/GH/Panan_HT_ASC/Processed_data/panan0025/CSHT_mean_CPrhoUhT/${file_in:70:8}.ocean_uvt_z.nc
#${file_in:70:8}
cdo select,name=uo,vo,thetao,volcello $file_in $file_out
sleep 240

if [ "${file_in:74:2}" = "12" ]; then
    echo "We have a total of 12 months already. Lets calculate the annual mean"
    cd $script_dir
    qsub ./Annual_mean_from_separated.sh
else
    echo "Strings are not equal. Separating the next set of data"
fi



# output_num=${indir:66:3}
# if [[ "${calc_year[@]}" =~ $output_num ]];
# then 
# echo "Submitting the code to calculate annual mean"
# sleep 60
# cd $script_dir
# qsub annual_mean_0025.sh
# # cd /home/156/wf4500/v45_wf4500/Project_panan/GH/Panan_HT_ASC/Processed_data/panan0025/CSHT_mean_CPrhoUhT

# # #List of files
# # file00=$(ls >~/listforfiles.txt; sed '1q;d' ~/listforfiles.txt)
# # file01=$(ls >~/listforfiles.txt; sed '2q;d' ~/listforfiles.txt)
# # file02=$(ls >~/listforfiles.txt; sed '3q;d' ~/listforfiles.txt)
# # file03=$(ls >~/listforfiles.txt; sed '4q;d' ~/listforfiles.txt)
# # file04=$(ls >~/listforfiles.txt; sed '5q;d' ~/listforfiles.txt)
# # file05=$(ls >~/listforfiles.txt; sed '6q;d' ~/listforfiles.txt)
# # file06=$(ls >~/listforfiles.txt; sed '7q;d' ~/listforfiles.txt)
# # file07=$(ls >~/listforfiles.txt; sed '8q;d' ~/listforfiles.txt)
# # file08=$(ls >~/listforfiles.txt; sed '9q;d' ~/listforfiles.txt)
# # file09=$(ls >~/listforfiles.txt; sed '10q;d' ~/listforfiles.txt)
# # file10=$(ls >~/listforfiles.txt; sed '11q;d' ~/listforfiles.txt)
# # file11=$(ls >~/listforfiles.txt; sed '12q;d' ~/listforfiles.txt)

# # cdo ensmean $file00 $file01 $file02 $file03 $file04 $file05 $file06 $file07 $file08 $file019 $file10 $file11 ${file_in:70:8}.ocean_yearly_uvt_z.nc

# # rm $file00 $file01 $file02 $file03 $file04 $file05 $file06 $file07 $file08 $file019 $file10 $file11
# # cdo ensmean 19910101.ocean_uvt_z.nc 19910201.ocean_uvt_z.nc 19910301.ocean_uvt_z.nc 19910401.ocean_uvt_z.nc 19910501.ocean_uvt_z.nc 19910601.ocean_uvt_z.nc 19910701.ocean_uvt_z.nc 19910801.ocean_uvt_z.nc 19910901.ocean_uvt_z.nc 19911001.ocean_uvt_z.nc 19911101.ocean_uvt_z.nc 19911201.ocean_uvt_z.nc  1991.ocean_yearly_uvt_z.nc


# else echo "not calculating annual mean"
# fi

