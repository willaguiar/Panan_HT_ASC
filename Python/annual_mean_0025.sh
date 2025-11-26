#!/bin/bash
#PBS -N annual_0025
#PBS -P g40
#PBS -q normalbw
#PBS -l walltime=3:00:00
#PBS -l mem=512GB
#PBS -l software=netcdf
#PBS -l ncpus=56
#PBS -l storage=gdata/v45+gdata/hh5+gdata/cj50+gdata/ik11+scratch/x77+scratch/v45+gdata/x77+gdata/nm03+scratch/nm03



script_dir=/home/156/wf4500/v45_wf4500/Project_panan/GH/Panan_HT_ASC/Python/
cd $script_dir

# load conda
module use /g/data/hh5/public/modules
module unload conda
module load conda/analysis3-22.04
module load nco
module load cdo

module list

sleep 60
echo "calculting annual mean"
cd /home/156/wf4500/v45_wf4500/Project_panan/GH/Panan_HT_ASC/Processed_data/panan0025/CSHT_mean_CPrhoUhT
sleep 60
#List of files
file00=$(ls ./*01.ocean_uvt_z.nc >~/listforfiles.txt; sed '1q;d' ~/listforfiles.txt)
file01=$(ls ./*01.ocean_uvt_z.nc >~/listforfiles.txt; sed '2q;d' ~/listforfiles.txt)
file02=$(ls ./*01.ocean_uvt_z.nc >~/listforfiles.txt; sed '3q;d' ~/listforfiles.txt)
file03=$(ls ./*01.ocean_uvt_z.nc >~/listforfiles.txt; sed '4q;d' ~/listforfiles.txt)
file04=$(ls ./*01.ocean_uvt_z.nc >~/listforfiles.txt; sed '5q;d' ~/listforfiles.txt)
file05=$(ls ./*01.ocean_uvt_z.nc >~/listforfiles.txt; sed '6q;d' ~/listforfiles.txt)
file06=$(ls ./*01.ocean_uvt_z.nc >~/listforfiles.txt; sed '7q;d' ~/listforfiles.txt)
file07=$(ls ./*01.ocean_uvt_z.nc >~/listforfiles.txt; sed '8q;d' ~/listforfiles.txt)
file08=$(ls ./*01.ocean_uvt_z.nc >~/listforfiles.txt; sed '9q;d' ~/listforfiles.txt)
file09=$(ls ./*01.ocean_uvt_z.nc >~/listforfiles.txt; sed '10q;d' ~/listforfiles.txt)
file10=$(ls ./*01.ocean_uvt_z.nc >~/listforfiles.txt; sed '11q;d' ~/listforfiles.txt)
file11=$(ls ./*01.ocean_uvt_z.nc >~/listforfiles.txt; sed '12q;d' ~/listforfiles.txt)

cdo ensmean $file00 $file01 $file02 $file03 $file04 $file05 $file06 $file07 $file08 $file019 $file10 $file11 ${file00:2:4}.ocean_yearly_uvt_z.nc

# $file00 $file01 $file02 $file03 $file04 $file05 $file06 $file07 $file08 $file019 $file10 $file11
# cdo ensmean 19910101.ocean_uvt_z.nc 19910201.ocean_uvt_z.nc 19910301.ocean_uvt_z.nc 19910401.ocean_uvt_z.nc 19910501.ocean_uvt_z.nc 19910601.ocean_uvt_z.nc 19910701.ocean_uvt_z.nc 19910801.ocean_uvt_z.nc 19910901.ocean_uvt_z.nc 19911001.ocean_uvt_z.nc 19911101.ocean_uvt_z.nc 19911201.ocean_uvt_z.nc  1991.ocean_yearly_uvt_z.nc


echo "Annual mean finished"

