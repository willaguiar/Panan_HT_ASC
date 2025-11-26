# # Compute monthly ASC along contour and save

import cosima_cookbook as cc
from cosima_cookbook import distributed as ccd
import matplotlib.pyplot as plt
import numpy as np
import netCDF4 as nc
import xarray as xr
import glob,os
import cmocean.cm as cmocean
import os
import sys
import glob

import logging
logging.captureWarnings(True)
logging.getLogger('py.warnings').setLevel(logging.ERROR)
logging.getLogger('distributed.utils_perf').setLevel(logging.ERROR)

from dask.distributed import Client

import climtas.nci
import warnings # ignore these warnings
warnings.filterwarnings("ignore", category = FutureWarning)
warnings.filterwarnings("ignore", category = UserWarning)
warnings.filterwarnings("ignore", category = RuntimeWarning)

if __name__ == '__main__':

	climtas.nci.GadiClient()
	
	session = cc.database.create_session('/home/156/wf4500/databases/access/panan005_rerun.db') #panan005
	
	
	
	monthdays = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
	month = str(int(sys.argv[1]))
	month = month.zfill(2)
	year = str(sys.argv[2])
	start_time=year+'-'+month  
	 #Start_time0 and end_time0 are for importing the daily transport, and it ahsthe number of days in the month    
	start_time0=year+'-'+month +'-01'     
	end_time0=year+'-'+month +'-' + str(monthdays[int(int(sys.argv[1])-1)])
	print(start_time0) 
	print(end_time0) 
	exp = 'panan_005deg_jra55_ryf_2024_12_14'
	
	print("Start date =" + start_time) 
	year2=str(int(start_time[0:4])+1)
	month2=str(int(start_time[5:7])+1)
	month2=str(int(month2))
	month2 = month2.zfill(2)    
	print("month2 is =" + month2) 
	print("year2 is =" + str(year2))     
	
	imon = int(sys.argv[1])
	if imon <12:
		end_time=year+'-'+month2
	else:
		end_time=year2+'-01'
	
	print("End date =" + end_time) 
	
	
	time_period = str(int(start_time[:4]))+'-'+str(int(end_time[:4]))
	
	# reference density value:
	rho_0 = 1035.0
	# specific heat capacity of sea water:
	cp = 3992.1
	lat_range = slice(-90,-59)
	
	isobath_depth = 1000
	
	# pick a freezing point temp:
	temp_freezing = -1.9
	
	
	
	
	
	#Getting mass transport in kg/s
	vmo_nt = cc.querying.getvar(exp,'vmo',session,ncfile='%month_rho%',start_time=start_time,end_time=end_time,chunks={}).sel(time=slice(start_time,end_time))
	umo_et = cc.querying.getvar(exp,'umo',session,ncfile='%month_rho%',start_time=start_time,end_time=end_time,chunks={}).sel(time=slice(start_time,end_time))
	
	vmo_nt = vmo_nt.sel(yq=lat_range).sel(time=slice(start_time0,end_time0))
	umo_et = umo_et.sel(yh=lat_range).sel(time=slice(start_time0,end_time0))
	vmo_nttime=vmo_nt.time
	
	
	#importing volume
	vol_centre = cc.querying.getvar(exp,'volcello',session,ncfile='%month_rho2.nc',start_time=start_time,end_time=end_time,chunks={})\
	.sel(time=slice(start_time,end_time)).sel(yh=lat_range).sel(time=slice(start_time0,end_time0))
	
	#Making volume on xq and yq grid  ######
	
	#First making a halo for proper edges interpolation
	lon_interval = vol_centre.xh.diff('xh').mean()
	lon_rightappend = vol_centre.xh[-1].values+np.cumsum([lon_interval,lon_interval,lon_interval,lon_interval,lon_interval])
	lon_leftappend = vol_centre.xh[0].values - \
	np.flip(np.cumsum([lon_interval,lon_interval,lon_interval,lon_interval]))
	
	newlon = np.concatenate((lon_leftappend,vol_centre.xh.values,lon_rightappend), axis=0)
	vol_centre_halo = xr.concat([vol_centre.isel(xh=slice(-5,-1)),vol_centre,vol_centre.isel(xh=slice(0,5))], dim='xh')
	vol_centre_halo['xh'] = newlon
	
	
	#Now interpolating volumes onto the xq and yq grids
	vol_yhxq = vol_centre_halo.interp(xh=umo_et.xq)
	vol_yqxq = vol_centre_halo.interp(xh=umo_et.xq,yh=vmo_nt.yq)
	vol_yqxh = vol_centre_halo.interp(yh=vmo_nt.yq)
	vol_yhxh = vol_centre.copy()
	
	#dxs in all grid points
	dx_xhyh = cc.querying.getvar(exp,'dxt',session,ncfile='19990101.ocean_static.nc')
	dx_xqyh = cc.querying.getvar(exp,'dxCu',session,ncfile='19990101.ocean_static.nc')
	dx_xhyq = cc.querying.getvar(exp,'dxCv',session,ncfile='19990101.ocean_static.nc')
	dx_xqyq = dx_xhyh.rename({'xh':'xq','yh':'yq'}).copy()
	dx_xqyq=  xr.concat([dx_xqyq[:,-1],dx_xqyq], dim='xq')
	dx_xqyq['yq'] = dx_xhyq.yq[:-1]
	dx_xqyq['xq'] = dx_xqyh.xq
	
	#dys in all grid points
	dy_xhyh = cc.querying.getvar(exp,'dyt',session,ncfile='19990101.ocean_static.nc')
	dy_xqyh = cc.querying.getvar(exp,'dyCu',session,ncfile='19990101.ocean_static.nc')
	dy_xhyq = cc.querying.getvar(exp,'dyCv',session,ncfile='19990101.ocean_static.nc')
	dy_xqyq = dy_xhyh.rename({'xh':'xq','yh':'yq'}).copy()
	dy_xqyq=  xr.concat([dy_xqyq[:,-1],dy_xqyq], dim='xq')
	dy_xqyq['yq'] = dy_xhyq.yq[:-1]
	dy_xqyq['xq'] = dy_xqyh.xq
	
	#setting the reference rho
	rho_ref = 1035 #kg m^{-3}, used as standard by Pan-Antarctic
	
	#first for U
	U_xq = (umo_et*(1/rho_ref) * (1/vol_yhxq) * dx_xqyh).compute()
	
	#now for V
	V_yq = (vmo_nt*(1/rho_ref) * (1/vol_yqxh) * dy_xhyq).compute()
	
	
	
	
	U_xq = U_xq.drop_vars('xh').where(U_xq>-9e10).where(U_xq<9e10)
	V_yq  = V_yq .drop_vars('yh').where(V_yq>-9e10).where(V_yq<9e10)
	
	U_qmean=xr.open_dataset('/g/data/ik11/users/wf4500/Project_panan/GH/Panan_HT_ASC/Processed_data/panan005_rerun/EKE_along_contour_rho/U_mean_1991_2000.nc').uo
	V_qmean=xr.open_dataset('/g/data/ik11/users/wf4500/Project_panan/GH/Panan_HT_ASC/Processed_data/panan005_rerun/EKE_along_contour_rho/V_mean_1991_2000.nc').vo
	
	U_xq = U_xq - U_qmean
	V_yq = V_yq - V_qmean
	
	
	#importing heat transports for the grid along the isobath
	#panan01 daily as f(z,time)
	src_p01 = '/g/data/ik11/users/wf4500/Project_panan/GH/Panan_HT_ASC/Processed_data/panan01_rerun/CSHT_month_rhoonline/*'
	#panan005 daily as f(z,time)
	src_p005 = '/g/data/ik11/users/wf4500/Project_panan/GH/Panan_HT_ASC/Processed_data/panan005_rerun/CSHT_month_rhoonline/*'
	#panan0025 daily as f(z,time)
	src_p0025 = '/g/data/ik11/users/wf4500/Project_panan/GH/Panan_HT_ASC/Processed_data/panan0025/CSHT_month_rhoonline/*'
	
	lat_slice  = slice(-83,-59)
	
	figdir='/home/156/wf4500/v45_wf4500/Project_panan/GH/Panan_HT_ASC/figs/'
	
	
	#importing panan005
	gl =glob.glob(src_p005)
	gl=sorted(gl)
	print("%i netcdf data files"%len(gl))
	p005_HTmean=xr.open_mfdataset(gl, concat_dim='time', combine='nested').mean('time')
	p005_HT=xr.open_mfdataset(gl, concat_dim='time', combine='nested')
	
	
	# Extracting values along isobath
	U_along_contour_005 = U_xq.interp(xq=p005_HTmean.lon_along_contour,\
	                                 yh=p005_HTmean.lat_along_contour,method='nearest')\
	.drop_vars({'xh','xq','yh'})
	V_along_contour_005 = V_yq.interp(xh=p005_HTmean.lon_along_contour,\
	                                 yq=p005_HTmean.lat_along_contour,method='nearest')\
	.drop_vars({'xh','yq','yh'})
	
	EKE_005 = 0.5 * ((U_along_contour_005**2) + (V_along_contour_005**2))
	EKE_005.name = 'EKE_m2s2'
	
	
	
	# convert to longitude coordinate and average into 3 degree longitude bins:
	# in degrees:
	bin_width = 3
	bin_spacing = 0.25
	lon_west = -280
	lon_east = 80
	lon_along_contour = np.array(p005_HTmean.lon_along_contour)
	lat_along_contour = np.array(p005_HTmean.lat_along_contour)
	EKE_005_np=np.array(EKE_005.isel(time=0))
	# new coordinate and midpoints of longitude bins:
	full_lon_coord = np.arange(lon_west,lon_east+bin_spacing,bin_spacing)
	lon_bin_midpoints = np.arange(lon_west+bin_width/2,lon_east-bin_width/2,bin_spacing)
	n_bin_edges = len(full_lon_coord)
	
	# sum into longitude bins:
	# need to be very careful of loops, we can't just mask over longitude values, but instead pick indices 
	# on the isobath contour and sum continously along contour between defined indices.
	# (i.e. lon_along_contour is not monotonic)
	# find points on contour to define edges of longitude bins:
	bin_edge_indices = np.zeros(n_bin_edges)
	for lon_bin in range(n_bin_edges-1):
	    # find first isobath point that has the right longitude:
	    first_point = np.where(lon_along_contour>=full_lon_coord[lon_bin])[0][0]
	    # then find all other isobath points with the same longitude as that first point:
	    same_lon_points = np.where(lon_along_contour==lon_along_contour[first_point])[0]
	    # we want the most southerly of these points on the same longitude line:
	    bin_edge_indices[lon_bin] = same_lon_points[np.argmin(lat_along_contour[same_lon_points])]
	
	# define east/west edges:
	bin_edge_indices = bin_edge_indices.astype(int)
	bin_edge_indices_west = bin_edge_indices[:-int(bin_width/bin_spacing)-1]
	bin_edge_indices_east = bin_edge_indices[int(bin_width/bin_spacing):-1]
	n_bins = len(bin_edge_indices_west)
	
	# sum heat transport from isobath coord into new longitude coord:
	EKE_3degbins = np.zeros([99,n_bins])
	for lon_bin in range(n_bins):
		EKE0 = EKE_005_np[:,bin_edge_indices_west[lon_bin]:bin_edge_indices_east[lon_bin]]
		EKE_3degbins[:,lon_bin] = np.sum(EKE0,axis=1)
	
	factor = np.nansum(EKE_005_np) / np.nansum(EKE_3degbins)
	EKE_3degbins = EKE_3degbins * factor
	
	
	
	p005_binnedEKE = p005_HT.sel(time=slice(start_time0,end_time0)).binned_cross_slope_heat_trans.copy()
	p005_binnedEKE.values =np.expand_dims(EKE_3degbins,axis=0)
	p005_binnedEKE.name =  'EKE_binned_m2s2'
	
	
	EKE_005_tosave = xr.merge((p005_binnedEKE,EKE_005))
	
	
	#saving file
	dir_file = '/g/data/ik11/users/wf4500/Project_panan/GH/Panan_HT_ASC/Processed_data/panan005_rerun/EKE_along_contour_rho/'
	name_file = dir_file + 'EKE_along_contour_' + start_time + '.nc'
	EKE_005_tosave['factor']= factor
	
	EKE_005_tosave.to_netcdf(name_file)
	
	
	EKE_005_tosave
	
	
	
	
	
	
	print('Finished successful')
	