# # Compute monthly ASC along contour and save

import cosima_cookbook as cc
from cosima_cookbook import distributed as ccd
import matplotlib.pyplot as plt
import numpy as np
import netCDF4 as nc
import xarray as xr
from gsw import sigma2
from gsw import CT_from_pt
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
from dask import delayed

import climtas.nci
import warnings # ignore these warnings
warnings.filterwarnings("ignore", category = FutureWarning)
warnings.filterwarnings("ignore", category = UserWarning)
warnings.filterwarnings("ignore", category = RuntimeWarning)

if __name__ == '__main__':

	climtas.nci.GadiClient()
	
	session = cc.database.create_session()
	
	#imon=int(sys.argv[1])
	#if imon<=9:
	#    month='0' + str(imon)
	#else:
	#    month=str(imon)
	
	year = str(sys.argv[2])
	monthdays = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
	month = str(int(sys.argv[1]))
	if int(sys.argv[1])<12: 
		month2 = str(int(sys.argv[1])+1).zfill(2)
		year2=year
	else:
		month2='01'
		year2=str(int(sys.argv[2])+1)
	
	month = month.zfill(2)
	year = str(sys.argv[2])
	start_time=year+'-'+month  
	start_time0=year+'-'+month +'-01'     
	end_time0=year+'-'+month +'-' + str(monthdays[int(int(sys.argv[1])-1)])
	end_time00=year2+'-'+month2 +'-01'
	print(start_time0) 
	print(end_time0) 
	exp = 'panant-005-zstar-ACCESSyr2'
	
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
	temp_freezing = -3.82
	
	
	print("Importing daily CSHT(z)")
	if exp=='panant-01-zstar-ACCESSyr2':
		src = '/home/156/wf4500/v45_wf4500/Project_panan/GH/Panan_HT_ASC/Processed_data/panan01/CSHT_daily_z/*'
		outbase= '/home/156/wf4500/v45_wf4500/Project_panan/GH/Panan_HT_ASC/Processed_data/panan01/'
	elif  exp=='panant-005-zstar-ACCESSyr2':
		src = '/home/156/wf4500/v45_wf4500/Project_panan/GH/Panan_HT_ASC/Processed_data/panan005/CSHT_daily_z/*'
		outbase= '/home/156/wf4500/v45_wf4500/Project_panan/GH/Panan_HT_ASC/Processed_data/panan005/'
	else:
		print("We dont have daily heat transport in panan0025. A error will occur downstream")

	HT=xr.open_mfdataset(src); HT=HT.sel(time=slice(start_time,end_time0))
	lon_bin_midpoints = HT.lon_bin_midpoints
	lon_along_contour = HT.lon_along_contour.isel(time=0).drop('time')
	lat_along_contour = HT.lat_along_contour.isel(time=0).drop('time')

	print("temperature and salinity")
	T = cc.querying.getvar(exp,'thetao',session,ncfile='%daily_z%',start_time=start_time,end_time=end_time00).sel(time=slice(start_time,end_time))
	S = cc.querying.getvar(exp,'so',session,ncfile='%daily_z%',start_time=start_time,end_time=end_time00).sel(time=slice(start_time,end_time))
	T_along=T.sel(xh=lon_along_contour,yh=lat_along_contour,method='nearest')
	S_along=S.sel(xh=lon_along_contour,yh=lat_along_contour,method='nearest')

	print("Calculating daily sigma_2 along contour")
	CT_along = CT_from_pt(S_along,T_along) # conservative temperature from potential temperature
	rho2_along=sigma2(S_along,CT_along)
	rho2_along = rho2_along.compute()

	print('Binning into the lon_bin_midpoints')
	Vol = cc.querying.getvar(exp,'volcello',session,ncfile='%daily_z%',start_time=start_time,end_time=end_time00).sel(time=slice(start_time,end_time))
	Vol_along=Vol.sel(xh=lon_along_contour,yh=lat_along_contour,method='nearest') #this is be used for the weighted average
	Vol_along = Vol_along.compute()

	#binning
	bin_width = 3
	bin_spacing = 0.25
	lon_west = -280
	lon_east = 80
	# new coordinate and midpoints of longitude bins:
	full_lon_coord = np.arange(lon_west,lon_east+bin_spacing,bin_spacing)
	lon_bin_midpoints0 = np.arange(lon_west+bin_width/2,lon_east-bin_width/2,bin_spacing)
	n_bin_edges = len(full_lon_coord)
	lon_along_contour0=np.array(lon_along_contour)
	lat_along_contour0=np.array(lat_along_contour)
	bin_edge_indices = np.zeros(n_bin_edges)
	for lon_bin in range(n_bin_edges-1):
		first_point = np.where(lon_along_contour0>=full_lon_coord[lon_bin])[0][0]
		same_lon_points = np.where(lon_along_contour0==lon_along_contour0[first_point])[0]
		bin_edge_indices[lon_bin] = same_lon_points[np.argmin(lat_along_contour0[same_lon_points])]
	bin_edge_indices = bin_edge_indices.astype(int)
	bin_edge_indices_west = bin_edge_indices[:-int(bin_width/bin_spacing)-1]
	bin_edge_indices_east = bin_edge_indices[int(bin_width/bin_spacing):-1]
	n_bins = len(bin_edge_indices_west)

	##sum heat transport from isobath coord into new longitude coord:
	meansigmabinned = np.zeros([int(monthdays[int(int(sys.argv[1])-1)]),75,n_bins])

	@delayed
	def meansigmabinnedv(meansigmabinned,rho2_along,bin_edge_indices_west,bin_edge_indices_east,Vol_along,n_bins):
		for lon_bin in range(n_bins):
			meansigmabin0 = rho2_along.isel(contour_index=slice(bin_edge_indices_west[lon_bin],bin_edge_indices_east[lon_bin])).weighted(Vol_along.fillna(0))
			meansigmabinned[:,:,lon_bin] = meansigmabin0.mean('contour_index')
		return meansigmabinned
	
	meansigmabinned = meansigmabinnedv(meansigmabinned,rho2_along,bin_edge_indices_west,bin_edge_indices_east,Vol_along,n_bins)
	meansigmabinned = meansigmabinned.compute()



	binnedsigma = xr.DataArray(data=meansigmabinned,dims=["time","z_l","lon_bin_midpoints"])
	binnedsigma.name='binned_sigma2'
	binnedsigma['z_l']=Vol_along.z_l
	binnedsigma['lon_bin_midpoints']=HT.lon_bin_midpoints
	binnedsigma['time']=HT.time
	rho2_along.name='rho2_along'
	rho2_along.attrs['units']='kg/m3 -1000'
	rho2_along.attrs['long_name']='Offline calculated sigma2'
	SIGMA2=xr.merge([rho2_along,binnedsigma])

	print("saving daily sigma_2")
	SIGMA2.to_netcdf(outbase+'rho2_dailyz_offline/' + 'sigma2_daily_1kmisobath_' + start_time + '.nc')

	print("Transposing HT from z to sigma2") 
	HT=HT.compute()
	MT01=xr.open_dataset('/g/data/e14/cs6673/mom6_comparison/data_DSW/vol_transp_across_1000m_isobath_panan_01deg_jra55_ryf_1m_2000.nc')
	refsigma=np.array(MT01.rho2_l)

	#setting the nan case to be substituted in the delayed calculation
	rho_ub=SIGMA2.rho2_along.isel(contour_index=0,time=0,z_l=slice(0,51)); 
	HTρn_ub=(HT.unbinned_heat_transp_across_contour).isel(contour_index=0,time=0)
	HTρn_ub=HTρn_ub.rename({'zl':'rho'}); HTρn_ub['rho']=np.array(rho_ub)
	HTρ2n_ub = HTρn_ub.isel(rho= ~np.isnan(HTρn_ub.rho)).drop_duplicates(dim='rho', keep='first')
	HTρ2n_ub = HTρ2n_ub.interp(rho = refsigma,kwargs={"bounds_error": False})
	HTρ2n_ub=HTρ2n_ub.where(HTρ2n_ub!=HTρ2n_ub)
	HTρ2n_ub = HTρ2n_ub.compute()
	factor=float((HT.unbinned_heat_transp_across_contour.mean('time').sum())/(HT.binned_cross_slope_heat_trans.mean('time').sum() + HT.zonal_convergence.mean('time').sum()))

	#Defining the function to transition from Z space to sigma space as a delayed object, for faster multicpu computation

	@delayed
	def depthtosigma(SIGMA2,HT,HTρ2n_ub,factor,refsigma):
		#alocating space
		HTρ2_ubo = np.zeros([int(np.size(SIGMA2.time)),99,int(np.size(SIGMA2.contour_index))]) *np.nan
		HTρ2_bo = np.zeros([int(np.size(SIGMA2.time)),99,int(np.size(SIGMA2.lon_bin_midpoints))]) *np.nan 
		HTρ2_zco = HTρ2_bo
		HTρ2_bzco = HTρ2_bo
		refsigma=refsigma-1000
		for n in (SIGMA2.contour_index.values - 1):
			for t in range(0,np.size(SIGMA2.time)):
				if n<int(np.size(SIGMA2.lon_bin_midpoints)):
					rho_ub=SIGMA2.rho2_along.rename({'z_l':'zl'}).isel(contour_index=n,time=t,zl=slice(0,51))
					rho_b=SIGMA2.binned_sigma2.rename({'z_l':'zl'}).isel(lon_bin_midpoints=n,time=t,zl=slice(0,51))
					if ~np.isnan(np.nanmean(rho_ub)): 
						HTρ_ub=(HT.unbinned_heat_transp_across_contour).isel(contour_index=n,time=t)
						HTρ_b=(HT.binned_cross_slope_heat_trans).isel(lon_bin_midpoints=n,time=t)
						HTρ_zc=(HT.zonal_convergence).isel(lon_bin_midpoints=n,time=t)
						HTρ_bzc=((HT.binned_cross_slope_heat_trans + HT.zonal_convergence)*factor).isel(lon_bin_midpoints=n,time=t)
						HTρ_ub=HTρ_ub.rename({'zl':'rho'});HTρ_b=HTρ_b.rename({'zl':'rho'})
						HTρ_zc=HTρ_zc.rename({'zl':'rho'});HTρ_bzc=HTρ_bzc.rename({'zl':'rho'})

						HTρ_ub['rho']=np.array(rho_ub); HTρ_b['rho']=np.array(rho_b)
						HTρ_zc['rho']=np.array(rho_b); HTρ_bzc['rho']=np.array(rho_b)


						HTρ2_ub = HTρ_ub.isel(rho= ~np.isnan(HTρ_ub.rho)).drop_duplicates(dim='rho', keep='first')
						HTρ2_ub = HTρ2_ub.interp(rho = refsigma,kwargs={"bounds_error": False})
						HTρ2_ubo[t,:,n]=np.array(HTρ2_ub.values)
						HTρ2_zc = HTρ_zc.isel(rho= ~np.isnan(HTρ_zc.rho)).drop_duplicates(dim='rho', keep='first')
						HTρ2_zc = HTρ2_zc.interp(rho = refsigma,kwargs={"bounds_error": False})
						HTρ2_zco[t,:,n]=np.array(HTρ2_zc.values)
						HTρ2_bzc = HTρ_bzc.isel(rho= ~np.isnan(HTρ_bzc.rho)).drop_duplicates(dim='rho', keep='first')
						HTρ2_bzc = HTρ2_bzc.interp(rho = refsigma,kwargs={"bounds_error": False})
						HTρ2_bzco[t,:,n]=np.array(HTρ2_bzc.values)
						HTρ2_b = HTρ_b.isel(rho= ~np.isnan(HTρ_b.rho)).drop_duplicates(dim='rho', keep='first')
						HTρ2_b = HTρ2_b.interp(rho = refsigma,kwargs={"bounds_error": False})
						HTρ2_bo[t,:,n]=np.array(HTρ2_b.values)
					else:
						HTρ2_ubo[t,:,n] = np.array(HTρ2n_ub.values)
						HTρ2_bo[t,:,n] = np.array(HTρ2n_ub.values)
						HTρ2_zco[t,:,n] = np.array(HTρ2n_ub.values)
						HTρ2_bzco[t,:,n] = np.array(HTρ2n_ub.values)
				else:
					if ~np.isnan(np.nanmean(rho_ub)): 
						HTρ_ub=(HT.unbinned_heat_transp_across_contour).isel(contour_index=n,time=t)
						HTρ_ub=HTρ_ub.rename({'zl':'rho'})
						HTρ_ub['rho']=np.array(rho_ub); 
						HTρ2_ub = HTρ_ub.isel(rho= ~np.isnan(HTρ_ub.rho)).drop_duplicates(dim='rho', keep='first')
						HTρ2_ub = HTρ2_ub.interp(rho = refsigma,kwargs={"bounds_error": False})
						HTρ2_ubo[t,:,n]=np.array(HTρ2_ub.values)
					else:
						HTρ2_ubo[t,:,n] = np.array(HTρ2n_ub.values)
	
		return HTρ2_ubo, HTρ2_bo, HTρ2_zco, HTρ2_bzco
	
	# Setting the stage for the delayed object
	sigmaconversion = depthtosigma(SIGMA2,HT,HTρ2n_ub,factor,refsigma)

	#running the delayed object
	HTρ2_ubo, HTρ2_bo, HTρ2_zco, HTρ2_bzco = sigmaconversion.compute()

	#defining rename functions
	def renamedimsB(HTρ2_b,SIGMA2):
		if np.shape(HTρ2_b.A)[0]==99: 
			HTρ2_b= HTρ2_b.rename({'A':'rho2_l'})
		elif np.shape(HTρ2_b.A)[0]==1428: 
			HTρ2_b= HTρ2_b.rename({'A':'lon_bin_midpoints'})
		elif np.shape(HTρ2_b.A)[0]==np.shape(SIGMA2.time)[0]:
			HTρ2_b= HTρ2_b.rename({'A':'time'})
		if np.shape(HTρ2_b.B)[0]==99: 
			HTρ2_b= HTρ2_b.rename({'B':'rho2_l'})
		elif np.shape(HTρ2_b.B)[0]==1428: 
			HTρ2_b= HTρ2_b.rename({'B':'lon_bin_midpoints'})
		elif np.shape(HTρ2_b.B)[0]==np.shape(SIGMA2.time)[0]:
			HTρ2_b= HTρ2_b.rename({'B':'time'})
		if np.shape(HTρ2_b.C)[0]==99: 
			HTρ2_b= HTρ2_b.rename({'C':'rho2_l'})
		elif np.shape(HTρ2_b.C)[0]==1428: 
			HTρ2_b= HTρ2_b.rename({'C':'lon_bin_midpoints'})
		elif np.shape(HTρ2_b.C)[0]==np.shape(SIGMA2.time)[0]:
			HTρ2_b= HTρ2_b.rename({'C':'time'})
		return HTρ2_b

	def renamedimsUB(HTρ2_ub,SIGMA2):
		if np.shape(HTρ2_ub.A)[0]==99: 
			HTρ2_ub= HTρ2_ub.rename({'A':'rho2_l'})
		elif np.shape(HTρ2_ub.A)[0]==6010: 
			HTρ2_ub= HTρ2_ub.rename({'A':'contour_index'})
		elif np.shape(HTρ2_ub.A)[0]==np.shape(SIGMA2.time)[0]:
			HTρ2_ub= HTρ2_ub.rename({'A':'time'})
		if np.shape(HTρ2_ub.B)[0]==99: 
			HTρ2_ub= HTρ2_ub.rename({'B':'rho2_l'})
		elif np.shape(HTρ2_ub.B)[0]==6010: 
			HTρ2_ub= HTρ2_ub.rename({'B':'contour_index'})
		elif np.shape(HTρ2_ub.B)[0]==np.shape(SIGMA2.time)[0]:
			HTρ2_ub= HTρ2_ub.rename({'B':'time'})
		if np.shape(HTρ2_ub.C)[0]==99: 
			HTρ2_ub= HTρ2_ub.rename({'C':'rho2_l'})
		elif np.shape(HTρ2_ub.C)[0]==6010: 
			HTρ2_ub= HTρ2_ub.rename({'C':'contour_index'})
		elif np.shape(HTρ2_ub.C)[0]==np.shape(SIGMA2.time)[0]:
			HTρ2_ub= HTρ2_ub.rename({'C':'time'})
		return HTρ2_ub	

	#creating and organizing the xarrays

	# #creating and organizing the xarrays

	#unbinned CSHT
	HTρ2_ub =  xr.DataArray(data=HTρ2_ubo, dims={'A','B','C'})
	HTρ2_ub = renamedimsUB(HTρ2_ub,SIGMA2)
	HTρ2_ub.name='Unbinned_Heat_transport'
	HTρ2_ub['time']=SIGMA2.time; HTρ2_ub['contour_index']=SIGMA2.contour_index
	HTρ2_ub['rho2_l']=refsigma
	HTρ2_ub


	#binned CSHT - no ZC
	HTρ2_b =  xr.DataArray(data=HTρ2_bo, dims={'A','B','C'})
	HTρ2_b = renamedimsB(HTρ2_b,SIGMA2)
	HTρ2_b.name='binned_HT'
	HTρ2_b['time']=SIGMA2.time; HTρ2_b['lon_bin_midpoints']=SIGMA2.lon_bin_midpoints
	HTρ2_b['rho2_l']=refsigma


	#binned  ZC
	HTρ2_zc =  xr.DataArray(data=HTρ2_zco, dims={'A','B','C'})
	HTρ2_zc = renamedimsB(HTρ2_zc,SIGMA2)
	HTρ2_zc.name='binned_ZC'
	HTρ2_zc['time']=SIGMA2.time; HTρ2_zc['lon_bin_midpoints']=SIGMA2.lon_bin_midpoints
	HTρ2_zc['rho2_l']=refsigma

	#binned  HT+ZC
	HTρ2_bzc =  xr.DataArray(data=HTρ2_bzco, dims={'A','B','C'})
	HTρ2_bzc = renamedimsB(HTρ2_bzc,SIGMA2)
	HTρ2_bzc.name='binned_HTplusZC'
	HTρ2_bzc['time']=SIGMA2.time; HTρ2_bzc['lon_bin_midpoints']=SIGMA2.lon_bin_midpoints
	HTρ2_bzc['rho2_l']=refsigma

	Dfactor=xr.DataArray(data=factor)
	Dfactor.name = 'Binning_factor'

	#Joining into 1 dataset
	tosave=xr.merge([HTρ2_ub,HTρ2_b,HTρ2_zc,HTρ2_bzc,Dfactor])
	#setting save dir an name
	if exp=='panant-01-zstar-ACCESSyr2': 
		res='01'
	elif exp=='panant-005-zstar-ACCESSyr2': 
		res='005'
	savedir='/home/156/wf4500/v45_wf4500/Project_panan/GH/Panan_HT_ASC/Processed_data/panan' + res +'/CSHT_daily_rho/Ant_cross_slope_heat_terms_offline_1km_'+ start_time+'.nc'
	tosave.to_netcdf(savedir)




	print('Finished successful')
