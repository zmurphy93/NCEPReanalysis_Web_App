# -*- coding: utf-8 -*-
"""
@author: zmurp
"""
from datetime import datetime
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import metpy.calc as mpcalc
import netCDF4 as nc
import numpy as np
from metpy.units import units


# =============================================================================
# USER-DEFINED VARIABLES: LAT/LON OF THE REGION TO STUDY. DATE OF THE CASE STUDY.
# =============================================================================
WLON = 200
ELON = 315
SLAT = 10
NLAT = 60

YR = 2008
MON = 12
DAY = 12
HR = 6
# =============================================================================
# Choose from: NCEP, CPC, NARR, 20TH_CENTURY
# =============================================================================
DATASET = 'NCEP'
DATE = datetime(int(YR), MON, DAY, HR)

PRES_DATA = []
SLP_DATAS = []
SFC_DATA = []
BASE_URL = 'http://www.esrl.noaa.gov/psd/thredds/dodsC/Datasets/ncep.reanalysis/'
PRES_VAR = ['air', 'hgt', 'rhum', 'shum', 'uwnd', 'vwnd']
SLP_VAR = ['slp', 'pr_wtr.eatm']
SFC_VAR = ['air.2m.gauss', 'pres.sfc.gauss','shum.2m.gauss','uwnd.10m.gauss','vwnd.10m.gauss' ]

for item in PRES_VAR:
    PRES_NNR = BASE_URL+'pressure/PRES_VAR.YEAR.nc'
    PRES_NNR = PRES_NNR.replace('YEAR', str(YR))
    PRES_NNR = PRES_NNR.replace('PRES_VAR', item)
    PRES_DATA.append(PRES_NNR)

for item in SLP_VAR:
    SLP_NNR = BASE_URL+'surface/PRES_VAR.YEAR.nc'
    SLP_NNR = SLP_NNR.replace('YEAR', str(YR))
    SLP_NNR = SLP_NNR.replace('PRES_VAR', item)
    SLP_DATAS.append(SLP_NNR)

for item in SFC_VAR:
    SFC_NNR = BASE_URL+'surface_gauss/PRES_VAR.YEAR.nc'
    SFC_NNR = SFC_NNR.replace('YEAR', str(YR))
    SFC_NNR = SFC_NNR.replace('PRES_VAR', item)
    SFC_DATA.append(SFC_NNR)

# =============================================================================
# Retrieve data
# =============================================================================
AIR_DATA = nc.Dataset(PRES_DATA[0])
HGT_DATA = nc.Dataset(PRES_DATA[1])
HUM_DATA = nc.Dataset(PRES_DATA[2])
SHUM_DATA = nc.Dataset(PRES_DATA[3])
UWND_DATA = nc.Dataset(PRES_DATA[4])
VWND_DATA = nc.Dataset(PRES_DATA[5])

SLP_DATA = nc.Dataset(SLP_DATAS[0])
PWAT_DATA= nc.Dataset(SLP_DATAS[1])

T2M_DATA = nc.Dataset(SFC_DATA[0])
PRES_DATA = nc.Dataset(SFC_DATA[1])
SH2M_DATA = nc.Dataset(SFC_DATA[2])
U2M_DATA = nc.Dataset(SFC_DATA[3])
V2M_DATA = nc.Dataset(SFC_DATA[4])
# =============================================================================
# Preliminary data cleaning (Setting Time, Using Lat Lon Grids, ETC.)
# =============================================================================
AIR_TIME_VAR = AIR_DATA.variables['time']
TIME_INDEX = nc.date2index(DATE,AIR_TIME_VAR)
LAT = AIR_DATA['lat'][:]
LON = AIR_DATA['lon'][:]
SFLAT = T2M_DATA['lat'][:]
SFLON = T2M_DATA['lon'][:]
DX, DY = mpcalc.lat_lon_grid_spacing(LON, LAT)
# =============================================================================
# Projection
# =============================================================================
crs = ccrs.Miller(central_longitude=-95)

def plot_background(ax):
    ax.set_extent([WLON, ELON, SLAT, NLAT])
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.5)
    ax.add_feature(cfeature.LAKES.with_scale('50m'), linewidth=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    return ax
# =============================================================================
# FIG #1: 250: JET STREAM, GEOPOTENTIAL HEIGHT, DIVERGENCE
# =============================================================================
H250 = HGT_DATA.variables['hgt'][TIME_INDEX, 8, :, :]
U250 = UWND_DATA.variables['uwnd'][TIME_INDEX, 8, :, :]*units('m/s')
V250 = VWND_DATA.variables['vwnd'][TIME_INDEX, 8, :, :]*units('m/s')
SPEED250 = mpcalc.get_wind_speed(U250, V250)
DIV250 = mpcalc.divergence(U250, V250, DX, DY, dim_order='YX')
DIV250 = (DIV250*(units('1/s')))
# =============================================================================
# FIG #2: 500: VORTICITY, GEOPOTENTIAL HEIGHT, VORTICITY ADVECTION
# =============================================================================
H500 = HGT_DATA.variables['hgt'][TIME_INDEX, 5, :, :]
U500 = UWND_DATA.variables['uwnd'][TIME_INDEX, 5, :, :]*units('m/s')
V500 = VWND_DATA.variables['vwnd'][TIME_INDEX, 5, :, :]*units('m/s')
DX, DY = mpcalc.lat_lon_grid_spacing(LON, LAT)
VORT500 = mpcalc.vorticity(U500, V500, DX, DY, dim_order='YX')
VORT500 = (VORT500*(units('1/s')))
VORT_ADV500 = mpcalc.advection(VORT500, [U500, V500], (DX, DY), dim_order='yx')
# =============================================================================
# FIG #3: 700: Q-VECTORS+CONVERGENCE, GEOPOTENTIAL HEIGHT
# =============================================================================
H700 = HGT_DATA.variables['hgt'][TIME_INDEX, 3, :, :]
T700 = AIR_DATA.variables['air'][TIME_INDEX, 3, :, :]*units('kelvin')
U700 = UWND_DATA.variables['uwnd'][TIME_INDEX, 3, :, :]*units('m/s')
V700 = VWND_DATA.variables['vwnd'][TIME_INDEX, 3, :, :]*units('m/s')
PWAT = PWAT_DATA.variables['pr_wtr'][TIME_INDEX, :, :]
QVEC700 = mpcalc.q_vector(U700, V700, T700, 700*units.mbar, DX, DY)
QC700 = mpcalc.divergence(QVEC700[0], QVEC700[1], DX, DY, dim_order = 'yx')*(10**18)
QVECX = QVEC700[0]
QVECY = QVEC700[1]
# =============================================================================
# FIG #4: 850: GEOPOTENTIAL HEIGHT, TEMP, WINDS, TEMP-ADVECTION, FRONTOGENESIS
# =============================================================================
H850 = HGT_DATA.variables['hgt'][TIME_INDEX, 2, :, :]
T850 = AIR_DATA.variables['air'][TIME_INDEX, 2, :, :]*units('kelvin')
U850 = UWND_DATA.variables['uwnd'][TIME_INDEX, 2, :, :]*units('m/s')
V850 = VWND_DATA.variables['vwnd'][TIME_INDEX, 2, :, :]*units('m/s')
T_ADV850 = mpcalc.advection(T850 * units.kelvin, [U850, V850], (DX, DY), dim_order='yx') * units('K/sec')
PT850 = mpcalc.potential_temperature(850*units.mbar, T850)
FRONT_850 = mpcalc.frontogenesis(PT850, U850, V850, DX, DY, dim_order = 'YX')
# =============================================================================
# FIG #5: 850: GEOPOTENTIAL HEIGHT, EQUIV. POT. TEMP, WINDS, LAPSE RATES
# =============================================================================
H500 = HGT_DATA.variables['hgt'][TIME_INDEX, 5, :, :]
H700 = HGT_DATA.variables['hgt'][TIME_INDEX, 3, :, :]
T500 = AIR_DATA.variables['air'][TIME_INDEX, 5, :, :]*units('degC')
T700 = AIR_DATA.variables['air'][TIME_INDEX, 3, :, :]*units('degC')
LR = -1000*(T500-T700)/(H500-H700)
H850 = HGT_DATA.variables['hgt'][TIME_INDEX, 2, :, :]
T850 = AIR_DATA.variables['air'][TIME_INDEX, 2, :, :]*units('kelvin')
SH850 = SHUM_DATA.variables['shum'][TIME_INDEX, 2, :, :]
DP850 = mpcalc.dewpoint_from_specific_humidity(SH850, T850, 850*units.mbar)
EPT850 = mpcalc.equivalent_potential_temperature(850*units.mbar, T850, DP850)
# =============================================================================
# FIG #6: 925: MOISTURE FLUX, MOISTURE FLUX CONVERGENCE,
# =============================================================================
RH925 = HUM_DATA.variables['rhum'][TIME_INDEX, 1, :, :]
SH925 = SHUM_DATA.variables['shum'][TIME_INDEX, 1, :, :]
U925 = UWND_DATA.variables['uwnd'][TIME_INDEX, 1, :, :]*units('m/s')
V925 = VWND_DATA.variables['vwnd'][TIME_INDEX, 1, :, :]*units('m/s')
H925 = HGT_DATA.variables['hgt'][TIME_INDEX, 1, :, :]
SH_ADV925 = mpcalc.advection(SH925, [U925, V925], (DX, DY), dim_order='yx')
SH_DIV925 = SH925*(mpcalc.divergence(U925, V925, DX, DY, dim_order='YX'))
MFLUXX = SH925*U925
MFLUXY = SH925*V925
MFC_925 = SH_ADV925+SH_DIV925
# =============================================================================
# FIG #7: SFC: MSLP, WIND, TEMPERATURE
# =============================================================================
SLP = SLP_DATA.variables['slp'][TIME_INDEX, :, :]*units('hPa')
T2M = T2M_DATA.variables['air'][TIME_INDEX, :, :]*units('kelvin')
U2M = U2M_DATA.variables['uwnd'][TIME_INDEX, :, :]*units('m/s')
V2M = V2M_DATA.variables['vwnd'][TIME_INDEX, :, :]*units('m/s')
# =============================================================================
# FIG #8: SFC: HEAT INDEX OR WINDCHILL
# =============================================================================
T2M = T2M_DATA.variables['air'][TIME_INDEX, :, :]*units('kelvin')
SH2M = SH2M_DATA.variables['shum'][TIME_INDEX, :, :]
U2M = U2M_DATA.variables['uwnd'][TIME_INDEX, :, :]*units('m/s')
V2M = V2M_DATA.variables['vwnd'][TIME_INDEX, :, :]*units('m/s')
PRES = PRES_DATA.variables['pres'][TIME_INDEX, :, :]*units('Pa')
RHUM = mpcalc.relative_humidity_from_specific_humidity(SH2M, T2M, PRES)
T2M = T2M.to('degF')
SFC_SPEED = mpcalc.get_wind_speed(U2M, V2M)
SFC_SPEED = SFC_SPEED.to('mph')
APPARENT_TEMP = mpcalc.apparent_temperature(T2M, RHUM, SFC_SPEED)
# =============================================================================
# =============================================================================
# =============================================================================
# Make a grid of lat/lon values to use for plotting with Basemap.
lons, lats = np.meshgrid(np.squeeze(LON), np.squeeze(LAT))
slons, slats = np.meshgrid(np.squeeze(SFLON), np.squeeze(SFLAT))
fig, axarr = plt.subplots(nrows=2, ncols=4, figsize=(35, 25), subplot_kw={'projection': crs})
axlist = axarr.flatten()
fig.tight_layout()
for ax in axlist:
    plot_background(ax)
# =============================================================================
# FIG 1
# =============================================================================
clevs250wnd = np.arange(30, 110, 10)
clevsH250 = np.arange(9000, 12000, 100)
clevs250div = np.arange (2, 22, 2)
cf1 = axlist[0].contourf(lons, lats, np.squeeze(SPEED250), clevs250wnd, cmap='BuPu', transform=ccrs.PlateCarree())
c1a = axlist[0].contour(lons, lats, np.squeeze(H250), clevsH250, colors='black', linewidths=2,transform=ccrs.PlateCarree())
c1c = axlist[0].contour(lons, lats, (np.squeeze(DIV250)*10**5), clevs250div, colors='darkorange', linestyles = '-', linewidths=3,transform=ccrs.PlateCarree())
axlist[0].barbs(lons, lats, U250.magnitude, V250.magnitude, length=6, regrid_shape=20, pivot='middle', transform=ccrs.PlateCarree())
axlist[0].clabel(c1a, fontsize=10, inline=1, inline_spacing=1, fmt='%i', rightside_up=True)
axlist[0].set_title('250-hPa Wind Speeds, Divergence, & Heights', fontsize=16)
cb1 = fig.colorbar(cf1, ax=axlist[0], orientation='horizontal', shrink=1.0,  pad=0.0)
cb1.set_label('Wind Speeds (m/s)', size='x-large')
# =============================================================================
# FIG 2
# =============================================================================
clevs500vort = np.arange(0.5, 10.5, 0.5)
clevs500vortadv = np.arange(1, 11, 1)
clevsH500 = np.arange(4500, 6400, 100)
cf2 = axlist[1].contourf(lons, lats, np.squeeze(VORT500*(10**5)), clevs500vort, cmap='YlOrBr', transform=ccrs.PlateCarree())
c2a = axlist[1].contour(lons, lats, np.squeeze(H500),clevsH500, colors='black', linewidths=2,transform=ccrs.PlateCarree())
c2b = axlist[1].contour(lons, lats, (np.squeeze(VORT_ADV500)*10**9), clevs500vortadv, colors='maroon', linewidths=3,transform=ccrs.PlateCarree())
axlist[1].barbs(lons, lats, U500.magnitude, V500.magnitude, length=6, regrid_shape=20, pivot='middle', transform=ccrs.PlateCarree())
axlist[1].clabel(c2a, fontsize=10, inline=1, inline_spacing=1, fmt='%i', rightside_up=True)
axlist[1].set_title('500-hPa, Vort., Vort Advection, Wind Speeds & Heights', fontsize=16)
cb2 = fig.colorbar(cf2, ax=axlist[1], orientation='horizontal', shrink=1.0, pad=0)
cb2.set_label('Rel. Vorticity (1/s)', size='x-large')
# =============================================================================
# FIG 3
# =============================================================================
clevsPwat = np.arange(0, 65, 5)
clevsQCpos = np.arange(-30, 0, 3)
clevsQCneg = np.arange(3, 30, 3)
clevsH700 = np.arange(2500, 3500, 100)
cf3 = axlist[2].contourf(lons, lats, np.squeeze(PWAT), clevsPwat, cmap='terrain_r', transform=ccrs.PlateCarree())
c3a = axlist[2].contour(lons, lats, np.squeeze(QC700), clevsQCpos, colors='green', linestyles = '-', transform=ccrs.PlateCarree())
c3b = axlist[2].contour(lons, lats, np.squeeze(QC700), clevsQCneg, colors='indigo', linestyles='-', transform=ccrs.PlateCarree())
c3c = axlist[2].contour(lons, lats, np.squeeze(H700), colors='black', linewidths=2,transform=ccrs.PlateCarree())
axlist[2].clabel(c3c, fontsize=10, inline=1, inline_spacing=1, fmt='%i', rightside_up=True)
QVector = axlist[2].quiver(lons, lats, QVECX.magnitude, QVECY.magnitude, regrid_shape=20, pivot='middle', color='black', transform=ccrs.PlateCarree())
axlist[2].set_title('700-hPa Q-Vectors, Q-Vector Convergence', fontsize=16)
cb3 = fig.colorbar(cf3, ax=axlist[2], orientation='horizontal', shrink=1.0, pad=0)
cb3.set_label('Precipitable Water (mm)', size='x-large')
# =============================================================================
# FIG 4
# =============================================================================
clevsT850 = np.arange(230, 304, 4)
clevsTA850 = np.arange(-15, 16, 1)
clevsFront850 = np.arange(2, 20, 2)
cf4 = axlist[3].contourf(lons, lats, np.squeeze(T_ADV850*(10**4)), clevsTA850, cmap='bwr_r', transform=ccrs.PlateCarree())
c4 = axlist[3].contour(lons, lats, np.squeeze(T850), clevsT850, colors='gray', linewidths=2,transform=ccrs.PlateCarree())
c4a = axlist[3].contour(lons, lats, np.squeeze(FRONT_850*(10**10)), clevsFront850, colors='darkorchid', linewidths=2,transform=ccrs.PlateCarree())
axlist[3].barbs(lons, lats, U850.magnitude, V850.magnitude, length=6, regrid_shape=20, pivot='middle', transform=ccrs.PlateCarree())
axlist[3].clabel(c4, fontsize=10, inline=1, inline_spacing=1, fmt='%i', rightside_up=True)
axlist[3].set_title('850-hPa, Temp, Advection, Frontogenesis, Winds', fontsize=16)
cb4 = fig.colorbar(cf4, ax=axlist[3], orientation='horizontal', shrink=1.0, pad=0)
cb4.set_label('Temp Advection (C/s)', size='x-large')
# =============================================================================
# FIG 5
# =============================================================================
clevsEPT = np.arange(230, 370, 5)
clevsH850 = np.arange(900, 2100, 50)
clevsLR = np.arange(6, 12.5, 0.5)
cf5 = axlist[4].contourf(lons, lats, np.squeeze(EPT850), clevsEPT, cmap='gnuplot', transform=ccrs.PlateCarree())
c5a = axlist[4].contour(lons, lats, np.squeeze(H850), clevsH850, colors='black', linewidths=2,transform=ccrs.PlateCarree())
c5b = axlist[4].contour(lons, lats, np.squeeze(LR), clevsLR, colors='aqua', linewidths=2,transform=ccrs.PlateCarree())
axlist[4].barbs(lons, lats, U850.magnitude, V850.magnitude, length=6, regrid_shape=20, pivot='middle', transform=ccrs.PlateCarree())
axlist[4].clabel(c5a, fontsize=10, inline=1, inline_spacing=1, fmt='%i', rightside_up=True)
axlist[4].clabel(c5b, fontsize=10, inline=1, inline_spacing=1, fmt='%i', rightside_up=True)
axlist[4].set_title('850-hPa Theta-E, 700-500-hPa Lapse Rates, Heights', fontsize=16)
cb5 = fig.colorbar(cf5, ax=axlist[4], orientation='horizontal', shrink=1.0, pad=0)
cb5.set_label('Theta-E(Kelvin)', size='x-large')
# =============================================================================
# FIG 6
# =============================================================================
clevsRH = np.arange(50, 110, 5)
clevsH925 = np.arange(450, 1550, 50)
clevsMFC = np.arange(1, 11, 0.5)
cf6 = axlist[5].contourf(lons, lats, (np.squeeze(RH925)), clevsRH, cmap='YlGnBu', transform=ccrs.PlateCarree())
c5a = axlist[5].contour(lons, lats, np.squeeze(H925), clevsH925, colors='black', linewidths=2,transform=ccrs.PlateCarree())
c5b = axlist[5].contour(lons, lats, (np.squeeze(MFC_925)*10**7), clevsMFC, colors='mediumvioletred', linewidths=2,transform=ccrs.PlateCarree())
MFlux = axlist[5].quiver(lons, lats, MFLUXX.magnitude, MFLUXY.magnitude, scale=3, regrid_shape=20, pivot='middle', color='black', transform=ccrs.PlateCarree())
axlist[5].clabel(c5a, fontsize=10, inline=1, inline_spacing=1, fmt='%i', rightside_up=True)
axlist[5].set_title('925-hPa Rel. Hum., Moisture Flux, and MF Convergence', fontsize=16)
cb6 = fig.colorbar(cf6, ax=axlist[5], orientation='horizontal', shrink=1.0, pad=0)
cb6.set_label('RH (%)', size='x-large')
# =============================================================================
# FIG 7
# =============================================================================
clevsSLP = np.arange(960, 1070, 5)
clevsT = np.arange(-60, 125, 5)
cf7 = axlist[6].contourf(slons, slats, np.squeeze(T2M), clevsT, cmap='nipy_spectral', transform=ccrs.PlateCarree())
c7a = axlist[6].contour(lons, lats, np.squeeze(SLP/100), clevsSLP, colors='black', linewidths=2,transform=ccrs.PlateCarree())
axlist[6].barbs(slons, slats, U2M.magnitude, V2M.magnitude, length=6, regrid_shape=20, pivot='middle', transform=ccrs.PlateCarree())
axlist[6].clabel(c7a, fontsize=10, inline=1, inline_spacing=1, fmt='%i', rightside_up=True)
axlist[6].set_title('Sea-Level Pressure, 2-M Temperature, 10-M Winds', fontsize=16)
cb7 = fig.colorbar(cf7, ax=axlist[6], orientation='horizontal', shrink=1.0, pad=0)
cb7.set_label('2-Meter Temp (Degrees (F)) ', size='x-large')
# =============================================================================
# FIG 8
# =============================================================================
clevsAT = np.arange(-70, 135, 5)
cf8 = axlist[7].contourf(slons, slats, np.squeeze(APPARENT_TEMP), clevsAT, cmap='gist_rainbow_r', transform=ccrs.PlateCarree())
axlist[7].set_title('2-M Apparent Temperature', fontsize=16)
cb8 = fig.colorbar(cf8, ax=axlist[7], orientation='horizontal', shrink=1.0, pad=0)
cb8.set_label('Heat Index/Wind Chill (Degrees (F))', size='x-large')
# =============================================================================
# FINAL PLOTTING SECTION.
# =============================================================================
# Set figure title
fig.suptitle('Date: '+str(YR)+'-'+str(MON)+'-'+str(DAY)+' '+str(HR)+':00UTC', fontsize=32)
# Display the plot
plt.show()
