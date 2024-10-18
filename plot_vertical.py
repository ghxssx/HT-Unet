import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from netCDF4 import Dataset
import matplotlib

# 定义文件路径
filename1 = r'/public/gh/chang/increase/increase_3Dvar.nc'
filename0 = r'/public/gh/chang/increase/increase_egeunet.nc'
filename2 = r'/public12/gh/wrfout/20230206/wrfrst_d01_2023-02-06_00:00:00'

# 读取数据
ncgrib = Dataset(filename1, 'r')
ncgrib0 = Dataset(filename0, 'r')
wrfoutgrib = Dataset(filename2, 'r')

# 提取经纬度信息
lati = np.squeeze(wrfoutgrib.variables['XLAT'][:])
Mlati, Nlati = lati.shape
longti = np.squeeze(wrfoutgrib.variables['XLONG'][:])
Mlongti, Nlongti = longti.shape

# 提取数据
data_00 = ncgrib.variables['SO2'][:]
data_000 = ncgrib0.variables['data'][:]
# data_00 = (data_00-data_000)
# data_00 = data_00 *64064 / 22.414
# print(data_00)
data1 = np.squeeze(data_00[0, :Mlati, :Nlati])
data11 = np.squeeze(data_000[0, :Mlati, :Nlati])
Mmax, Nmax = np.unravel_index(data1.argmax(), data1.shape)
Mmax0, Nmax0 = np.unravel_index(data11.argmax(), data11.shape)

ph = np.squeeze(wrfoutgrib.variables['PH'][0, :, Mmax, Nmax])
phb = np.squeeze(wrfoutgrib.variables['PHB'][0, :, Mmax, Nmax])
height = (ph + phb) / 9.81
height = height[:34]

long_contour = np.squeeze(longti[Mmax, :Nlati])

kheight = 10
hgt1 = np.arange(Nlati)
hgt2 = np.arange(kheight)

SO2 = np.squeeze(data_00[:, Mmax, :Nlati]) * 0.0
SO22 = np.squeeze(data_000[:, Mmax, :Nlati]) * 0.0
# datalinshi = ncgrib.variables['data'][:, Mmax, :Nlati]
datalinshi = ncgrib.variables['SO2'][:, Mmax, :Nlati]
datalinshi0 = ncgrib0.variables['data'][:, Mmax, :Nlati]
value = np.squeeze(datalinshi)[:kheight, :]
value0 = np.squeeze(datalinshi0)[:kheight, :]

# 定义红蓝色映射
colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]  # 蓝色 -> 白色 -> 红色
cmap = LinearSegmentedColormap.from_list('red_blue_cmap', colors, N=256)

# 绘制图形
plt.figure(1)
contour_value = value-value0
contour_num = 12
# contour_level = np.linspace(-0.00001, 0.00001, 10000) 
contour_level = np.linspace(-5, 5,21)
# contour_level = [-10,0,10,15,18,21,24,27,30,40,45,50]
plt.contourf(hgt1, hgt2, contour_value, levels=contour_level, cmap=cmap)

# plt.contourf(hgt1, hgt2, contour_value, cmap=cmap)
cbar = plt.colorbar(location='bottom')
# cbar.set_label('SO2' + ' ($\mu g/m^3$)', fontsize=20)
plt.xlabel('Grid num', fontsize=20)
plt.ylabel('Level', fontsize=20)
# plt.title('egeunet ' + 'SO2' + ' increase', fontsize=20)
matplotlib.rcParams['font.family'] = 'Times New Roman'
plt.savefig('/public/gh/chang/increase/' + '_poumiantu_3dvar827_cha_911.png', dpi=300)
plt.show()
