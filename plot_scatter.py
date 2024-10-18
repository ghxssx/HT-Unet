import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np
# 读取第一个nc文件
file1 = '/public/gh/chang/pre_test_nc/wrfpre_d01_2023-11-19_18:00:00.nc'
data1 = nc.Dataset(file1)

z1 = data1.variables['data'][:]  # 获取变量'z'的值

# 读取第二个nc文件
file2 = '/public/gh/result/wrfinput_d01/20231119/wrfrst_d01_2023-11-19_18:00:00.nc'
data2 = nc.Dataset(file2)
# x2 = data2.variables['x'][:]  # 获取变量'x'的值
# y2 = data2.variables['y'][:]  # 获取变量'y'的值
z2 = data2.variables['SO2'][:]  # 获取变量'z'的值


min_val = np.min([z1.min(), z2.min()])
max_val = np.max([z1.max(), z2.max()])


# 绘制散点图
fig, ax = plt.subplots()
ax.scatter(z2, z1, s=1)  # 绘制散点图
ax.set_xlabel('3Dvar')  # 设置x轴标签
ax.set_ylabel('deep learn assimilation')  # 设置y轴标签
# ax.plot([0, 0], [90, 90], color='red')  # 绘制x=y的直线
# ax.plot([min(z1), max(z1)], [min(z1), max(z1)], color='red', label='Slope = 1')
ax.plot([min_val, max_val], [min_val, max_val], color='red', label='Slope = 1')  # 绘制斜率为1的直线
# ax.set_title('散点图')  # 设置标题
ax.grid(True)
output_file = '/public/gh/result/scatter_plot.png'
plt.savefig(output_file, dpi=300)

print("操作完成，结果已保存到", output_file)
