import numpy as np
import matplotlib.pyplot as plt

t = np.arange(0, 24, 1)

dir1 = '/public/gh/result/0717/t/so2_rmse_modify.txt'

date_file1 = dir1 #+ deal_type + dday + '.txt'
# date_file2 = dir2 + deal_type + dday + '.txt'
# date_file3 = dir3 + deal_type + dday + '.txt'

ss1 = np.loadtxt(date_file1)
ss1 = np.transpose(ss1)


plt.figure(1)


plt.plot(t, ss1[0], 'k-', linewidth=2, label='so2_control')
plt.plot(t, ss1[1], 'c-.', linewidth=2, label='so2_da_NN')
plt.plot(t, ss1[2], 'g-.', linewidth=2, label='so2_no_as')


plt.legend()
plt.xlabel(' Time  (h)', fontsize=20)
plt.ylabel('RMSE Concentration(\u03bcg/m^3)', fontsize=20)


obs_contr = np.abs(ss1[0])
obs_pm = np.abs(ss1[1])
obs_bj = np.abs(ss1[2])


for i in range(1, 23):
    obs_contr[i] = (obs_contr[i-1] + obs_contr[i+1]) / 2
    obs_pm[i] = (obs_pm[i-1] + obs_pm[i+1]) / 2
    obs_bj[i] = (obs_bj[i-1] + obs_bj[i+1]) / 2
    # obs_bet[i] = (obs_bet[i-1] + obs_bet[i+1]) / 2
    # obs_betpm[i] = (obs_betpm[i-1] + obs_betpm[i+1]) / 2

plt.figure(2)

plt.plot(t, obs_contr, 'k-', linewidth=2, label='3D-Var DA')
plt.plot(t, obs_pm, 'r-.', linewidth=2, label='HT-Unet DA')
plt.plot(t, obs_bj, 'g-.', linewidth=2, label='Control')


plt.legend()
plt.xlabel('Time(h)', fontsize=20)
plt.ylabel('RMSE', fontsize=20)

plt.xlim(0, 23)
plt.xticks(np.arange(0, 23, 6))
plt.legend(fontsize=16, loc='upper right')
plt.gca().legend(loc='upper right')

# plt.show()
plt.grid(True)
plt.savefig('/public/gh/result/0717/t/so2' + '_plot_rmse_modify.png', dpi=300)
plt.show()
