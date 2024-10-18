import numpy as np
import matplotlib.pyplot as plt

t = np.arange(0, 24, 1)

dir1 = '/public/gh/result/0717/t/so2_mean_modify.txt'


date_file1 = dir1 #+ deal_type + dday + '.txt'


ss1 = np.loadtxt(date_file1)
ss1 = np.transpose(ss1)


plt.figure(1)
print(t.shape,ss1[0].shape)
# print(ss1[0])

# Plotting the concentration values
plt.plot(t, ss1[0], 'k-', linewidth=2, label='so2_obs')
plt.plot(t, ss1[1], 'c-.', linewidth=2, label='so2_control')
plt.plot(t, ss1[2], 'r-.', linewidth=2, label='so2_da_PM')

plt.legend()
plt.xlabel(' Time (h)', fontsize=20)
plt.ylabel('Mean Concentration(\u03bcg/m^3)', fontsize=20)

# Calculating and plotting absolute differences
obs = ss1[0]
obs_contr = np.abs(ss1[0] - ss1[1])
obs_pm = np.abs(ss1[0] - ss1[2])
# obs_bet = np.abs(ss2[0] - ss2[2])
# obs_betpm = np.abs(ss3[0] - ss3[2])

for i in range(1, 23):
    obs_contr[i] = (obs_contr[i-1] + obs_contr[i+1]) / 2
    obs_pm[i] = (obs_pm[i-1] + obs_pm[i+1]) / 2
    # obs_bet[i] = (obs_bet[i-1] + obs_bet[i+1]) / 2
    # obs_betpm[i] = (obs_betpm[i-1] + obs_betpm[i+1]) / 2

plt.figure(2)

plt.plot(t, obs_contr, 'k-', linewidth=2, label='Obs-Control')
plt.plot(t, obs_pm, 'r-.', linewidth=2, label='Obs-Da_so2')


plt.legend()
plt.xlabel('Forecast duration (h)', fontsize=20)
plt.ylabel('Mean Concentration(\u03bcg/m^3)', fontsize=20)

plt.xlim(0, 24)
plt.xticks(np.arange(0, 25, 6))
plt.legend(fontsize=16, loc='upper right')
plt.gca().legend(loc='upper right')

# plt.show()
plt.grid(True)
plt.savefig('/public/gh/result/0717/t/so2' + '_plot_mean_modify.png', dpi=300)
plt.show()
