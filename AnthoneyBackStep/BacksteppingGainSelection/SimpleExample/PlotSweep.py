import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# Loading data from sweep run
sweep_info = pd.read_csv('sweep/evaluations_low_res.csv')
X = sweep_info['k1'].to_numpy()
Y = sweep_info['k2'].to_numpy()
N = len(np.unique(X))
M = len(np.unique(Y))
X_plt = np.reshape(X, (M,N))
Y_plt = np.reshape(Y, (N,M))

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
    })

# Let's plot position stuff
gs1 = mpl.gridspec.GridSpec(2, 2)
fig_pos = plt.figure()
fig_pos.suptitle('Position Expected Metrics', fontsize=20)
ax1_pos = fig_pos.add_subplot(gs1[0,0])
ax1_ctr_pos = ax1_pos.contourf(X_plt, Y_plt, np.reshape(sweep_info['E_Ts_x1'].to_numpy(), ((N,M))), levels=20)
ax1_pos.set_xlabel('$K_1$', fontsize=16)
ax1_pos.set_ylabel('$K_2$', fontsize=16)
ax1_pos.set_title('Settling Time (5\%)', fontsize=16)
cbar1_pos = fig_pos.colorbar(ax1_ctr_pos, ax=ax1_pos)
cbar1_pos.set_label('$T_s$ (s)', fontsize=12)

ax2_pos = fig_pos.add_subplot(gs1[0,1])
ax2_ctr_pos = ax2_pos.contourf(X_plt, Y_plt, np.reshape(sweep_info['E_Tr_x1'].to_numpy(), ((N,M))), levels=20)
ax2_pos.set_xlabel('$K_1$', fontsize=16)
ax2_pos.set_ylabel('$K_2$', fontsize=16)
ax2_pos.set_title('Rise Time (5\%)', fontsize=16)
cbar2_pos = fig_pos.colorbar(ax2_ctr_pos, ax=ax2_pos)
cbar2_pos.set_label('$T_r$ (s)', fontsize=12)

ax3_pos = fig_pos.add_subplot(gs1[1,0])
ax3_ctr_pos = ax3_pos.contourf(X_plt, Y_plt, np.reshape(sweep_info['E_OS_x1'].to_numpy(), ((N,M))), levels=20)
ax3_pos.set_xlabel('$K_1$', fontsize=16)
ax3_pos.set_ylabel('$K_2$', fontsize=16)
ax3_pos.set_title('Overshoot', fontsize=16)
cbar3_pos = fig_pos.colorbar(ax3_ctr_pos, ax=ax3_pos)
cbar3_pos.set_label('$OS$ (m)', fontsize=12)

ax4_pos = fig_pos.add_subplot(gs1[1,1])
ax4_ctr_pos = ax4_pos.contourf(X_plt, Y_plt, np.reshape(sweep_info['E_Osc_x1'].to_numpy(), ((N,M))), levels=20)
ax4_pos.set_xlabel('$K_1$', fontsize=16)
ax4_pos.set_ylabel('$K_2$', fontsize=16)
ax4_pos.set_title('Oscillation Count', fontsize=16)
cbar4_pos = fig_pos.colorbar(ax4_ctr_pos, ax=ax4_pos)
cbar4_pos.set_label('$Osc$', fontsize=12)

plt.show()

# Let's plot velocity stuff
fig_vel = plt.figure()
fig_vel.suptitle('Velocity Expected Metrics', fontsize=20)
ax1_vel = fig_vel.add_subplot(gs1[0,0])
ax1_ctr_vel = ax1_vel.contourf(X_plt, Y_plt, np.reshape(sweep_info['E_Ts_x2'].to_numpy(), ((N,M))), levels=20)
ax1_vel.set_xlabel('$K_1$', fontsize=16)
ax1_vel.set_ylabel('$K_2$', fontsize=16)
ax1_vel.set_title('Settling Time (5\%)', fontsize=16)
cbar1_vel = fig_vel.colorbar(ax1_ctr_vel, ax=ax1_vel)
cbar1_vel.set_label('$T_s$ (s)', fontsize=12)

ax2_vel = fig_vel.add_subplot(gs1[0,1])
ax2_ctr_vel = ax2_vel.contourf(X_plt, Y_plt, np.reshape(sweep_info['E_Tr_x2'].to_numpy(), ((N,M))), levels=20)
ax2_vel.set_xlabel('$K_1$', fontsize=16)
ax2_vel.set_ylabel('$K_2$', fontsize=16)
ax2_vel.set_title('Rise Time (5\%)', fontsize=16)
cbar2_vel = fig_vel.colorbar(ax2_ctr_vel, ax=ax2_vel)
cbar2_vel.set_label('$T_r$ (s)', fontsize=12)

ax3_vel = fig_vel.add_subplot(gs1[1,0])
ax3_ctr_vel = ax3_vel.contourf(X_plt, Y_plt, np.reshape(sweep_info['E_OS_x2'].to_numpy(), ((N,M))), levels=20)
ax3_vel.set_xlabel('$K_1$', fontsize=16)
ax3_vel.set_ylabel('$K_2$', fontsize=16)
ax3_vel.set_title('Overshoot', fontsize=16)
cbar3_vel = fig_vel.colorbar(ax3_ctr_vel, ax=ax3_vel)
cbar3_vel.set_label('$OS$ ($\\frac{m}{s}$)', fontsize=12)

ax4_vel = fig_vel.add_subplot(gs1[1,1])
ax4_ctr_vel = ax4_vel.contourf(X_plt, Y_plt, np.reshape(sweep_info['E_Osc_x2'].to_numpy(), ((N,M))), levels=20)
ax4_vel.set_xlabel('$K_1$', fontsize=16)
ax4_vel.set_ylabel('$K_2$', fontsize=16)
ax4_vel.set_title('Oscillation Count', fontsize=16)
cbar4_vel = fig_vel.colorbar(ax4_ctr_vel, ax=ax4_vel)
cbar4_vel.set_label('$Osc$', fontsize=12)

plt.show()

# Let's plot control stuff
gs2 = mpl.gridspec.GridSpec(2, 1)
fig_ctr = plt.figure()
fig_ctr.suptitle('Control Expected Metrics', fontsize=20)
ax1_ctr = fig_ctr.add_subplot(gs2[0])
ax1_ctr_ctr = ax1_ctr.contourf(X_plt, Y_plt, np.reshape(sweep_info['E_Up'].to_numpy(), ((N,M))), levels=20)
ax1_ctr.set_xlabel('$K_1$', fontsize=16)
ax1_ctr.set_ylabel('$K_2$', fontsize=16)
ax1_ctr.set_title('Peak Control', fontsize=16)
cbar1_ctr = fig_ctr.colorbar(ax1_ctr_ctr, ax=ax1_ctr)
cbar1_ctr.set_label('$U_p$ ($N$)', fontsize=12)

ax2_ctr = fig_ctr.add_subplot(gs2[1])
ax2_ctr_ctr = ax2_ctr.contourf(X_plt, Y_plt, np.reshape(sweep_info['E_Ui'].to_numpy(), ((N,M))), levels=20)
ax2_ctr.set_xlabel('$K_1$', fontsize=16)
ax2_ctr.set_ylabel('$K_2$', fontsize=16)
ax2_ctr.set_title('Integral of Control', fontsize=16)
cbar2_ctr = fig_ctr.colorbar(ax2_ctr_ctr, ax=ax2_ctr)
cbar2_ctr.set_label('$U_i$ ($N^2$)', fontsize=12)

plt.show()

# Let's plot the cost function now
gs = mpl.gridspec.GridSpec(1, 1)
fig = plt.figure()
fig.suptitle('Total Cost Function', fontsize=20)
ax1 = fig.add_subplot(gs[0])
ax1_ctr = ax1.contourf(X_plt, Y_plt, np.reshape(sweep_info['J'].to_numpy(), ((N,M))), levels=40)
ax1.set_xlabel('$K_1$', fontsize=16)
ax1.set_ylabel('$K_2$', fontsize=16)
cbar1 = fig.colorbar(ax1_ctr, ax=ax1)
cbar1.set_label('$J$', fontsize=12)
plt.show()
