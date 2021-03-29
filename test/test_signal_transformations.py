import numpy as np
import math
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1] / 'src/'))
import signal_transformation as sgtf

## transformation task params
noise_param = 15 #noise_amount
scale_param = 1.1 #scaling_factor
permu_param = 20 #permutation_pieces
tw_piece_param = 9 #time_warping_pieces
twsf_param = 1.05 #time_warping_stretch_factor

T = 10
t = np.linspace(0, T, 2560)
x = np.load(Path(__file__).parents[1] / 'data/sample_ecg.npy')
x = x[0, :len(t)].reshape(-1,1)

plt.plot(t, x)
plt.title('Original signal')
# x_noise = sgtf.add_noise(x, 0.3)
# plt.plot(t, x_noise)

x_noise = sgtf.add_noise_with_SNR(x, noise_param=20)
plt.figure()
plt.plot(t, x)
plt.plot(t, x_noise)
plt.title('Signal with Gaussian noise addition')

x_scaled = sgtf.scaled(x, scale_param)
plt.figure()
plt.plot(t, x)
plt.plot(t, x_scaled)
plt.title('Scaled signal')

x_negated = sgtf.negate(x)
plt.figure()
plt.plot(t, x)
plt.plot(t, x_negated)
plt.title('Negated signal')

x_fliped   = sgtf.hor_filp(x)
plt.figure()
plt.plot(t, x)
plt.plot(t, x_fliped)
plt.title('Flipped signal')

x_perm = sgtf.permute(x, permu_param)
plt.figure()
plt.plot(t, x)
plt.plot(t, x_perm)
plt.title('Randomly permuted signal')

x_tw = sgtf.time_warp(x, permu_param, twsf_param, 1/twsf_param)

plt.figure()
plt.plot(t, x)
plt.plot(t,x_tw)
plt.title('Warped signal')

plt.show()