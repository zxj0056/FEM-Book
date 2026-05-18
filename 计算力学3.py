import warnings
warnings.filterwarnings("ignore")

import math
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FuncFormatter


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def compute_pi(n):
    theta = 2 * math.pi / n
    side_length = 2 * math.sin(theta / 2)
    perimeter = n * side_length
    return perimeter / 2


n_list = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
pi_true = math.pi
pi_approx = [compute_pi(n) for n in n_list]
error_actual = [abs(p - pi_true) for p in pi_approx]


error_ref = [0.5 / n for n in n_list]


plt.figure(figsize=(10,6), dpi=120)


plt.plot(n_list, error_actual, 'o-', color='#1f77b4',
         linewidth=2, markersize=5, label='实际收敛误差')


plt.plot(n_list, error_ref, '--', color='#d62728',
         linewidth=2, label='理论1阶收敛参考线(斜率=-1)')


plt.xscale('log', base=10)
plt.yscale('log', base=10)
ax = plt.gca()


def log_formatter(x, pos):
    exp = int(round(math.log10(x)))
    return r'$10^{%d}$' % exp

ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0,)))
ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0,)))
ax.xaxis.set_major_formatter(FuncFormatter(log_formatter))
ax.yaxis.set_major_formatter(FuncFormatter(log_formatter))


plt.xlabel('离散划分单元数量 n')
plt.ylabel('绝对误差 |π近似值 - π真值|')
plt.title('有限元离散收敛：误差收敛速率对比')

plt.grid(True, which='both', alpha=0.3, linestyle='-')
plt.legend()
plt.tight_layout()

plt.show()