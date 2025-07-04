import matplotlib.pyplot as plt
import numpy as np

# Hardware limits
peak_flops = 220.8  # GFLOPS
peak_bw = 8611.4 / 1024 # Result from stream bandwidth

# X: operational intensity
oi = np.logspace(-2, 2, 100)
y_bound = np.minimum(oi * peak_bw, peak_flops)

# Your code
your_oi = 0.25
your_gflops = 9.03

plt.figure(figsize=(10, 6))
plt.loglog(oi, y_bound, label='Roofline', linewidth=3)
plt.axhline(peak_flops, color='r', linestyle='--', label='Peak GFLOPS')
plt.axvline(peak_flops / peak_bw, color='g', linestyle='--', label='BW Limit')
plt.scatter([your_oi], [your_gflops], color='orange', s=100, label='Your Code')

plt.xlabel(f'Operational Intensity [FLOPs/Byte] ({peak_bw} * x) ')
plt.ylabel('Performance [GFLOPS]')
plt.title('Roofline Model')
plt.grid(True, which='both')
plt.legend()
plt.show()

