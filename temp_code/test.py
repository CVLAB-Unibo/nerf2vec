import matplotlib.pyplot as plt
import numpy as np

# Sample data
n_units = [64, 256, 512]  # Number of units in linear layer
psnr_set1 = [23.50798246595594, 23.97099600897895, 26.0391526222229]  # First set of PSNR values
psnr_set2 = [34.5078550974528, 35.84525712331136, 34.34378661049737]  # Second set of PSNR values
time_elapsed = [14.87, 37.99, 88.99]  # Time elapsed in seconds

# Create the figure and axis
fig, ax1 = plt.subplots(figsize=(8, 6))

ax1.set_ylim(10, 40)

# Create bars for the first set of PSNR values
bar_width = 0.15
index = np.arange(len(n_units))
bar1 = ax1.bar(index, psnr_set1, bar_width, label='Identity Encoding', alpha=0.7, color='b')

# Create bars for the second set of PSNR values
bar2 = ax1.bar(index + bar_width, psnr_set2, bar_width, label='Frenquency Encoding', alpha=0.7, color='r')

# Set the first y-axis labels
# ax1.set_xlabel('Number of Units in Linear Layer')
# ax1.set_ylabel('PSNR Value')

ax1.set_xticks(index + bar_width / 2)
ax1.set_xticklabels(n_units)


# Create the second y-axis for time elapsed
ax2 = ax1.twinx()
ax2.plot(index + bar_width / 2, time_elapsed, 'o--', label='Time Elapsed (s)', linewidth=2, color='darkgreen')

# Set the second y-axis labels
# ax2.set_ylabel('Time Elapsed (s)', color='darkgreen')
fig.legend(loc='upper left',  bbox_to_anchor=(0.12, 0.88), labels=['Identity Encoding', 'Frenquency Encoding', 'Average Time Elapsed (s)'])

# Save the plot as an image (e.g., PNG)
plt.savefig('encodings_and_units.png', bbox_inches='tight')

# Don't forget to close the plot to release resources
plt.close()
