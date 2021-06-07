import pandas as pd
import config
import matplotlib.pyplot as plt

# Simple data visualization of polluting gases

df = pd.read_csv(filepath_or_buffer='C:\\github_repos\\Universal-Embeddings\\data\\test_data.csv')

fig, ax = plt.subplots(2, 3, figsize=(12,6))

ax[0,0].plot(df['dt'], df['aqi'], color='green')
ax[0,0].set_title('aqi')
ax[0,0].set_xticks([])

ax[0,1].plot(df['dt'], df['co'], color='blue')
ax[0,1].set_title('co')
ax[0,1].set_xticks([])

ax[0,2].plot(df['dt'], df['no'], color='red')
ax[0,2].set_title('no')
ax[0,2].set_xticks([])

ax[1,0].plot(df['dt'], df['no2'], color='pink')
ax[1,0].set_title('no2')
ax[1,0].set_xticks([])

ax[1,1].plot(df['dt'], df['o3'], color='purple')
ax[1,1].set_title('o3')
ax[1,1].set_xticks([])

ax[1,2].plot(df['dt'], df['so2'], color='orange')
ax[1,2].set_title('so2')
ax[1,2].set_xticks([])

plt.legend()
plt.show()