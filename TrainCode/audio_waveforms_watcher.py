import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

df = pd.read_csv("../ProcessedDatasets/audio_waveforms.csv")

index = 299

# Преобразуем строку в массив, убирая возможные проблемы с форматированием
waveform = np.fromstring(df.iloc[index]["waveform"].strip("[]"), sep=" ")
class_label = df.iloc[index]["class"]

print(df["class"])

plt.figure(figsize=(10, 4))
plt.plot(range(len(waveform)), waveform, color='green' if class_label == 1 else 'red')
plt.title(f"Waveform (Class: {'Real' if class_label == 1 else 'Fake'})")
plt.xlabel("Time (samples)")
plt.ylabel("Amplitude")
plt.grid()
plt.show()
