from datasets import load_dataset
import pandas as pd
from datasets import Dataset, Audio

data = pd.read_csv("bird_dataset.csv")
paths = data['Path'].tolist()
# append 'dataset/' to each path
paths = ['dataset/' + path for path in paths]

audio_dataset = Dataset.from_dict({"audio": paths}).cast_column("audio", Audio())
print(audio_dataset[0]["audio"])