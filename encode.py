from data_processing import DataConverter
import numpy as np

target_name = "ag_news_original"

data_holder = DataConverter(f"data/raw/{target_name}.csv")
data = np.concatenate([data_holder.label, data_holder.embeddings], axis=1)
print(f"Finished! Saving...")
np.save(f"data/embeddings/{target_name}.npy", data)

data = np.load(f"data/embeddings/{target_name}.npy")
