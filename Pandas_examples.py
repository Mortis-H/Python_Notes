### 1. Writing into csv with unnecessary index
import pandas as pd
df = pd.read_csv("input_name.csv", index_col=False)
df.to_csv("output_name.csv", index=False)
