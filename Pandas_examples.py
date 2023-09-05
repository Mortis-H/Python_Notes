### 1. Writing into csv with unnecessary index
import pandas as pd
df = pd.read_csv("input_name.csv", index_col=False)
df.to_csv("output_name.csv", index=False)

### 2. Using column names which include spaces
import pandas as pd
df = pd.read_csv("input_name.csv")
df['New_Column'] = df['Age'].str[:1]

### 3. Filter dataset like a PRO with QUERY method
import pandas as pd
df = pd.read_csv("input_name.csv")
df.query('Year < 1992 and Time > 10')

### 4. Query strings with(@ symbol) to easily reach variables
import pandas as pd
df = pd.read_csv("input_name.csv")
min_year = 1992
min_time = 10
df.query('Year < @min_year and Time > @min_time')
