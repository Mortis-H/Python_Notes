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

### 5. "inplace" method could be removed in future versions, better explicitly overwrite modifications
import pandas as pd
df = pd.read_csv("input_name.csv")
df.fillna(0)
df.reset_index()

### 6. better Vectorization instead of iteration
import pandas as pd
df = pd.read_csv("input_name.csv")
df['result'] = df['Year'] > 2000

### 7. Vectorization method are preferable than Apply method
import pandas as pd
df = pd.read_csv("input_name.csv")
df['result'] = df['Year'] ** 2

### 8. df.copy() method
import pandas as pd
df = pd.read_csv("input_name.csv")
df_new = df.query(f'Year < {min_year} and Time > {min_time}').copy()
df_new['First_Name'] = df_new['Name'].str[-5:]

### 9. Chaining formulas is better than creating many intermediate dataframes
import pandas as pd
df = pd.read_csv("input_name.csv")
df_out = df.query('Year > 1999')
           .gorupby(['Gender'])'Time'.min()
           .sort_values('Time')

### 10. Properly set column dtypes
import pandas as pd
df = pd.read_csv("input_name.csv")
df['Date'] = pd.to_datetime(df['Date'])
# Also see: pd.to_numeric() and pd.to_timedelta()
df.info()

### 11. Using Boolean instead of Strings
import pandas as pd
df = pd.read_csv("input_name.csv")

df['sub10'] = df['Time'] < 10

### 12. pandas plot method instead of matplotlib import
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("input_name.csv")

ax = df.plot(kind='scatter',
            x='Year',
            y='Time'
            title='Year vs Speed')

### 13. pandas str.upper() instead apply and etc
import pandas as pd
df = pd.read_csv("input_name.csv")

df['Name_Uppercase'] = df['Name'].str.upper()
# also str.strip(), str.lower(), str.replace(), and str.split()

### 14. use data pipeline once instead of repeating many times
import pandas as pd
def process_data(df):
    df['Time_Norm'] = df['Time'] / df['Time'].mean()
    df['Place'] = df['Place'].str.lower()
    return df

df_1 = pd.read_csv("input_name_1.csv")
df_2 = pd.read_csv("input_name_2.csv")

df_1 = process_data(df_1)
df_2 = process_data(df_2)
