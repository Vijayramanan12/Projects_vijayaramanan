import pandas as pd

data = pd.read_csv('/Users/vijayramanan/Downloads/archive/IBM HR Employee Attrition Data.csv')
df = pd.DataFrame(data[['Education']].value_counts()).reset_index()
print(df)