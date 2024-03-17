import pandas as pd

# Veri setini yükle
df = pd.read_csv('../dataset/instructions.csv', index_col=0)

# Veri setinin büyüklüğünü yaz
print(len(df))