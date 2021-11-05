import pyreadstat
# Read the sas7bdat file

# Written by TJG on 11/5/21
df, meta = pyreadstat.read_xport('DEMO.xpt')
type(df)
df.head()
df.to_csv('data_from_sas.csv', index=False)
