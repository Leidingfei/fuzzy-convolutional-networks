

### given 2 files, merge. 
# files:  

from datetime import datetime
### given 2 files, merge. 
# files:  
import pandas as pd

def convert_date(d):
	return datetime.strptime(d, '%b %d, %Y').strftime('%Y-%m-%d')



#smi1: 1 dec 2007 and onwards need
df1=pd.read_csv("usdeuro1.csv")
print(df1.head())
print(df1.tail())

df2=pd.read_csv("usdeuro2.csv")

# print(df2[df2["Date"]=='Aug 12, 1999'])
print(df2[32:])





df3 = pd.concat([df1,df2], sort=False).reset_index(drop=True)
df3["Date"] = df3["Date"].apply(convert_date)
df3.to_csv('usdeuro_final.csv')

