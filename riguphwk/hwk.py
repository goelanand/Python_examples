#! /usr/bin/python3


import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns


conn = psycopg2.connect("dbname=rudata user=datauser password=caDJOSIo host=rigup-data-challenge.cjznq3oye5gp.us-east-1.rds.amazonaws.com")

cur = conn.cursor()

cur.execute("SELECT * from daily_metrics")
df = pd.DataFrame(cur.fetchall(), columns=['id','metric','value','day'])
df['day'] = df['day'].astype('datetime64[ns]')
#df.set_index('day',inplace=True)
values = df.value

fig, ax = plt.subplots(figsize=(15,7))
ax.bar(df.index, df['value'])
ax.set_ylabel('Day')
ax.set_xlabel('Daily Visitors')
ax.set_title('Daily Visitors by Date')
plt.show()
