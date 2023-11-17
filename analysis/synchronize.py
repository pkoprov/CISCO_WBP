import numpy as np
import pandas as pd
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import r2_score

from analysis.plotting import shift_for_maximum_correlation


df = pd.DataFrame(columns=["Robot1", "Robot2","DateTime"])

df["Robot1"] = pd.read_csv(r'data/Kernels/UR/UR-5e Cary/UR-5e Cary_1670847516.csv').iloc[:,1]
df["Robot2"] = pd.read_csv(r'data/Kernels/UR/UR-5e Cary/UR-5e Cary_1670847537.csv').iloc[:,1]
df["DateTime"] = pd.read_csv(r'data/Kernels/UR/UR-5e Cary/UR-5e Cary_1670847516.csv').iloc[:,0]

x = np.array(df["Robot1"].fillna(0))
y = np.array(df["Robot2"].fillna(0))
y_shifted = shift_for_maximum_correlation(x, y)[0]

plt.plot(x)
plt.plot(y)
plt.plot(y_shifted)


distance, path = fastdtw(x, y)

result = []
for i in range(0,len(path)):
    result.append([df['DateTime'].iloc[path[i][0]],
    df['Robot1'].iloc[path[i][0]],
    df['Robot2'].iloc[path[i][1]]])
df_sync = pd.DataFrame(data=result,columns=['DateTime','Robot1','Robot2']).dropna()
df_sync = df_sync.drop_duplicates(subset=['DateTime'])
df_sync = df_sync.sort_values(by='DateTime')
df_sync = df_sync.reset_index(drop=True)
# df_sync.to_csv('C:/Users/.../synchronized_dataset.csv',index=False)
plt.plot(df_sync['Robot1'],label='Robot1')
plt.plot(df_sync['Robot2'],label='Robot2')
plt.plot(df['Robot1'],label='Robot1')
plt.plot(df['Robot2'],label='Robot2')



r2_score(df['Robot1'],df['Robot2'])
r2_score(df_sync['Robot1'],df_sync['Robot2'])

def chart(df):
    df_columns = list(df)
    df['DateTime'] = pd.to_datetime(df['DateTime'],format='%d-%m-%y %H:%M')
    df['DateTime'] = df['DateTime'].dt.strftime(' %H:%M on %B %-d, %Y')
    df = df.sort_values(by='DateTime')

    fig = px.line(df, x="DateTime", y=df_columns,
                  labels={
                      "DateTime": "DateTime",
                      "value": "Value",
                      "variable": "Variables"
                      },
                  hover_data={"DateTime": "|%d-%m-%Y %H:%M"})
    fig.update_layout(
        font_family="IBM Plex Sans",
        font_color="black"
        )
    fig.update_xaxes(
        rangeselector=dict(
            buttons=list([
                dict(count=12, label="12h", step="hour", stepmode="backward"),
                dict(count=1, label="1d", step="day", stepmode="backward"),
                dict(count=7, label="7d", step="day", stepmode="backward"),
                dict(step="all")
            ])
            )
        )

    st.write(fig)