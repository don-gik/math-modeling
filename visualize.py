import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


data = ["일시", "평균기온(°C)", "일강수량(mm)", "평균 풍속(m/s)", "평균 상대습도(%)"]

df = pd.read_csv("SURFACE_ASOS_105_DAY_2019_2019_2025.csv", encoding="EUC-KR", usecols=data)
df["date"] = pd.to_datetime(df["일시"])
df = df.set_index("date")

rolling_mean = df.rolling(window=3).mean()

rolling_mean.plot()

plt.title("Basic Plot")
plt.xlabel("Dates")
plt.ylabel("Values")

plt.show()
