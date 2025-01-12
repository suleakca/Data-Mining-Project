import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

df = pd.read_csv("hour.csv")

plt.figure(figsize=(10, 6))
sb.swarmplot(data=df, x='mnth', y='registered')
plt.xlabel('Month')
plt.ylabel('Registered User')
plt.title('Month vs Registered Users')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


#####################
months = df['mnth']
registered_user = df['registered']
rate_usage = registered_user / registered_user.sum()

plt.figure(figsize=(10, 6))
plt.bar(months, rate_usage, color='skyblue')
plt.xlabel('Month')
plt.ylabel('Rate')
plt.title('Bicycle Usage Ratio by Month')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
