import pandas as pd
import statistics
import matplotlib.pyplot as plt

#Load data
df = pd.read_excel('IRCTC Stock Price.xlsx', sheet_name='Sheet1')

#Calculate mean and variance of the Price column
price_mean = statistics.mean(df['Price'])
price_variance = statistics.variance(df['Price'])

print(f"Mean of Price: {price_mean}")
print(f"Variance of Price: {price_variance}")
wednesdays_data = df[df['Day'] == 'Wed']
wednesdays_mean = statistics.mean(wednesdays_data['Price'])

print(f"Mean of Price on Wednesdays: {wednesdays_mean}")
print(f"Population mean: {price_mean}")

april_data = df[df['Month'] == 'Apr']
april_mean = statistics.mean(april_data['Price'])

print(f"Mean of Price in April: {april_mean}")
print(f"Population mean: {price_mean}")

loss_probability = len(df[df['Chg%'] < 0]) / len(df)

print(f"Probability of making a loss: {loss_probability}")

wednesdays_profit_data = wednesdays_data[wednesdays_data['Chg%'] > 0]
wednesdays_profit_probability = len(wednesdays_profit_data) / len(wednesdays_data)

print(f"Probability of making a profit on Wednesday: {wednesdays_profit_probability}")

conditional_probability = len(wednesdays_profit_data) / len(wednesdays_data)

print(f"Conditional probability of making profit given today is Wednesday: {conditional_probability}")

#Maping days to numerical values for plotting
day_map = {'Mon': 1, 'Tue': 2, 'Wed': 3, 'Thu': 4, 'Fri': 5}
df['Day_num'] = df['Day'].map(day_map)

#Scatter plot
plt.scatter(df['Day_num'], df['Chg%'])
plt.xlabel('Day of the Week')
plt.ylabel('Chg%')
plt.title('Scatter plot of Chg% against Day of the Week')
plt.xticks(list(day_map.values()), list(day_map.keys()))
plt.show()