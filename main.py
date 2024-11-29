import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Ensure output directory exists
os.makedirs('Figs', exist_ok=True)

# Load data
ocean_temp_data = pd.read_csv(r'Data\1905-2019sst.csv')

# Convert COLLECTION_DATE to datetime
ocean_temp_data['COLLECTION_DATE'] = pd.to_datetime(
    ocean_temp_data['COLLECTION_DATE'], errors='coerce'
)
invalid_dates = ocean_temp_data['COLLECTION_DATE'].isnull().sum()
print(f"\nInvalid dates found: {invalid_dates}")

# Add Year and Month columns
ocean_temp_data['Year'] = ocean_temp_data['COLLECTION_DATE'].dt.year
ocean_temp_data['Month'] = ocean_temp_data['COLLECTION_DATE'].dt.month

# Filter out rows corresponding to the gap
start_gap, end_gap = '1949-07-01', '1950-10-04'
ocean_temp_data = ocean_temp_data[
    ~((ocean_temp_data['COLLECTION_DATE'] >= start_gap) & (ocean_temp_data['COLLECTION_DATE'] <= end_gap))
]

# Interpolate missing values for Sea Surface Temp Ave C
if 'Sea Surface Temp Ave C' in ocean_temp_data.columns:
    ocean_temp_data['Sea Surface Temp Ave C'] = pd.to_numeric(
        ocean_temp_data['Sea Surface Temp Ave C'], errors='coerce'
    ).interpolate(method='linear')

# Long-term trends
yearly_avg_temp = ocean_temp_data.groupby('Year')['Sea Surface Temp Ave C'].mean()

plt.figure(figsize=(12, 6))
plt.plot(yearly_avg_temp.index, yearly_avg_temp.values, label='Avg Temp (°C)', linewidth=2)
plt.title('Long-Term Trends in Average Sea Surface Temperature (1905–2019)', fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Average Temperature (°C)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12)
plt.savefig(os.path.join('Figs', '1LongTermTrendsLine.svg'))
plt.show()

# 2. Seasonality: Monthly averages across all years
monthly_avg_temp = ocean_temp_data.groupby('Month')['Sea Surface Temp Ave C'].mean()

plt.figure(figsize=(10, 6))
plt.bar(monthly_avg_temp.index, monthly_avg_temp.values, alpha=0.8, width=0.6)
plt.title('Seasonal Patterns in Average Sea Surface Temperature', fontsize=16)
plt.xlabel('Month', fontsize=12)
plt.ylabel('Average Temperature (°C)', fontsize=12)
plt.xticks(ticks=range(1, 13), labels=[
    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
])
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.savefig(os.path.join('Figs', 'Monthly Averages.svg'))
plt.show()


# 2.1 Adding range visualization with boxplot to show monthly variation
plt.figure(figsize=(12, 8))
ocean_temp_data['Month_Name'] = ocean_temp_data['Month'].map({
    1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
    7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
})

# Create a boxplot to show monthly temperature variation
plt.boxplot(
    [ocean_temp_data[ocean_temp_data['Month'] == i]['Sea Surface Temp Ave C'].dropna() for i in range(1, 13)],
    labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
    showfliers=False
)

plt.title('Monthly Variation in Average Sea Surface Temperature', fontsize=16)
plt.xlabel('Month', fontsize=12)
plt.ylabel('Temperature (°C)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.savefig('Figs\MonthlyAveragesBox.svg')
plt.show()

# 3. Rolling Averages: Smooth trends and highlight anomalies
# Calculate a 10-year rolling average for smooth trends
rolling_avg = yearly_avg_temp.rolling(window=10, center=True).mean()

plt.figure(figsize=(12, 6))
plt.plot(yearly_avg_temp.index, yearly_avg_temp.values, label='Yearly Avg Temp (°C)', alpha=0.6)
plt.plot(rolling_avg.index, rolling_avg.values, label='10-Year Rolling Avg (°C)', color='red', linewidth=2)
plt.title('Long-Term Trends with 10-Year Rolling Average', fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Average Temperature (°C)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12)
plt.savefig('Figs\Long-Term-Trends-wRolling.svg')
plt.show()

# 4. Highlight Anomalies: Scatter plot for extreme years
# Calculate the mean and standard deviation of yearly average temperatures
mean_temp = yearly_avg_temp.mean()
std_temp = yearly_avg_temp.std()

# Define thresholds for anomalies (e.g., beyond 2 standard deviations)
anomaly_upper = mean_temp + 2 * std_temp
anomaly_lower = mean_temp - 2 * std_temp

# Identify anomalies
anomalies = yearly_avg_temp[(yearly_avg_temp > anomaly_upper) | (yearly_avg_temp < anomaly_lower)]

plt.figure(figsize=(12, 6))
plt.scatter(yearly_avg_temp.index, yearly_avg_temp.values, label='Yearly Avg Temp (°C)', alpha=0.6)
plt.scatter(anomalies.index, anomalies.values, color='red', label='Anomalies', zorder=5)
plt.axhline(anomaly_upper, color='orange', linestyle='--', label='+2 Std Dev', alpha=0.8)
plt.axhline(anomaly_lower, color='blue', linestyle='--', label='-2 Std Dev', alpha=0.8)
plt.title('Highlighting Anomalies in Average Sea Surface Temperature', fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Average Temperature (°C)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12)
plt.savefig('Figs\AnomolyScatter.svg')
plt.show()


# Pivot the data for heatmap: Rows = Years, Columns = Months
heatmap_data = ocean_temp_data.pivot_table(
    index='Year', columns='Month', values='Sea Surface Temp Ave C', aggfunc=np.mean
)

# Plotting the heatmap
plt.figure(figsize=(12, 8))
plt.imshow(heatmap_data, aspect='auto', cmap='coolwarm', origin='lower')
plt.colorbar(label='Avg Temp (°C)')
plt.title('Yearly Heatmap of Average Sea Surface Temperatures', fontsize=16)
plt.xlabel('Month', fontsize=12)
plt.ylabel('Year', fontsize=12)
plt.xticks(ticks=np.arange(12), labels=[
    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
])
plt.yticks(ticks=np.arange(0, len(heatmap_data.index), 10), labels=heatmap_data.index[::10])
plt.grid(False)
plt.savefig('Figs\Heatmap.svg')
plt.show()