import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import seaborn
seaborn.set_theme()

# Replace 'your_dataset.csv' with your actual file path
df = pd.read_csv('properties.csv')

# Map Location to numeric
location_mapping = {'Countryside': 1, 'Suburb': 2, 'Downtown': 3}
df['Location_Encoded'] = df['Location'].map(location_mapping)

# Calculate price per square foot
df['SquarefootPrice'] = df['SalePrice'] / df['Size_sqft']

# Save the updated dataset
#df.to_csv('properties_processed.csv', index=False)

#print("Processing complete. File saved as 'properties_processed.csv'.")


locations = df['Location'].unique()

fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

for ax, location in zip(axs, locations):
    subset = df[df['Location'] == location]
    ax.hist(subset['SalePrice'], bins=15, alpha=0.7, color='blue')
    ax.set_title(f'SalePrice Distribution - {location}')
    ax.set_xlabel('SalePrice')
    ax.set_ylabel('Frequency')

plt.tight_layout()
plt.show()

# Create scatter plots
colors = {1: 'green', 2: 'blue', 3: 'red'}
fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True, sharex=True)

for ax, (location_label, color) in zip(axs, colors.items()):
    subset = df[df['Location_Encoded'] == location_label]
    ax.scatter(subset['Size_sqft'], subset['SalePrice'], alpha=0.5, c=color, label=f'Location {location_label}')
    ax.set_title(f'SalePrice vs Size_sqft - Location {location_label}')
    ax.set_xlabel('Size (sqft)')
    ax.set_ylabel('Sale Price')
    ax.legend()

plt.tight_layout()
plt.show()


# Prepare data
X = df[['Size_sqft']].values
y = df['SalePrice'].values

# Fit linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict SalePrice for plotting regression line
X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_plot = model.predict(X_plot)

# Plot data and regression line
plt.scatter(X, y, alpha=0.5, color='blue', label='Data points')
plt.plot(X_plot, y_plot, color='red', linewidth=2, label='Regression line')
plt.xlabel('Size (sqft)')
plt.ylabel('Sale Price')
plt.title('Linear Regression of SalePrice on Size_sqft')
plt.legend()
plt.show()

# Model coefficients
print(f'Slope (coefficient): {model.coef_[0]}')
print(f'Intercept: {model.intercept_}')
print(f'R-squared: {model.score(X, y)}')
