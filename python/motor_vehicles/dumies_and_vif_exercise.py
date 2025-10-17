import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
sns.set_theme()

#local switches
debugging = True

#Load
raw_data = pd.read_csv("1.04. Real-life example.csv")
print(raw_data.head(20))
print(raw_data.describe(include='all'))

#Preprocess
#  select variables of interest
data = raw_data.drop(['Model'],axis=1)
print(data.describe(include='all'))

#  Deal with missing values
print(data.isnull().sum())
data_no_mv = data.dropna(axis=0)
print(data_no_mv.isnull().sum())
print(data_no_mv.describe(include='all'))

#Explore Probability Density Functions
if not debugging:
#    sns.distplot(data_no_mv['Price'])
    sns.displot(data_no_mv['Price'])
    plt.show()

#Dealing with outliers
#  see where the quartiles lay
print(f"75% of prices are under ${data_no_mv['Price'].quantile(0.75)}")
print(f"99% of prices are under ${data_no_mv['Price'].quantile(0.99)}")
print(f"75% of mileages are under {data_no_mv['Mileage'].quantile(0.75)} Kmiles")
print(f"99% of mileages are under {data_no_mv['Mileage'].quantile(0.99)} Kmiles")
print(f"75% of engines are under {data_no_mv['EngineV'].quantile(0.75)}L")
print(f"99% of engines are under {data_no_mv['EngineV'].quantile(0.99)}L")
print(f"75% of cars were made after {data_no_mv['Year'].quantile(0.25)}")
print(f"99% of cars were made after {data_no_mv['Year'].quantile(0.01)}")

#remove 1% outliers
q1 = data_no_mv['Price'].quantile(0.99)
data_1 = data_no_mv[data_no_mv['Price'] < q1]

q2 = data_1['Mileage'].quantile(0.99)
data_2 = data_1[data_1['Mileage'] < q2]

#  Asume all engines are under 6 and a half liters
data_3 = data_2[data_2['EngineV'] < 6.5]

# 'Year' outliers are in the minimum end
q4 = data_3['Year'].quantile(0.01)
data_4 = data_3[data_3['Year'] > q4]

data_cleaned = data_4.reset_index(drop=True)

#display difference before and after outliers removal
if not debugging:
    f1, axs = plt.subplots(2,4, sharey='col', figsize = (16,8))
    axs[0,0].set_title('Price')
    axs[0,0].hist(data_no_mv['Price'])
    axs[1,0].hist(data_cleaned['Price'])
    axs[0,1].set_title('Mileage')
    axs[0,1].hist(data_no_mv['Mileage'])
    axs[1,1].hist(data_cleaned['Mileage'])
    axs[0,2].set_title('EngineV')
    axs[0,2].hist(data_no_mv['EngineV'])
    axs[1,2].hist(data_cleaned['EngineV'])
    axs[0,3].set_title('Year')
    axs[0,3].hist(data_no_mv['Year'])
    axs[1,3].hist(data_cleaned['Year'])
    plt.show()

if not debugging:
    f, (ax1,ax2,ax3) = plt.subplots(1,3,sharey=True,figsize=(15,3))
    ax1.set_title('Price and Year')
    ax1.scatter(data_cleaned['Year'],data_cleaned['Price'])
    ax2.set_title('Price and EngineV')
    ax2.scatter(data_cleaned['EngineV'],data_cleaned['Price'])
    ax3.set_title('Price and Mileage')
    ax3.scatter(data_cleaned['Mileage'],data_cleaned['Price'])
    plt.show()
    sns.displot(data_cleaned['Price'])

log_price = np.log(data_cleaned['Price'])
data_cleaned['log_price'] = log_price
print(data_cleaned.describe())

if not debugging:
    fig, (ax1,ax2,ax3) = plt.subplots(1,3,sharey=True,figsize=(15,3))
    ax1.set_title('log_price and Year')
    ax1.scatter(data_cleaned['Year'],data_cleaned['log_price'])
    ax2.set_title('log_price and EngineV')
    ax2.scatter(data_cleaned['EngineV'],data_cleaned['log_price'])
    ax3.set_title('log_price and Mileage')
    ax3.scatter(data_cleaned['Mileage'],data_cleaned['log_price'])
    plt.show()

data_cleaned = data_cleaned.drop(['Price'], axis=1)

# Multicollinearity
print(data_cleaned.columns.values)
from statsmodels.stats.outliers_influence import variance_inflation_factor
variables = data_cleaned[['Mileage','Year','EngineV']]
print(f"esto está en variables: {variables} y es de tipo {type(variables)}")
vif=pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(variables.values,i) for i in range(variables.shape[1])]
vif["Features"] = variables.columns

print(vif)  # Year have a high collinearity; we'll drop it.

data_no_multicollinearity = data_cleaned.drop(['Year'],axis=1)
variables = data_no_multicollinearity[['Mileage','EngineV']]
vif2=pd.DataFrame()
vif2['VIF'] = [variance_inflation_factor(variables.values,i) for i in range(variables.shape[1])]
vif2['Features'] = variables.columns
print(vif2)  # Checking that collinearity is lower after dropping Year.

# Dummy variables
data_with_dummies = pd.get_dummies(data_no_multicollinearity, drop_first=True)
print(data_with_dummies.head())
print(data_with_dummies.columns.values)

#Rearrange columns
cols = ['log_price', 'Mileage', 'EngineV', 'Brand_BMW', 'Brand_Mercedes-Benz',
 'Brand_Mitsubishi', 'Brand_Renault', 'Brand_Toyota', 'Brand_Volkswagen',
 'Body_hatch', 'Body_other', 'Body_sedan', 'Body_vagon', 'Body_van',
 'Engine Type_Gas', 'Engine Type_Other', 'Engine Type_Petrol',
 'Registration_yes']

data_preprocessed = data_with_dummies[cols]
print(data_preprocessed.head(10))

# ------------------------------------------------------------------------
# EXERCISE
# ------------------------------------------------------------------------
# Calculate the variance inflation factors for all variables contained in data_preprocessed. Point any strange observation
## Part 1
vif3 = pd.DataFrame()
data_vars_numeric = data_preprocessed.select_dtypes(include=['number'])
vif3["Features"] = data_vars_numeric.columns
vif3["VIF"] = [variance_inflation_factor(data_vars_numeric.values, i) for i in range(data_vars_numeric.shape[1])]
print(vif3)

## Part 2
data_vars = data_vars_numeric.drop(['log_price'],axis=1)
vif4 = pd.DataFrame()
vif4['VIF'] = [variance_inflation_factor(data_vars.values, i) for i in range(data_vars.shape[1])]
vif4['Features'] = data_vars.columns
print(vif4)


## Part 3
# It is not possible to apply variance_inflation_factor to dummies: they fail to behave as numeric and produce an error.