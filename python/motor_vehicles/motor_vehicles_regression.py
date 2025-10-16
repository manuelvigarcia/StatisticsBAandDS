import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
sns.set_theme()

#load data from "1.04. Real-life example.csv"
raw_data = pd.read_csv("1.04. Real-life example.csv")
print(raw_data.describe(include='all'))

#Preprocessing
print("calculate null values count")
print(raw_data.isnull().sum())
print("null count is not relevant. Dropping all rows with a null")
data_no_mv = raw_data.dropna(axis=0)
print("null count now:")
print(data_no_mv.isnull().sum())
print(data_no_mv.describe(include='all'))

#get rid of the 1% highest prices
q = data_no_mv['Price'].quantile(0.99)
data_1 = data_no_mv[data_no_mv['Price']<q]
print(data_1.describe(include='all'))

#get rid of the 1% highest mileages
qm = data_1['Mileage'].quantile(0.99)
data_2 = data_1[data_1['Mileage']<qm]
print(data_2.describe(include='all'))

#Assume engine volume is always below 6.5L and discard anything higher
data_3 = data_2[data_2['EngineV']< 6.5]
print(data_3.describe(include='all'))

#discard the 1% of the oldest cars (.99 quantile of the year)
qy = data_3['Year'].quantile(0.01)
data_4 = data_3[data_3['Year'] > qy]
print(data_4.describe(include='all'))

clean_data = data_4.reset_index(drop=True)

#Comment model plot, since it difficult to interpret
#graf = sns.displot(clean_data['Model'])
#plt.show()

print(clean_data.describe(include='all'))


# try Interquartile Range Method to discard outliers.
def remove_outliers_iqr(data, column_tag, factor = 1.5, lower = True, upper = True):
    q1 = np.percentile(data.loc[:, column_tag], 25)
    q3 = np.percentile(data.loc[:, column_tag], 75)
    #q3 = np.percentile(data[column_tag], 75)
    iqr = q3 - q1
    lower_bound = min(data[column_tag])
    upper_bound = max(data[column_tag])
    if lower:
        lower_bound = q1 - factor * iqr

    if upper:
        upper_bound = q3 + factor * iqr

    print(f"filtering with {upper_bound} and {lower_bound}")
#    no_outliers = data[lower_bound < data[column_tag] < upper_bound]
    no_outliers = data[(data[column_tag] > lower_bound) & (data[column_tag] < upper_bound)]
    return no_outliers

print("Outliers with IRQ")

reduced_years = remove_outliers_iqr(data_no_mv,'Year',upper=False)
reduced_price = remove_outliers_iqr(reduced_years, 'Price', lower=False)
reduced_mileage = remove_outliers_iqr(reduced_price, 'Mileage',lower=False)
data_no_iqr = remove_outliers_iqr(reduced_mileage, 'EngineV',lower=False)

print("Comparison of outliers removal performance")
print(f"{clean_data.shape}  with 1% removal")
print(f"{data_no_iqr.shape}  with IQR method")

#Probability distribution function. Comparison after removing outliers
f1, axs = plt.subplots(3,4, sharey='col', figsize = (16,8))
axs[0,0].set_title("Year")
axs[0,0].hist(data_no_mv['Year'])
axs[1,0].hist(clean_data['Year'])
axs[2,0].hist(data_no_iqr['Year'])
axs[0,1].set_title("Price")
axs[0,1].hist(data_no_mv['Price'])
axs[1,1].hist(clean_data['Price'])
axs[2,1].hist(data_no_iqr['Price'])
axs[0,2].set_title('Mileage')
axs[0,2].hist(data_no_mv['Mileage'])
axs[1,2].hist(clean_data['Mileage'])
axs[2,2].hist(data_no_iqr['Mileage'])
axs[0,3].set_title('EngineV')
axs[0,3].hist(data_no_mv['EngineV'])
axs[1,3].hist(clean_data['EngineV'])
axs[2,3].hist(data_no_iqr['EngineV'])
plt.show()


# We continue with the data after removing the outliers with IQR method, since it is more aggressive with outliers

#plot Year, mileage And EngineV against price.
f, (ax1, ax2, ax3) = plt.subplots(1,3, sharey = True, figsize = (15,3))
ax1.scatter(data_no_iqr['Year'], data_no_iqr['Price'], alpha=0.2)
ax1.set_title('Price and Year')
ax2.scatter(data_no_iqr['EngineV'], data_no_iqr['Price'], alpha=0.2)
ax2.set_title('Price and EngineV')
ax3.scatter(data_no_iqr['Mileage'], data_no_iqr['Price'], alpha=0.2)
ax3.set_title('price and Mileage')
plt.show()

#Analyze Multicollinearity
from statsmodels.stats.outliers_influence import variance_inflation_factor
variables = data_no_iqr[['Mileage','Year','EngineV']]
vif = pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vif['Features'] = variables.columns
print(vif)
print(vif.describe(include='all'))

# Year has a high VIF, so we leave it out of further analysis
data_low_vif = data_no_iqr.drop(['Year'],axis=1)

#Verification that, without 'Year', there is lower variance inflation
variables = data_no_iqr[['Mileage','EngineV']]
vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vif['Features'] = variables.columns
print(vif)
print(vif.describe(include='all'))


#Dummy variables
data_with_dummies = pd.get_dummies(data_low_vif, drop_first=True)
print(data_with_dummies.describe())
#Get the columns names, in order to reárrange the columns
print(data_with_dummies.columns.values)


#feature selection with f_regression
# calculate F_regression and discard all features with p_value >= 0.001
from sklearn.feature_selection import f_regression
y = data_with_dummies['Price']
x = data_with_dummies[['Mileage', 'EngineV', 'Brand_BMW', 'Brand_Mercedes-Benz',
 'Brand_Mitsubishi', 'Brand_Renault', 'Brand_Toyota', 'Brand_Volkswagen',
 'Body_hatch', 'Body_other', 'Body_sedan', 'Body_vagon', 'Body_van',
 'Engine Type_Gas', 'Engine Type_Other', 'Engine Type_Petrol',
 'Registration_yes', 'Model_100', 'Model_116', 'Model_118', 'Model_120',
 'Model_19', 'Model_190', 'Model_200', 'Model_210', 'Model_220', 'Model_230',
 'Model_250', 'Model_316', 'Model_318', 'Model_320', 'Model_323', 'Model_325',
 'Model_328', 'Model_330', 'Model_335', 'Model_428', 'Model_5 Series',
 'Model_5 Series GT', 'Model_520', 'Model_523', 'Model_525', 'Model_528',
 'Model_530', 'Model_535', 'Model_540', 'Model_630', 'Model_730', 'Model_735',
 'Model_740', 'Model_80', 'Model_A 140', 'Model_A 150', 'Model_A 170',
 'Model_A 180', 'Model_A1', 'Model_A3', 'Model_A4', 'Model_A4 Allroad',
 'Model_A5', 'Model_A6', 'Model_A6 Allroad', 'Model_A7', 'Model_A8',
 'Model_ASX', 'Model_Amarok', 'Model_Auris', 'Model_Avalon', 'Model_Avensis',
 'Model_Aygo', 'Model_B 170', 'Model_B 180', 'Model_B 200', 'Model_Beetle',
 'Model_Bora', 'Model_C-Class', 'Model_CL 180', 'Model_CLA 200',
 'Model_CLA-Class', 'Model_CLC 180', 'Model_CLC 200', 'Model_CLK 200',
 'Model_CLK 220', 'Model_CLK 230', 'Model_CLK 240', 'Model_CLK 280',
 'Model_CLK 320', 'Model_CLS 350', 'Model_Caddy', 'Model_Camry',
 'Model_Caravelle', 'Model_Carina', 'Model_Carisma', 'Model_Celica',
 'Model_Clio', 'Model_Colt', 'Model_Corolla', 'Model_Corolla Verso',
 'Model_Cross Touran', 'Model_Duster', 'Model_E-Class', 'Model_Eclipse',
 'Model_Eos', 'Model_Espace', 'Model_FJ Cruiser', 'Model_Fluence',
 'Model_Fortuner', 'Model_G 320', 'Model_GL 320', 'Model_GL 420',
 'Model_GLK 220', 'Model_GLK 300', 'Model_Galant', 'Model_Golf GTI',
 'Model_Golf III', 'Model_Golf IV', 'Model_Golf Plus', 'Model_Golf V',
 'Model_Golf VI', 'Model_Golf VII', 'Model_Golf Variant',
 'Model_Grand Scenic', 'Model_Grandis', 'Model_Hiace', 'Model_Highlander',
 'Model_Hilux', 'Model_I3', 'Model_IQ', 'Model_Jetta', 'Model_Kangoo',
 'Model_Koleos', 'Model_L 200', 'Model_LT', 'Model_Laguna', 'Model_Lancer',
 'Model_Lancer Evolution', 'Model_Lancer X', 'Model_Lancer X Sportback',
 'Model_Land Cruiser 100', 'Model_Land Cruiser 200', 'Model_Land Cruiser 80',
 'Model_Land Cruiser Prado', 'Model_Latitude', 'Model_Logan', 'Model_Lupo',
 'Model_M5', 'Model_ML 250', 'Model_ML 270', 'Model_ML 280', 'Model_ML 320',
 'Model_ML 350', 'Model_ML 400', 'Model_Master', 'Model_Matrix',
 'Model_Megane', 'Model_Modus', 'Model_Multivan', 'Model_New Beetle',
 'Model_Outlander', 'Model_Outlander XL', 'Model_Pajero',
 'Model_Pajero Pinin', 'Model_Pajero Sport', 'Model_Pajero Wagon',
 'Model_Passat B3', 'Model_Passat B4', 'Model_Passat B5', 'Model_Passat B6',
 'Model_Passat B7', 'Model_Passat CC', 'Model_Phaeton', 'Model_Pointer',
 'Model_Polo', 'Model_Previa', 'Model_Prius', 'Model_Q3', 'Model_Q5',
 'Model_Q7', 'Model_R 320', 'Model_Rav 4', 'Model_S 140', 'Model_S 300',
 'Model_S 320', 'Model_S 350', 'Model_S 400', 'Model_S4', 'Model_S5',
 'Model_S8', 'Model_SLK 200', 'Model_SLK 350', 'Model_Sandero', 'Model_Scenic',
 'Model_Scion', 'Model_Scirocco', 'Model_Sharan', 'Model_Sienna',
 'Model_Space Star', 'Model_Space Wagon', 'Model_Sprinter 208',
 'Model_Sprinter 210', 'Model_Sprinter 211', 'Model_Sprinter 212',
 'Model_Sprinter 213', 'Model_Sprinter 311', 'Model_Sprinter 312',
 'Model_Sprinter 313', 'Model_Sprinter 315', 'Model_Sprinter 316',
 'Model_Sprinter 318', 'Model_Sprinter 319', 'Model_Symbol', 'Model_Syncro',
 'Model_T4 (Transporter)', 'Model_T4 (Transporter) ',
 'Model_T5 (Transporter)', 'Model_T5 (Transporter) ',
 'Model_T6 (Transporter)', 'Model_T6 (Transporter) ', 'Model_TT',
 'Model_Tacoma', 'Model_Tiguan', 'Model_Touareg', 'Model_Touran',
 'Model_Trafic', 'Model_Up', 'Model_Vaneo', 'Model_Vento', 'Model_Venza',
 'Model_Viano', 'Model_Virage', 'Model_Vista', 'Model_Vito', 'Model_X1',
 'Model_X3', 'Model_X5', 'Model_X5 M', 'Model_X6', 'Model_Yaris', 'Model_Z3',
 'Model_Z4']]

results_all = f_regression(x,y)
p_values = results_all[1].round(3)
print(p_values)

features_pvalues = pd.DataFrame()
features_pvalues['pvalues'] = p_values
features_pvalues['Features'] = ['Mileage', 'EngineV', 'Brand_BMW', 'Brand_Mercedes-Benz',
 'Brand_Mitsubishi', 'Brand_Renault', 'Brand_Toyota', 'Brand_Volkswagen',
 'Body_hatch', 'Body_other', 'Body_sedan', 'Body_vagon', 'Body_van',
 'Engine Type_Gas', 'Engine Type_Other', 'Engine Type_Petrol',
 'Registration_yes', 'Model_100', 'Model_116', 'Model_118', 'Model_120',
 'Model_19', 'Model_190', 'Model_200', 'Model_210', 'Model_220', 'Model_230',
 'Model_250', 'Model_316', 'Model_318', 'Model_320', 'Model_323', 'Model_325',
 'Model_328', 'Model_330', 'Model_335', 'Model_428', 'Model_5 Series',
 'Model_5 Series GT', 'Model_520', 'Model_523', 'Model_525', 'Model_528',
 'Model_530', 'Model_535', 'Model_540', 'Model_630', 'Model_730', 'Model_735',
 'Model_740', 'Model_80', 'Model_A 140', 'Model_A 150', 'Model_A 170',
 'Model_A 180', 'Model_A1', 'Model_A3', 'Model_A4', 'Model_A4 Allroad',
 'Model_A5', 'Model_A6', 'Model_A6 Allroad', 'Model_A7', 'Model_A8',
 'Model_ASX', 'Model_Amarok', 'Model_Auris', 'Model_Avalon', 'Model_Avensis',
 'Model_Aygo', 'Model_B 170', 'Model_B 180', 'Model_B 200', 'Model_Beetle',
 'Model_Bora', 'Model_C-Class', 'Model_CL 180', 'Model_CLA 200',
 'Model_CLA-Class', 'Model_CLC 180', 'Model_CLC 200', 'Model_CLK 200',
 'Model_CLK 220', 'Model_CLK 230', 'Model_CLK 240', 'Model_CLK 280',
 'Model_CLK 320', 'Model_CLS 350', 'Model_Caddy', 'Model_Camry',
 'Model_Caravelle', 'Model_Carina', 'Model_Carisma', 'Model_Celica',
 'Model_Clio', 'Model_Colt', 'Model_Corolla', 'Model_Corolla Verso',
 'Model_Cross Touran', 'Model_Duster', 'Model_E-Class', 'Model_Eclipse',
 'Model_Eos', 'Model_Espace', 'Model_FJ Cruiser', 'Model_Fluence',
 'Model_Fortuner', 'Model_G 320', 'Model_GL 320', 'Model_GL 420',
 'Model_GLK 220', 'Model_GLK 300', 'Model_Galant', 'Model_Golf GTI',
 'Model_Golf III', 'Model_Golf IV', 'Model_Golf Plus', 'Model_Golf V',
 'Model_Golf VI', 'Model_Golf VII', 'Model_Golf Variant',
 'Model_Grand Scenic', 'Model_Grandis', 'Model_Hiace', 'Model_Highlander',
 'Model_Hilux', 'Model_I3', 'Model_IQ', 'Model_Jetta', 'Model_Kangoo',
 'Model_Koleos', 'Model_L 200', 'Model_LT', 'Model_Laguna', 'Model_Lancer',
 'Model_Lancer Evolution', 'Model_Lancer X', 'Model_Lancer X Sportback',
 'Model_Land Cruiser 100', 'Model_Land Cruiser 200', 'Model_Land Cruiser 80',
 'Model_Land Cruiser Prado', 'Model_Latitude', 'Model_Logan', 'Model_Lupo',
 'Model_M5', 'Model_ML 250', 'Model_ML 270', 'Model_ML 280', 'Model_ML 320',
 'Model_ML 350', 'Model_ML 400', 'Model_Master', 'Model_Matrix',
 'Model_Megane', 'Model_Modus', 'Model_Multivan', 'Model_New Beetle',
 'Model_Outlander', 'Model_Outlander XL', 'Model_Pajero',
 'Model_Pajero Pinin', 'Model_Pajero Sport', 'Model_Pajero Wagon',
 'Model_Passat B3', 'Model_Passat B4', 'Model_Passat B5', 'Model_Passat B6',
 'Model_Passat B7', 'Model_Passat CC', 'Model_Phaeton', 'Model_Pointer',
 'Model_Polo', 'Model_Previa', 'Model_Prius', 'Model_Q3', 'Model_Q5',
 'Model_Q7', 'Model_R 320', 'Model_Rav 4', 'Model_S 140', 'Model_S 300',
 'Model_S 320', 'Model_S 350', 'Model_S 400', 'Model_S4', 'Model_S5',
 'Model_S8', 'Model_SLK 200', 'Model_SLK 350', 'Model_Sandero', 'Model_Scenic',
 'Model_Scion', 'Model_Scirocco', 'Model_Sharan', 'Model_Sienna',
 'Model_Space Star', 'Model_Space Wagon', 'Model_Sprinter 208',
 'Model_Sprinter 210', 'Model_Sprinter 211', 'Model_Sprinter 212',
 'Model_Sprinter 213', 'Model_Sprinter 311', 'Model_Sprinter 312',
 'Model_Sprinter 313', 'Model_Sprinter 315', 'Model_Sprinter 316',
 'Model_Sprinter 318', 'Model_Sprinter 319', 'Model_Symbol', 'Model_Syncro',
 'Model_T4 (Transporter)', 'Model_T4 (Transporter) ',
 'Model_T5 (Transporter)', 'Model_T5 (Transporter) ',
 'Model_T6 (Transporter)', 'Model_T6 (Transporter) ', 'Model_TT',
 'Model_Tacoma', 'Model_Tiguan', 'Model_Touareg', 'Model_Touran',
 'Model_Trafic', 'Model_Up', 'Model_Vaneo', 'Model_Vento', 'Model_Venza',
 'Model_Viano', 'Model_Virage', 'Model_Vista', 'Model_Vito', 'Model_X1',
 'Model_X3', 'Model_X5', 'Model_X5 M', 'Model_X6', 'Model_Yaris', 'Model_Z3',
 'Model_Z4']
print(features_pvalues)
selected = features_pvalues[features_pvalues['pvalues'] < 0.001]
print(selected.shape)

print(data_with_dummies.shape)
data_final = data_with_dummies.loc[:, data_with_dummies.columns.isin(selected['Features'])]

print(data_final.shape)  # 51 columns down from 241

results = f_regression(data_final,y)
end_p_values = results[1].round(4)
print(end_p_values)  # These values are OK. We keep going with these variables.

# Linear regression model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

targets = y
inputs = data_final

print(targets.shape)
print(inputs.shape)
print(inputs.columns)

scaler = StandardScaler()
scaler.fit(inputs)
inputs_scaled = scaler.transform(inputs)

x_train, x_test, y_train, y_test = train_test_split(inputs_scaled, targets,test_size=0.2, random_state=365)

# Create regression
reg = LinearRegression()
reg.fit(x_train, y_train)
y_hat = pd.DataFrame(reg.predict(x_train))
print(f"dependent variable: {type(y_train)}")
print(f"Then, the predicted values: {type(y_hat)}")
print(y_hat.columns)


#Probability distribution function. Comparison after removing outliers
f2, axs = plt.subplots(1,2, sharey='col', figsize = (16,8))
axs[0].set_title("Price train data")
axs[0].hist(y_train)
axs[1].set_title("Price Predicted")
axs[1].hist(y_hat)
plt.show()



# verify y_hat is close y_train
plt.scatter(y_train, y_hat,alpha=0.2)
plt.xlabel('Targets (y_train)', size=18)
plt.ylabel('Predictions(y_hat)', size = 18)
plt.show()

#Check anomalies by looking into the differences
sns.histplot(y_train - y_hat, alpha=0.2)
plt.title("Residuals Probability Distribution Function", size=18)
plt.show() # With an average of 10.000 higher than train data, the model is overpricing

#Now to qualify the model
# R-squared
print(f"R-Squared: {reg.score(x_train, y_train)}")

print(f"Intercept: {reg.intercept_}")

print(f"Coefficients: {reg.coef_}")

print("Regression summary")
reg_summary = pd.DataFrame(inputs.columns.values, columns=['Features'])
reg_summary['Weights'] = reg.coef_
print(reg_summary)

# Testing
# check how well the model can predict.
y_hat_test = reg.predict(x_test)

y_test = y_test.reset_index(drop=True)

# Display the predictions
plt.scatter(y_test, y_hat_test, alpha=0.2)
plt.xlabel("Targest(y_test)", size=18)
plt.ylabel('Predictions (y_hat_test',size=18)
plt.show()  # This figure is pretty much like the one after the training. Model learned. It overprices, but it learned.

# Manually check the predictions
df_pf = pd.DataFrame(y_hat_test, columns=['Prediction'])
df_pf['Target'] = y_test
print (df_pf)

# Difference between targets and predictions
df_pf['Residual'] = df_pf['Target'] - df_pf['Prediction']

# percentage difference
df_pf['Difference%'] = np.absolute(df_pf['Residual']/df_pf['Target']*100)
print(df_pf)
print(df_pf.describe())

#This deviation from what was done during the course yield worse results. It is a worse model.