import pandas as pd
import datetime
import random
import matplotlib.pyplot as plt
#import numpy as np
# #import seaborn as sns

def check_date_columns(df):
    print("Verifying date and time columns")
    for i in range(0, 20):
        idx = random.choice(df.index)
        print(f"Original string: {df.at[idx,'departure']}, extracted date: {df.at[idx,'departure_date']}, extracted time: {df.at[idx,'departure_hour']}")
        print(f"Departure:       {df.at[idx,'departure']}, arrival:        {df.at[idx,'arrival']},        duration:       {df.at[idx,'duration']}, HH:MM:SS: {df.at[idx,'duration_delta']}")
    date_columns = df.loc[:,['departure', 'duration','departure_hour','departure_date','departure_dayofweek','duration_delta','arrival']]
    print(date_columns.describe())

def high_speed_label(is_high_speed):
    if (is_high_speed):
        return 'High-Speed'
    return 'Standard'


def speed_level(str):
    print(f"type of str: {type(str)}")
    if str.item() == 'AVE' or str.item() == 'AVE-TGV':
        return 'High-Speed'
    return 'Standard'

def play_with_dates():
    formatstr = "%Y-%m-%d %H:%M:%S"
    date_string = train.iat[0, 3]
    date = datetime.datetime.strptime(date_string, formatstr)
    print(f"this string <{date_string}> gives this date: {date}; dia {date.day}, mes {date.month}, año {date.year}")


def check_cities(dataframe):
    # Verify that only Madrid and Barcelona are origin and destination
    print("Verify cities involved . . .")
    cities = {"MADRID", "BARCELONA"}
    for i in range(0, len(dataframe.index) - 1):
        origin = dataframe.iat[i, 1]
        if origin not in cities:
            cities.add(origin)
            print(origin)
        destination = dataframe.iat[i, 2]
        if destination not in cities:
            cities.add(destination)
            print(destination)

    print(cities)

def check_type(dataframe):
    # See which types of trains are used
    print("Verify vehicle classes involved . . .")
    types = {"AVE-TGV"}
    for i in range(0, len(dataframe.index) - 1):
        v_type = dataframe.iat[i, 5]
        if v_type not in types:
            types.add(v_type)
            print(v_type)
    print(types)


def check_valid_rows(df):
    valid_rows = df.count()  # --> number of rows not-NaN in each column (it is a series)
    print(f"number of rows in each column that do not have NaN:\n {valid_rows}")
    print(type(valid_rows))

def export_invalid_rows(df):
    filename = datetime.datetime.now().__str__() + "_invalid.csv"
    filename = filename.replace(":", "_")
    try:
        with open(filename, "x") as f:
            f.write("type,class,price,fare\n")
            for i in range(0, len(df.index) - 1):
                v_type  = df.iat[i, 5]
                v_class  = df.iat[i, 6]
                price  = df.iat[i, 7]
                fare  = df.iat[i, 8]
                if v_type != 'AVE':
                    #line = v_type + "," + v_class + "," + price.__str__() + "," + fare + "\n"
                    #line = "{},{},{},{}\n".format(v_type, v_class, price, fare)
                    line = f"{v_type},{v_class},{price},{fare}\n"
                    f.write(line)
    except FileExistsError:
        print("File already exists.")

def check_high_speed_columns(df):
    high_speed_columns = df.loc[:,['vehicle_type','vehicle_class','vehicle_category']]
    print(high_speed_columns.describe())

def export_high_speed_columns(df):
    filename = datetime.datetime.now().__str__() + ".csv"
    filename = filename.replace(":", "_")
    try:
        with open(filename, "x") as f:
            f.write("type,isHighSpeed,Speed\n")

            for i in range(0, len(df.index) - 1):
                v_type = df.iat[i, 5]
                # hs_bool= df.iat[i, 9].__str__()
                speed = df.iat[i, 9]
                # speed2 = df.iat[i, 11]
                if v_type != 'AVE':
                    # line = f"{v_type},{hs_bool},{speed},{speed2}\n"
                    # line = f"{v_type},{hs_bool},{speed}\n"
                    line = f"{v_type},{speed}\n"
                    f.write(line)
    except FileExistsError:
        print("File already exists.")

def print_counts_for_column_value(df, column, value):
    print(df[df[column].eq(value)].count())

print("Load data . . .")
train = pd.read_csv("train_ticket_data.csv")
#train = train.iloc[0:5000,:]
print(f"Original size of the dataframe: {train.size}")  # 13733550
print("dataframe info:")
print(f"type of the whole DataFrame: {type(train)}")
print(train.info())
print("dataframe abstract:")
print(train)
print(f"first cell){type(train.iat[0,0])}")
print("first row (formatted:")
print(train.loc[0]) # --> prints first row formatted
print(f"first vehicle_class: {train.at[0,'vehicle_class']}")

print("Visualize NaN and remove them")
check_valid_rows(train)
# export_invalid_rows(train)
train = train.dropna(axis=0)
check_valid_rows(train)

print("values in the first row:")
print(train.at[0,'Unnamed: 0'])     # --> numpy.int64
print(train.at[0,'origin'])         # --> str
print(train.at[0,'destination'])    # --> str
print(train.at[0,'departure'])      # --> str
print(train.at[0,'duration'])       # --> numpy.float64
print(train.at[0,'vehicle_type'])   # --> str
print(train.at[0,'vehicle_class'])  # --> str
print(train.at[0,'price'])          # --> numpy.float64
print(train.at[0,'fare'])           # --> str

# types of all columns
print(f"Unnamed: {type(train.iat[0,0])}\nOrigin: {type(train.iat[0,1])}\ndestination: {type(train.iat[0,2])}\n")
print(f"departure: {type(train.iat[0,3])}\nduration: {type(train.iat[0,4])}\nVehicle_type: {type(train.iat[0,5])}\n")
print(f"vehicle_class: {type(train.iat[0,6])}\nprice: {type(train.iat[0,7])}\nFare: {type(train.iat[0,8])}\n")

#check_cities(train) # {'BARCELONA', 'MADRID'}

# {'LD-MD', 'AVE-TGV', 'AVE-LD', 'MD-AVE', 'AVE', 'LD-AVE', 'R. EXPRES'} before cleaning NaN
"""
1.405.766 AVE --> High-Speed
   58.652 AVE-TGV --> High-Speed
      124 LD-MD  ninguno tiene precio --> se van como NaN
    4.478 AVE-LD ninguno tiene precio --> se van como NaN
      296 MD-AVE ninguno tiene precio --> se van como NaN
    7.380 LD-AVE ninguno tiene precio --> se van como NaN
   49.254 R. EXPRES --> Standard son los único "Standard", por eso petaba mientras había el error de EXPRESS
"""

# check_type(train)   # {'R. EXPRES', 'AVE-TGV', 'AVE'} 'AVE' and 'AVE-TGV' to be High-Sppeed; rest: Standard


print("Change departure string to datetime date and time")
print(f"departure: {type(train.iat[0,3])}")
train['departure'] = pd.to_datetime(train['departure'])
print(f"departure: {type(train.iat[0,3])}")
print(f"\nCurrent size of the dataframe: {train.size}\n")  #13129974
print("Add extra columns. . .")

print("\thour and date")
train['departure_hour'] = train['departure'].map(lambda x: x.time())
train['departure_date'] = train['departure'].map(lambda x: x.date())
print("\tday of the week")
train['departure_dayofweek'] = train['departure'].map(lambda x: x.weekday()) #Sunday = 6
print("\tarrival")
train['duration_delta'] = pd.to_timedelta(train['duration'],"hours","raise")
train['arrival'] = train['departure'] + train['duration_delta']


print("\tvehicle_category")
# 108299900 ns assigning with apply
# 104585400 ns assigning with map
# train['High-Speed'] = train['vehicle_type'].eq('AVE') | train['vehicle_type'].eq('AVE-TGV')
train['vehicle_category'] = train['vehicle_type'].apply(lambda x:'High-Speed' if x == 'AVE' or x == 'AVE-TGV' else 'Standard')

print("Check created columns . . .")
# export_high_speed_columns(train)
check_high_speed_columns(train)
check_date_columns(train)
print("Dataframe stats")
print(train.info())
print(train.head(10))
print(train.tail(10))
print(train.describe())



# # Data extraction
print("Data Extraction . . .")
print("description")
train_description= train.describe()
print(f"average Price: {train_description.loc['mean', 'price']}")
print(f"minimum Price: {train_description.loc['min', 'price']}")
print(f"maximum Price: {train_description.loc['max', 'price']}")
print("price")
# Average price
average_price = train['price'].mean()
print(f'Average price: {round(average_price, 2)}')
# Row with minimum price
print("minimum")
min_price = train.nsmallest(1,'price', 'first').iloc[0]
print(min_price)
# Row with maximum price
print("maximum")
max_price = train.nlargest(1,'price', 'first').iloc[0]
print(max_price)
# Average trip duration 'duration'
print(f"average duration: {train_description.loc['mean', 'duration']}")
print(f"minimum duration: {train_description.loc['min', 'duration']}")
print(f"maximum duration: {train_description.loc['max', 'duration']}")
# Row with minimum trip duration
min_time = train.nsmallest(1,'duration', 'first').iloc[0]
# Row with maximum trip duration
max_time = train.nlargest(1,'duration', 'first').iloc[0]
# Compare prices between Madrid and Barcelona as origin
from_madrid = train.loc[train['origin'].eq('MADRID')]
avg_price_from_madrid = from_madrid.describe().loc['mean','price']
from_barcel = train.loc[train['origin'].eq('BARCELONA')]
avg_price_from_barcel = from_barcel.describe().loc['mean','price']
print(f"Average prices: from Madrid: {avg_price_from_madrid}\tfrom Barcelona: {avg_price_from_barcel}")
print(from_madrid.describe())
print(from_barcel.describe())

print("\nPlot data. . .")
# Line Chart with daily riders vs. time and daily revenue vs. time
print("First, select departure_time and price columns only")
price_vs_time=train.loc[:,['departure_date','price']]
print(price_vs_time.info())
print(price_vs_time.head())

print("Then group by date and count() to see how many travelled each date")
riders_per_date=price_vs_time.groupby(['departure_date']).count().reset_index()
print(riders_per_date.info())
print(riders_per_date.head(31))
riders_per_date['price'].plot(marker='x', ylabel='x = passengers', xlabel='date')

print("then group by date and sum() to calculate the revenue on each date")
revenue_per_date=price_vs_time.groupby(['departure_date']).sum().reset_index()
print(revenue_per_date.info())
print(revenue_per_date.head(31))
revenue_per_date['price'].plot(marker='o', secondary_y=True, ylabel='o = revenue')

plt.show()

# Bar chart with daily riders from Madrid and from Barcelona to compare
riders_from_barcel=from_barcel.loc[:,['departure_dayofweek','price']].groupby(['departure_dayofweek']).count()
riders_from_barcel = riders_from_barcel.rename(columns={'price':'From Barcelona'})
#riders_from_barcel['price'].plot(kind='bar', ylabel='pasengers', xlabel='date', title='Pasengers according to origin')
riders_from_madrid=from_madrid.loc[:,['departure_dayofweek','price']].groupby(['departure_dayofweek']).count()
riders_from_madrid = riders_from_madrid.rename(columns={'price':'From Madrid'})
riders = pd.concat([riders_from_barcel, riders_from_madrid], axis=1)
#riders_from_madrid['price'].plot(xlabel='date')
riders.plot(kind='bar', xlabel='Monday to Sunday')
plt.show()

# Bar chart Average price by vehicle_category
price_by_category = train.loc[:,['price','vehicle_category']].groupby(['vehicle_category']).mean()
price_by_category.plot(xlabel='vehicle_category', kind='bar')
plt.show()

# Bar chart Average trip duration by vehicle_category
duration_by_category = train.loc[:,['duration', 'vehicle_category']].groupby(['vehicle_category']).mean()
duration_by_category.plot(kind='bar', xlabel='trip duration')
plt.show()
# Heat map of average price per day of week (column) and departure time (row)
print('heatmap')
price_by_day_hour=train.loc[:,['price','departure_hour','departure_dayofweek']].groupby(['departure_dayofweek','departure_hour'])
print(price_by_day_hour.groups)
heat_map = price_by_day_hour.mean()
heat_map.info()
heat_map.head()



