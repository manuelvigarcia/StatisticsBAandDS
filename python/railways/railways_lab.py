import pandas as pd
import datetime
#import numpy as np

#import matplotlib.pyplot as plt
#import seaborn as sns

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
    # Verify that only Madrid and Barcelona are origin and destination
    print("Verify vehicle classes involved . . .")
    types = {"AVE"}
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


print("Load data . . .")
train = pd.read_csv("train_ticket_data.csv")
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
#check_type(train)   # {'LD-MD', 'AVE-TGV', 'AVE-LD', 'MD-AVE', 'AVE', 'LD-AVE', 'R. EXPRES'} 'AVE' and 'AVE-TGV' to be High-Sppeed; rest: Standard


print("Change departure string to datetime")
print(f"departure: {type(train.iat[0,3])}")
train['departure'] = pd.to_datetime(train['departure'])
print(f"departure: {type(train.iat[0,3])}")


print(f"Current size of the dataframe: {train.size}")  #13129974

print("Add extra columns. . .")

train['High-Speed'] = train['vehicle_type'] == 'AVE' or train['vehicle_type'] == 'AVE-TGV'

print(train.info())

print(train.head(10))
print(train.tail(10))



