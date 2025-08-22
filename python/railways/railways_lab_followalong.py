"""

Cleaning, Analyzing, and Visualizing Spanish Train Ticket Data with Python
Ending...
Your project assignment
YM
Your Manager
Aug 21, 9:45 AM
Hi Manuel Vicente! Thanks for working on this project for our team.

As our newest traffic analyst at Fuerte, a Spanish Train company, you have been tasked with analyzing ridership patterns in the train system. Management wants us to produce summary reports on general ridership  that examine the volume of passengers, ticket prices paid, and length of trips taken by our customers. The primary focus is producing a webpage that gives customers a better idea of what the best travel times and fares are depending on their needs, which will reduce the amount of time our customer service agents spend explaining this information. For now, we'll focus on trips between Madrid and Barcelona, our busiest route. First, read in the data provided, train_ticket_data.csv. Check the column data types, and change them if needed. Then, create four columns and report on the percentage of missing values in each column, including the following:

    arrival: add the duration column (in hours) to the departure column,

    departure_dayofweek: extract the day of week (as an integer or text) from departure.

    departure_hour: extract the hour from departure.

    vehicle_category: 'AVE' and 'AVE-TGV' should be classified as 'High-Speed'; the rest classified as 'Standard'.

Then, collect the following information for a report to management:

    The average, minimum, and maximum prices and durations of trips (examine the rows where the minimum and maximum occur).

    Whether or not prices for trains departing from Madrid are more expensive than those departing from Barcelona.

    A dual axis line chart with the number of total daily riders as one line, and total daily revenue as the other.

    A bar chart of average ridership by day of week and departure location to visualize whether more riders take the Madrid-Barcelona train or vice versa on a daily basis.

    Build two bar charts (average price by vehicle_category and average duration by vehicle category) to help  customers get an idea of the tradeoffs in time and cost for high-speed vs. standard trains.

    A heatmap of average price with day of week as columns and time of day as rows. Use this to determine which departure times and days should be avoided if customers are looking for a value. Export the table used to build the heatmap to a flat file format of your choice to serve as a prototype for our customer help page.

How you'll work
Your project has been broken into a set of tasks. To complete these tasks, use the provided workspace. You can launch your workspace by clicking below or using the button in the top right of the screen.
Each task includes step-by-step instructions as well as helpful documentation and necessary assets in the Resources section.

"""
#Read in the dataset train_data.csv using the read_csv function.
#    train = pd.read_csv('train_ticket_data.csv') # will be rewritten at step 2

#Call the head and info methods on the DataFrame to get a basic understanding of the data.
#   train.head() #will be rewritten as a more advanced version
#   train.info() #will be rewritten as a more advanced version

#Check the variable types and change the object data types to category and the departure column to datetime64.
# Changing object to category generally saves significant memory, while casting departure as datetime64
# makes possible to use datetime functions. This can be done with the .astype() method, but we can streamline
# our workflow by casting datatypes within read_csv().

import pandas as pd
from pandas.api.types import CategoricalDtype
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

print("Load data . . .")
train = pd.read_csv('train_ticket_data.csv',
                    parse_dates=['departure'],
                    dtype={
                        'origin':'category',
                        'destination':'category',
                        'vehicle_type':'category',
                        'vehicle-class':'category',
                        'fare':'category',
                    }
                    ).drop('Unnamed: 0', axis=1)
print(train.head())
print(train.info())
print(f"Original size of the dataframe: {train.size}") #12207600 vs 13733550

#Create four columns and report on the percentage of missing values in each column.

# created ordered category for weekday. integer dayofweek is ok, but less helpful for customers
cat_type = CategoricalDtype(
    categories=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
    ordered=True
)

# Create new columns: arrival, departure day_of_week and hour, vehicle_category
train = train.assign(
    arrival = train['departure'] + pd.to_timedelta(train['duration'], unit='h'),
    departure_dayofweek = (train['departure'].dt.day_name().astype(cat_type)),
    departure_hour = train['departure'].dt.hour,
    vehicle_category = (train['vehicle_type'].map({"AVE": "High-Speed",
                                                  "AVE-TGV": "High-Speed",
                                                  "R. EXPRES": "Standard",
                                                  "LD-AVE": "Standard",
                                                  "AVE-LD": "Standard",
                                                  "MD-AVE": "Standard",
                                                  "LD-MD": "Standard",
                                                  }))
)

#Eliminate rows with NaN in any column
print("\nVer datos no NaN:")
print(train.count())
train = train.dropna(axis=0)
print("Y despues de quitar NaNs:")
print(train.count())
print("datos de trenes LD-AVE:")
print(train.loc[train['vehicle_type'].eq('LD-AVE')].head())
#Key summary statistics for Management
#  See a summary statistics and identify price values
print(train.describe().round())

# filter rows where price is the max
print(train.loc[train["price"] == train['price'].max()])

# filter rows where price is the min
print(train.loc[train["price"] == train['price'].min()])

#  Calculate the mean price  of trains leaving Madrid and Barcelona by
#  filtering down to the departure location of interest and taking the mean of the price column.
print(f"Average price MADRID --> BARCELONA: {round(train.query("origin == 'MADRID'").price.mean(),4)}")    # 86.0155
print(f"Average price BARCELONA --> MADRID: {round(train.query("origin == 'BARCELONA'").price.mean(),4)}") # 86.3311


#Build a dual axis line chart with the number of total riders as one line and total revenue as the other.
# The x-axis be all the dates in the dataset. Use matplotlib's .plot() function, with the help of .twinx() to create a second axis.
# First, group data.
train_summary = (train.groupby(train['departure'].dt.date)
                 .agg(ridership=('price','count'),
                      total_fare=("price", 'sum'),
                      )
                 ).reset_index()
print(train_summary.head())

# dual y-axis plot
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_title("Madrid-Barcelona Daily Ridership and Revenue May 2019")
ax.plot(train_summary["departure"],
        train_summary["ridership"] / 1000, #Display in K
        c="#830065",
        label="Ridership"
       )
ax.set_ylabel("Ridership (Thousands)",fontsize=12)
ax2 = ax.twinx()
ax2.plot(train_summary["departure"],
         train_summary["total_fare"] / 1000000, #Display in Millions
         c="#666366",
         label="Revenue"
        )
ax2.set_ylabel("Revenue (Millions)",fontsize=12)
fig.autofmt_xdate(rotation=45)
fig.legend(bbox_to_anchor=(.9,.3))
plt.show()

#Build a bar chart of average ridership by day of week and departure location to visualize this.
# Determine if one route is always busier or if it depends on the weekday.

# For the bar chart, aggregate the original DataFrame once again.
# Group the DataFrame by departure_dayofweek and origin.
weekly_ridership = (train.groupby(['departure_dayofweek','origin'], observed=False)
# Calculate the count of rows for each day of week and departure combination, using .groupby(), and .agg().
                    .agg(ridership=('price','count'))
# Then, reset the DataFrame index with reset_index(), which will make selecting the necessary columns for plotting a bit easier.
                    ).reset_index()
print(weekly_ridership.head())
# Use seaborn's barplot function, sns.barplot(x, y, hue, data), to build the grouped bar chart.
# Set x as departure_dayofweek, y as ridership, and hue as origin.
sns.barplot(x='departure_dayofweek', y='ridership',hue='origin', data=weekly_ridership)
sns.despine()
plt.show()
#or, with detailed formatting
fig,ax = plt.subplots(figsize=(12, 8))
sns.barplot(
    data=weekly_ridership,
    x='departure_dayofweek',
    y='ridership',
    hue='origin',
    palette = ["#830065","#666366"]
).set(
    title="Ridership by Weekday and Departure Location",
    xlabel="Weekday",
    ylabel="Ridership Volume")
sns.despine()
plt.show()

# Average price by vehicle_category
#  Aggregate the ticket DataFrame, grouping by vehicle_category with .groupby().
#  Calculate the mean of the price and duration variables by category using .agg().
train_summary = (
    train.groupby('vehicle_category')
    .agg(
        average_fare=("price", 'mean'),
        average_duration=('duration', 'mean')
    )
)
print(train_summary)

#Build a bar chart using plt.bar(), with 'x' as vehicle_category and the 'height' of each bar as the average price.
# Produce a second bar chart with 'x' as vehicle_category and the 'height 'of each bar as the average_duration.
# Consider using plt.subplots(1, 2) to plot these side by side in the same figure.
fig, ax = plt.subplots(1,2,figsize=(8,5))
fig.suptitle("Standard Trains are Cheaper, but take much longer")
ax[0].bar(
    x=train_summary.index,
    height=train_summary['average_fare'],
    color = '#830065',
)
ax[0].set_title("Price by Train Type")
ax[0].set_ylabel("Price (Euro)")
ax[1].bar(
    x=train_summary.index,
    height=train_summary['average_duration'],
    color = '#666366',
)
ax[1].set_title("Trip Duration by Train Type")
ax[1].set_ylabel("Trip Duration (Hours)")
plt.show()

# Average duration by vehicle category.
