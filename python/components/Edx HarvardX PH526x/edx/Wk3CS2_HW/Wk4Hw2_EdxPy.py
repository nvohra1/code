import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime



birddata = pd.read_csv("https://courses.edx.org/asset-v1:HarvardX+PH526x+2T2019+type@asset+block@bird_tracking.csv", index_col=0)
birddata.head()
print(birddata)

###################################
# First, use `groupby` to group up the data.
grouped_birds = birddata.groupby("bird_name")
print(grouped_birds)
# Now operations are performed on each group.
mean_speeds = grouped_birds.speed_2d.mean()
print("mean speed  ", mean_speeds)

# The `head` method prints the first 5 lines of each bird.
#grouped_birds.head()

# Find the mean `altitude` for each bird.
# Assign this to `mean_altitudes`.
mean_altitudes = grouped_birds.altitude.mean()
print("mean_altitudes  ", mean_altitudes)
################

# Convert birddata.date_time to the `pd.datetime` format.
birddata.date_time = pd.to_datetime(birddata.date_time)
print(birddata.date_time)
# Create a new column of day of observation
birddata["date"] = birddata.date_time.dt.date
print(birddata["date"] )
# Check the head of the column.
#birddata.date.head()
print(birddata.date)

grouped_bydates = birddata.groupby("date")
print(grouped_bydates)
mean_altitudes_perday = grouped_bydates.altitude.mean()
print(type(mean_altitudes_perday))
print(mean_altitudes_perday)
mean_altitudes_perday['date'].value[1]'.date["2013-09-12"]

######################
grouped_birdday = birddata.groupby(["bird_name", 'date'])


mean_altitudes_perday = grouped_birdday.altitude.mean()

# look at the head of `mean_altitudes_perday`.
mean_altitudes_perday.head()

#########################

eric_daily_speed  = grouped_birdday.speed_2d.mean()['Eric']
sanne_daily_speed = grouped_birdday.speed_2d.mean()['Sanne']
nico_daily_speed  = grouped_birdday.speed_2d.mean()['Nico']

eric_daily_speed.plot(label="Eric")
sanne_daily_speed.plot(label="Sanne")
nico_daily_speed.plot(label="Nico")
plt.legend(loc="upper left")
plt.show()