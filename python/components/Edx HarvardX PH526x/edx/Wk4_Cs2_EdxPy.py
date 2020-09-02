import numpy as np, pandas as pd
import matplotlib.pyplot as plt

birddata = pd.read_csv("bird_tracking.csv")
print (birddata.info())
print (birddata)

#######4.2.2###########
bird_names = pd.unique(birddata.bird_name)
print(bird_names)

plt.figure(figsize=(7,7))
for bird_name in bird_names:
    ix = birddata.bird_name == bird_name
    print(ix) # mark data set as true if birdname matches
    x,y = birddata.longitude[ix], birddata.latitude[ix]
    print(x,y)
    plt.plot(x,y,".",label = bird_name)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend(loc="lower right")
plt.savefig("3traj.pdf")
###############

data = np.array(['g', 'e', 'e', 'k', 's','e', 'k', 's','e', 'k', 's'])
ser = pd.Series(data)
print(ser)
print(pd.unique(ser))

#####4.2.3###########

plt.figure(figsize=(8,4))
ix = birddata.bird_name == "Eric"
speed = birddata.speed_2d[ix]
print(np.isnan(speed).any())
ind = np.isnan(speed) # get the index where speed is not number is NaN
plt.hist(speed[~ind], bins=np.linspace(0,30,20), normed=True) # normed mean normalize the histogram where integral is 1
plt.xlabel("2D speed (m/s)")
plt.ylabel("Frequency")
plt.savefig("hist_birdspeed.pdf")

### panda plots... plt plots are more vercitile
### panda plots internally handle NaN values
birddata.speed_2d.plot(kind='hist', range=[0,30])
plt.xlabel("2D speed (m/s)")
plt.ylabel("Frequency")
plt.savefig("hist_birdspeed_pd.pdf")

#######4.2.4############

import datetime
import time
print(birddata.columns)
print(datetime.datetime.today())

time_1 = datetime.datetime.today()
time.sleep(3)
time_2 = datetime.datetime.today()
time_delta = time_2 - time_1
print(type(time_delta))
print(time_delta)

print(birddata.date_time[0:3])

date_str = birddata.date_time[0]
print(date_str[:-3])

d = datetime.datetime.strptime(date_str[:-3], "%Y-%m-%d %H:%M:%S")
print(d)

timestamps = []
for k in range(len(birddata)):
    timestamps.append(datetime.datetime.strptime(birddata.date_time.iloc[k][:-3], "%Y-%m-%d %H:%M:%S"))
print(timestamps)
birddata["timestamp"] = pd.Series(timestamps, index = birddata.index)
print(birddata)

times = birddata.timestamp[birddata.bird_name == "Eric"]
elapsed_time = [time - times[0] for time in times]
print(elapsed_time)
elapsed_time_days = np.array(elapsed_time) / datetime.timedelta(days = 1)
print(elapsed_time_days)
elapsed_time_hours = np.array(elapsed_time) / datetime.timedelta(hours = 1)
print(elapsed_time_hours)

plt.plot(elapsed_time_days)
plt.xlabel("Observations")
plt.ylabel("Elapsed time (days)")

# xmin, xmax = plt.xlim()
# ymin, ymax = plt.ylim()
# print(xmin, xmax)
# print(ymin, ymax )
# scale_factor = 0.05
# plt.xlim(xmin * scale_factor, xmax * scale_factor)
# plt.ylim(ymin * scale_factor, ymax * scale_factor)
plt.savefig("timeplot.pdf") # the scaling is not correct for this plot




#######4.2.4############
data = birddata[birddata.bird_name == "Eric"]
times = data.timestamp
elapsed_time = [time - times[0] for time in times]
elapsed_days = np.array(elapsed_time) / datetime.timedelta(days = 1)

next_day = 1
inds = []
daily_mean_speed = []
for (i,t) in enumerate(elapsed_days):
    if t < next_day:
        inds.append(i)
    else:
        daily_mean_speed.append(np.mean(data.speed_2d[inds]))
        next_day += 1
        inds = []

plt.figure(figsize=(8,6))
plt.plot(daily_mean_speed)
plt.xlabel("Days")
plt.ylabel("Mean Speed (m/s)")
plt.savefig("dms.pdf")

#
data = birddata[birddata.bird_name == "Sanne"]
print(data)
times = data.timestamp
print(times)


##########4.2.6#############
print("### Cartopy")
import cartopy.crs as ccrs
import cartopy.feature as cfeature

proj = ccrs.Mercator()

plt.figure(figsize=(10,10))
ax = plt.axes(projection = proj)
ax.set_extent((-25.0, 20.0, 52.0, 20.0))
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')
print(type(ax))

for name in bird_names:
    ix = birddata['bird_name'] == name
    x, y = birddata.longitude[ix], birddata.latitude[ix]
    ax.plot(x,y,'.', transform=ccrs.Geodetic(), label=name)

plt.legend(loc = "upper left")
plt.savefig("map.pdf")

