import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import contextily as ctx
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point


station_avg_risk = pd.read_parquet("data/station_avg_risk.parquet")
station_df = pd.read_parquet("data/station_features.parquet")


# Merge station_avg_risk with coordinates from station_df

station_df.rename(columns={"LocationSignature": "station_code"}, inplace=True)


res = station_avg_risk.set_index("station_code").join(station_df.set_index("station_code"), on=["station_code"])


# Create geometry
geometry = [
    Point(lon, lat) for lon, lat in zip(res["lon"], res["lat"])
]

gdf = gpd.GeoDataFrame(res, geometry=geometry, crs="EPSG:4326")

gdf = gdf.to_crs(epsg=3857)

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 10))

gdf.plot(
    ax=ax,
    column="station_avg_risk",
    cmap="viridis",
    vmin=0,
    vmax=1,
    legend=True,
    legend_kwds={"label": "Predicted value"},
    markersize=50,
    alpha=0.9
)


ctx.add_basemap(
    ax,
    source=ctx.providers.CartoDB.Positron
)
ax.set_title("Predicted values per train station (Stockholm)")
ax.axis("off")

plt.savefig("docs/assets/img/station_delay_risk_map.png")
plt.clf()
# Bar chart of top 10 stations with highest average risk
top10 = res.sort_values("station_avg_risk", ascending=False).head(10)
fig, ax = plt.subplots(figsize=(10, 10))
top10.plot(
    ax=ax,
    kind="bar",
    x="AdvertisedLocationName",
    y="station_avg_risk",
    #color="station_avg_risk",
    cmap="viridis",
    legend=False
)
ax.set_title("Top 10 Stations with Highest Average Delay Risk")
ax.set_xlabel("Station")
ax.set_ylabel("Average Delay Risk")
plt.savefig("docs/assets/img/top10_station_delay_risk.png")
