import json
from datetime import datetime
import pandas as pd
import isodate
import numpy as np
import matplotlib.pyplot as plt

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def items_to_rows(data, country):
    rows = []
    for it in data.get("items", []):
        sid = it.get("id")
        snip = it.get("snippet", {})
        stats = it.get("statistics", {})
        cd = it.get("contentDetails", {})
        published = snip.get("publishedAt")
        try:
            published = pd.to_datetime(published)
        except:
            published = pd.NaT
        duration = cd.get("duration")
        try:
            duration_s = int(isodate.parse_duration(duration).total_seconds()) if duration else np.nan
        except:
            duration_s = np.nan
        def toint(x):
            try:
                return int(x)
            except:
                return np.nan
        viewCount = toint(stats.get("viewCount"))
        likeCount = toint(stats.get("likeCount"))
        commentCount = toint(stats.get("commentCount"))
        tags = snip.get("tags") or []
        rows.append({
            "country": country,
            "videoId": sid,
            "title": snip.get("title"),
            "channelTitle": snip.get("channelTitle"),
            "categoryId": snip.get("categoryId"),
            "publishedAt": published,
            "duration_s": duration_s,
            "viewCount": viewCount,
            "likeCount": likeCount,
            "commentCount": commentCount,
            "tags_count": len(tags)
        })
    return rows

data_fr = load_json("youtube_mostpopular_fr.json")
data_us = load_json("youtube_mostpopular_us.json")
data_in = load_json("youtube_mostpopular_in.json")

categories_fr_json = load_json("ytb_categories_fr.json")
categories_us_json = load_json("ytb_categories_us.json")
categories_in_json = load_json("ytb_categories_in.json")

map_fr = {str(item["id"]): item["snippet"]["title"] for item in categories_fr_json.get("items", [])}
map_us = {str(item["id"]): item["snippet"]["title"] for item in categories_us_json.get("items", [])}
map_in = {str(item["id"]): item["snippet"]["title"] for item in categories_in_json.get("items", [])}
maps = {"FR": map_fr, "US": map_us, "IN": map_in}

rows = []
rows += items_to_rows(data_fr, "FR")
rows += items_to_rows(data_us, "US")
rows += items_to_rows(data_in, "IN")

df = pd.DataFrame(rows)
df["categoryId"] = df["categoryId"].astype(str)
df["viewCount"] = df["viewCount"].fillna(0)

def map_category(row):
    m = maps.get(row["country"], {})
    return m.get(row["categoryId"], "Unknown")

df["categoryName"] = df.apply(map_category, axis=1)

categories_fr = df[df.country == "FR"].groupby("categoryName")["viewCount"].sum().sort_values(ascending=False).head(10)
categories_us = df[df.country == "US"].groupby("categoryName")["viewCount"].sum().sort_values(ascending=False).head(10)
categories_in = df[df.country == "IN"].groupby("categoryName")["viewCount"].sum().sort_values(ascending=False).head(10)

fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(18,6))
ax1.barh(categories_fr.index[::-1], categories_fr.values[::-1])
ax1.set_title("France")
ax1.set_xlabel("View count")
ax1.invert_yaxis()

ax2.barh(categories_us.index[::-1], categories_us.values[::-1])
ax2.set_title("US")
ax2.set_xlabel("View count")
ax2.invert_yaxis()

ax3.barh(categories_in.index[::-1], categories_in.values[::-1])
ax3.set_title("India")
ax3.set_xlabel("View count")
ax3.invert_yaxis()

plt.tight_layout()
plt.show()

# Basic statistics
countries = df.groupby("country")["viewCount"].mean()
plt.figure(figsize=(10,5))
plt.bar(countries.index, countries.values)
plt.xlabel("Country")
plt.ylabel("Average view count")
plt.title("Average view count per country")
plt.show()
