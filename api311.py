import pandas as pd
import geopandas as gpd
import folium
from dash import Dash, html, dcc
import dash_leaflet as dl
from typing import Optional


class Year:
    def __init__(self, fp: str):
        self.data = pd.read_csv(fp, low_memory=False)
        self.gpd = None
        self.start_box = (42.3601, -71.0589)
        self.cache = None  # store subsets in cache - full data stays intact in data

    def make_points(self) -> None:
        self.data["geometry"] = gpd.points_from_xy(
            self.data.longitude, self.data.latitude
        )
        self.data = gpd.GeoDataFrame(self.data, geometry="geometry", crs="EPSG:4326")
        self.data = self.data.dropna(subset=["geometry"])
        self.data = self.data[~self.data.geometry.is_empty]
        self.cache = self.data  # by default assign full data to cache

    def serve_cache(self) -> folium.Map:
        map = folium.Map(location=self.start_box, zoom_start=10)

        for _, row in self.cache.iterrows():
            folium.Marker(
                [row["geometry"].y, row["geometry"].x],
                popup=row[["case_title", "subject", "location"]],
            ).add_to(map)
        return map

    def get_neighborhood_subset(
        self, neighborhood: str, cache: Optional[bool] = False
    ) -> pd.DataFrame:
        if cache:
            self.cache = self.data[self.data["neighborhood"] == neighborhood]
        return self.data[self.data["neighborhood"] == neighborhood]

    def _get_monthly_counts(self, col: str) -> pd.DataFrame:
        """
        For a given column, return a dataframe with months as columns and counts of requests per month as values.
        If col is open_dt or closed_dt, return total counts per month.
        Otherwise, return counts per month for each unique value in col.

        Arguments:
        col: str -- the column for which to calculate monthly counts

        Returns:
        pd.DataFrame -- a dataframe with months as columns and counts of requests per month as values.
        """
        self.data["open_dt"] = pd.to_datetime(self.data["open_dt"])
        self.data["closed_dt"] = pd.to_datetime(self.data["closed_dt"])
        self.data["open_month_name"] = self.data["open_dt"].dt.month_name()
        self.data["open_month_num"] = self.data["open_dt"].dt.month
        self.data["closed_month_name"] = self.data["closed_dt"].dt.month_name()
        self.data["closed_month_num"] = self.data["closed_dt"].dt.month

        if col in ("open_dt", "closed_dt"):
            monthly_groups = (
                self.data.groupby(["open_month_num", "open_month_name"])
                .aggregate("count")
                .reset_index()
            )
            monthly_requests = (
                monthly_groups.sort_values("open_month_num")
                .set_index("open_month_name")
                .drop("open_month_num", axis=1)[col]
            )
            return pd.DataFrame(monthly_requests)
        else:
            import calendar

            """For each unique val in col, make months columns with counts of requests per month as corresponding row values"""
            monthly_requests = {}
            for month in self.data["closed_month_name"].unique():
                monthly_requests[month] = (
                    self.data[self.data["closed_month_name"] == month]
                    .groupby(col)
                    .size()
                )
                month_order = list(calendar.month_name)[1:]
            return (
                pd.DataFrame(monthly_requests)
                .reindex(columns=month_order)
                .dropna(axis=1, how="all")
                .fillna(0)
            )

    def summarize(self, col: str) -> dict:
        describe = self.data[col].describe()
        monthly_counts = self._get_monthly_counts(col)
        unique = describe["unique"]
        unique_value_counts = self.data[col].value_counts()

        return {
            "describe": describe,
            "monthly": monthly_counts,
            "val_counts": unique_value_counts,
        }
