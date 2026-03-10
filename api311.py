import pandas as pd
import geopandas as gpd

from dash import Dash, html, dcc
import dash_leaflet as dl
from typing import Optional
from signatures import SignatureAnalyzer


class Year:
    def __init__(self, fp: str):
        self.data = pd.read_csv(fp, low_memory=False)
        self.gpd = None
        self.start_box = (42.3601, -71.0589)
        self.cache = None  # store subsets in cache - full data stays intact in data
        self.neighborhood_shapes = gpd.read_file(
            "data/neighborhood_shapes/Boston_Neighborhood_Boundaries.shp"
        )

    def make_points(self) -> None:
        self.data["geometry"] = gpd.points_from_xy(
            self.data.longitude, self.data.latitude
        )
        self.data = gpd.GeoDataFrame(self.data, geometry="geometry", crs="EPSG:4326")
        self.data = self.data.dropna(subset=["geometry"])
        self.data = self.data[~self.data.geometry.is_empty]
        self.cache = self.data  # by default assign full data to cache

    def get_subset(
        self, target_col: str, target_vals: list, cache: Optional[bool] = False
    ) -> pd.DataFrame:
        if cache:
            self.cache = self.data[self.data[target_col].isin(target_vals)]
        return self.data[self.data[target_col].isin(target_vals)]

    def _get_monthly_counts(
        self, col: str, full: Optional[bool] = True
    ) -> pd.DataFrame:
        """
        For a given column, return a dataframe with months as columns and counts of requests per month as values.
        If col is open_dt or closed_dt, return total counts per month.
        Otherwise, return counts per month for each unique value in col.

        Arguments:
        col: str -- the column for which to calculate monthly counts

        Returns:
        pd.DataFrame -- a dataframe with months as columns and counts of requests per month as values.
        """
        if full:
            source = self.data
        else:
            source = self.cache
        if source is not None:
            source["open_dt"] = pd.to_datetime(source["open_dt"])
            source["closed_dt"] = pd.to_datetime(source["closed_dt"])
            source["open_month_name"] = source["open_dt"].dt.month_name()
            source["open_month_num"] = source["open_dt"].dt.month
            source["closed_month_name"] = source["closed_dt"].dt.month_name()
            source["closed_month_num"] = source["closed_dt"].dt.month
        else:
            raise ValueError("No data available to calculate monthly counts.")

        if col in ("open_dt", "closed_dt"):
            monthly_groups = (
                source.groupby(["open_month_num", "open_month_name"])
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

            month_order = []
            """For each unique val in col, make months columns with counts of requests per month as corresponding row values"""
            monthly_requests = {}
            for month in source["closed_month_name"].unique():
                monthly_requests[month] = (
                    source[source["closed_month_name"] == month].groupby(col).size()
                )
                month_order = list(calendar.month_name)[1:]

            return (
                pd.DataFrame(monthly_requests)
                .reindex(columns=month_order)
                .dropna(axis=1, how="all")
                .fillna(0)
            )

    def _get_signatures(
        self,
        area_col: str = "neighborhood",
        type_col: str = "type",
        full: Optional[bool] = True,
    ) -> pd.DataFrame:
        if full:
            source = self.data
        else:
            source = self.cache
        if source is not None:
            sa = SignatureAnalyzer(area_col=area_col, type_col=type_col)
            sigs = sa.build_signatures(source, min_requests=30)
            return sigs
        else:
            raise ValueError("No data available to calculate signatures.")

    def summarize(
        self, category_col: str, target_col: str, full: Optional[bool] = True
    ) -> dict:
        describe = self.data[target_col].describe()
        monthly_counts = self._get_monthly_counts(target_col, full=full)
        unique_value_counts = self.data[target_col].value_counts()
        signatures = self._get_signatures(category_col, target_col, full=full)

        return {
            "describe": describe,
            "monthly": monthly_counts,
            "counts": unique_value_counts,
            "signatures": signatures,
        }
