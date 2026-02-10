from dash import Dash, html, dcc, callback, Output, Input
import folium
from folium.plugins import MarkerCluster
from api311 import Year

year15 = Year("data/cleaned2015.csv")
year25 = Year("data/cleaned2025.csv")
year15.make_points()
year25.make_points()

neighborhoods = sorted(year15.data["neighborhood"].dropna().unique().tolist())

app = Dash(__name__)

app.layout = html.Div(
    [
        html.Div(
            [
                html.H2("Boston 311 Requests"),
                html.Label("Neighborhood"),
                dcc.Dropdown(
                    id="neighborhood-filter",
                    options=[{"label": n, "value": n} for n in neighborhoods],
                    value=None,
                    placeholder="All Neighborhoods",
                ),
                html.Label("Request Type"),
                dcc.Dropdown(
                    id="type-filter",
                    options=[],
                    value=None,
                    placeholder="All Types",
                ),
            ],
            style={
                "width": "20%",
                "padding": "20px",
                "position": "fixed",
                "height": "100vh",
                "overflowY": "auto",
            },
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.H3("2015"),
                        html.Iframe(
                            id="map-2015",
                            style={"height": "90vh", "width": "100%", "border": "none"},
                        ),
                    ],
                    style={"width": "50%", "display": "inline-block"},
                ),
                html.Div(
                    [
                        html.H3("2025"),
                        html.Iframe(
                            id="map-2025",
                            style={"height": "90vh", "width": "100%", "border": "none"},
                        ),
                    ],
                    style={"width": "50%", "display": "inline-block"},
                ),
            ],
            style={"marginLeft": "22%"},
        ),
    ]
)


def build_map(data, n=5000):
    df = data.copy()
    if len(df) > n:
        df = df.sample(n=n, random_state=42)

    m = folium.Map(location=[42.3601, -71.0589], zoom_start=12)
    cluster = MarkerCluster().add_to(m)

    for _, row in df.iterrows():
        folium.Marker(
            location=[row.geometry.y, row.geometry.x],
            popup=f"{row['case_title']}<br>{row['location']}",
        ).add_to(cluster)

    return m._repr_html_()


@callback(
    Output("type-filter", "options"),
    Input("neighborhood-filter", "value"),
)
def update_type_options(neighborhood):
    df = year15.data
    if neighborhood:
        df = df[df["neighborhood"] == neighborhood]
    types = sorted(df["type"].dropna().unique().tolist())
    return [{"label": t, "value": t} for t in types]


@callback(
    Output("map-2015", "srcDoc"),
    Output("map-2025", "srcDoc"),
    Input("neighborhood-filter", "value"),
    Input("type-filter", "value"),
)
def update_maps(neighborhood, req_type):
    def filter_data(data):
        df = data.copy()
        if neighborhood:
            df = df[df["neighborhood"] == neighborhood]
        if req_type:
            df = df[df["type"] == req_type]
        return df

    df15 = filter_data(year15.data)
    df25 = filter_data(year25.data)

    return build_map(df15), build_map(df25)


if __name__ == "__main__":
    app.run(debug=True)
