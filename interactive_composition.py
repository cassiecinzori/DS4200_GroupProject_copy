"""
interactive_composition.py - Interactive Neighborhood Request Pattern Comparison
Creates an Altair visualization allowing users to compare neighborhoods and years
"""

from api311 import Year
import altair as alt
import pandas as pd


def clean_request_type_name(name):
    """Make request type names more readable"""
    replacements = {
        'Missed Trash/Recycling/Yard Waste/Bulk Item': 'Missed Trash/Recycling',
        'Request for Snow Plowing': 'Snow Plowing',
        'Request for Pothole Repair': 'Pothole Repair',
        'Street Light Outages': 'Street Lights',
        'Pothole Repair (Internal)': 'Pothole (Internal)',
        'Poor Conditions of Property': 'Poor Property Conditions',
        'Improper Storage of Trash (Barrels)': 'Improper Trash Storage',
        'Parks Lighting/Electrical Issues': 'Parks Lighting',
    }
    return replacements.get(name, name)


def create_interactive_comparison(year15, year25):
    """
    Create interactive Altair visualization for neighborhood comparison
    """
    print("Creating interactive neighborhood comparison...")

    # Clean data - remove blank neighborhoods
    data15 = year15.data[
        year15.data["neighborhood"].notna() &
        (year15.data["neighborhood"].str.strip() != "")
    ].copy()

    data25 = year25.data[
        year25.data["neighborhood"].notna() &
        (year25.data["neighborhood"].str.strip() != "")
    ].copy()

    # Get all neighborhoods sorted alphabetically
    all_neighborhoods = sorted(
        set(data15["neighborhood"].unique()) | set(data25["neighborhood"].unique())
    )

    # Build dataset - for each neighborhood, get top 10 types combined across both years
    data_list = []

    for hood in all_neighborhoods:
        # Get data for this neighborhood from both years
        hood_15 = data15[data15["neighborhood"] == hood]
        hood_25 = data25[data25["neighborhood"] == hood]

        # Combine and get top 10 most common types for THIS neighborhood
        combined = pd.concat([hood_15["type"], hood_25["type"]])
        top_types = combined.value_counts().head(10).index.tolist()

        # Add 2015 counts for top types
        for req_type in top_types:
            count = (hood_15["type"] == req_type).sum()
            if count > 0:  # Only add if there are requests
                data_list.append({
                    "Neighborhood": hood,
                    "Request Type": clean_request_type_name(req_type),
                    "Count": int(count),
                    "Year": "2015"
                })

        # Add 2025 counts for top types
        for req_type in top_types:
            count = (hood_25["type"] == req_type).sum()
            if count > 0:
                data_list.append({
                    "Neighborhood": hood,
                    "Request Type": clean_request_type_name(req_type),
                    "Count": int(count),
                    "Year": "2025"
                })

    df = pd.DataFrame(data_list)

    # Single neighborhood selection dropdown
    input_dropdown = alt.binding_select(
        options=all_neighborhoods,
        name='Select Neighborhood: '
    )

    neighborhood_param = alt.param(
        name='neighborhood',
        bind=input_dropdown,
        value='Dorchester'
    )

    # Year selection via legend
    year_selection = alt.selection_point(
        fields=['Year'],
        bind='legend'
    )

    # Filter to show only selected neighborhood
    base = alt.Chart(df).transform_filter(
        alt.datum.Neighborhood == neighborhood_param
    )

    # Create the bar chart
    chart = base.mark_bar(size=25).encode(
        y=alt.Y(
            'Request Type:N',
            sort='ascending',
            title='Request Type',
            axis=alt.Axis(labelLimit=300, labelFontSize=12)
        ),
        x=alt.X(
            'Count:Q',
            title='Number of Requests',
            scale=alt.Scale(zero=True)
        ),
        color=alt.Color(
            'Year:N',
            scale=alt.Scale(
                domain=['2015', '2025'],
                range=['#4C72B0', '#DD8452']
            ),
            legend=alt.Legend(
                title='Year (Click to Filter)',
                orient='top-right',
                labelFontSize=12,
                titleFontSize=13
            )
        ),
        yOffset=alt.YOffset('Year:N', scale=alt.Scale(paddingOuter=0.3)),
        opacity=alt.condition(
            year_selection,
            alt.value(0.9),
            alt.value(0.15)
        ),
        tooltip=[
            alt.Tooltip('Neighborhood:N', title='Neighborhood'),
            alt.Tooltip('Request Type:N', title='Request Type'),
            alt.Tooltip('Year:N', title='Year'),
            alt.Tooltip('Count:Q', title='Requests', format=',')
        ]
    ).add_params(
        neighborhood_param,
        year_selection
    ).properties(
        width=700,
        height=600,
        title={
            "text": "Interactive Neighborhood Request Pattern Comparison",
            "subtitle": "Select a neighborhood and click legend to toggle years",
            "fontSize": 18,
            "fontWeight": "bold",
            "subtitleFontSize": 13,
            "subtitleColor": "gray"
        }
    ).configure_axis(
        labelFontSize=12,
        titleFontSize=14,
        grid=True,
        gridOpacity=0.3
    ).configure_view(
        strokeWidth=0
    )

    return chart


def main():
    """Generate and save interactive visualization"""
    print("Loading data...")
    year15 = Year("data/cleaned2015.csv")
    year25 = Year("data/cleaned2025.csv")

    # Don't need to call make_points() for this visualization
    # year15.make_points()
    # year25.make_points()

    print(f"Loaded 2015: {len(year15.data):,} records")
    print(f"Loaded 2025: {len(year25.data):,} records")

    # Create interactive chart
    chart = create_interactive_comparison(year15, year25)

    # Save as HTML
    chart.save('interactive_neighborhood_comparison.html')
    print("\n✅ Saved: interactive_neighborhood_comparison.html")
    print("\nOpen this file in your browser to interact with it!")

    # Also display if running in Jupyter
    return chart


if __name__ == "__main__":
    chart = main()