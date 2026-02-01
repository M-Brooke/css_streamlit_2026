import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import base64
import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path


# Set page title
st.set_page_config(page_title="Profile and Spatial Data Explorer", layout="wide")

# Sidebar Menu
st.sidebar.title("Navigation")
menu = st.sidebar.radio(
    "Go to:",
    ["Researcher Profile", "Publications", "Spatial Data Explorer", "Contact"],
)

#Assets - resolve path relative to the current file
BASE_DIR = Path(__file__).parent
    
LOGO_PATH = BASE_DIR / "assets" / "Picture3.png"
PDF_PATH = BASE_DIR / "assets" / "UPD_Spatial_Trends_Report_2024_25_comp.pdf"
COVER_PATH = BASE_DIR / "assets" / "Spatial-Trends-Cover-Poster-2025.jpg"
DATA_PATH = BASE_DIR / "assets" / "sl_du2_grid_w_dens_Gross_RES_Density.csv"

st.logo(str(LOGO_PATH))

# Spatial Data Input

gv_dens_data = pd.read_csv(DATA_PATH)

# Legend
def render_legend(vmin, vmax, sel_year, width=5, height=0.5):
    """
    Renders a horizontal Blue->Red colorbar with vmin/vmax labels.
    """
    # Create a simple blue->red colormap to match your map
    cmap = LinearSegmentedColormap.from_list("blue_red", [(0, 0, 1), (1, 0, 0)])  # blue to red

    fig, ax = plt.subplots(figsize=(width, height))
    fig.subplots_adjust(bottom=0.5)

    # A gradient image for the colorbar
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    ax.imshow(gradient, aspect="auto", cmap=cmap)
    ax.set_axis_off()

    # Add a horizontal colorbar with vmin/vmax tick labels
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="35%", pad=0.25)

    cb = plt.colorbar(
        plt.cm.ScalarMappable(cmap=cmap),
        cax=cax,
        orientation="horizontal"
    )
    cb.set_ticks([0, 1])
    cb.set_ticklabels([f"{vmin:.1f} du/ha", f"{vmax:.1f} du/ha"])
    cb.set_label(f"Gross Res Density {sel_year} (Blue → Low, Red → High)", fontsize=9)

    st.pyplot(fig, clear_figure=True)

# Sections based on menu selection
if menu == "Researcher Profile":
    st.title("Researcher Profile 🏙️")
    st.sidebar.header("Profile Options")
    
    # Create two equal-width columns
    col1, col2 = st.columns(2)
    
    with col1:
        # Collect basic information
        name = "Michael Brooke"
        field = "Urban Planning"
        institution = "City of Cape Town"
        department = "Urban Planning and Design"
    
        # Display basic profile information
        st.write(f"**Name:** {name}")
        st.write(f"**Field of Research:** {field}")
        st.write(f"**Institution:** {institution}")
        st.write(f"**Department:** {department}")
    
        st.image("https://upload.wikimedia.org/wikipedia/commons/9/9c/Logo_of_Cape_Town%2C_South_Africa.svg")   
    
    with col2:
        st.image("https://spacesyntax.com/wp-content/uploads/2018/03/Picture1Presentation.jpg",)

elif menu == "Publications":
    st.title("Publications")
    st.sidebar.header("See our latest published reports")
    
    # Intro
    st.header("Spatial Trends Report")
    st.write("The City’s Urban Planning and Design Department releases a Spatial Trends Report annually. The report provides the latest information about land use development trends/patterns and change in Cape Town over time.")
    
    # Links to Reports  
    # Create two equal-width columns
    col1, col2 = st.columns(2)
    with col1:
        st.image(str(LOGO_PATH), width = 400)
    
    with col2:
        st.subheader("Access Reports Online")
       
        st.link_button(
        "2024-25 Report (PDF)",
        "https://resource.capetown.gov.za/documentcentre/Documents/City%20research%20reports%20and%20review/UPD_Spatial_Trends_Report_2024-25.pdf",
        type="primary")
        
        st.link_button(
        "2023-24 Report (PDF)",
        "https://resource.capetown.gov.za/documentcentre/Documents/City%20research%20reports%20and%20review/UPD_Spatial_Trends_Report_2023-24.pdf",
        type="primary")
        
        st.link_button(
        "2022-23 Report (PDF)",
        "https://resource.capetown.gov.za/documentcentre/Documents/City%20research%20reports%20and%20review/UPD_Spatial_Trends_Report_2022-23.pdf",
        type="primary") 
       
    # Report Sample - PDF Viewer
   
    def show_pdf(path: Path):
        if not path.exists():
            st.error(f"PDF not found: {path}")
            return
    
        pdf_bytes = path.read_bytes()
        base64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
    
        st.markdown(
            f"""
            <iframe
                src="data:application/pdf;base64,{base64_pdf}"
                width="100%"
                height="600">
            </iframe>
            """,
            unsafe_allow_html=True,
        )
    
    st.subheader("Local PDF Viewer")
    st.write("Sample pages from our latest report:")
    show_pdf(str(PDF_PATH))
       
elif menu == "Spatial Data Explorer":
    st.title("Spatial Data Explorer")
    st.header("Gross Res Density")
    st.sidebar.header("Controls")
    
    # Configuration for column names
    YEAR_COLUMNS = {
        "2012": "dens_grs_duha_2012",
        "2015": "dens_grs_duha_2015",
        "2018": "dens_grs_duha_2018",
        "2022": "dens_grs_duha_2022",
    }

    sel_year = st.sidebar.selectbox(
        "Select year",
        list(YEAR_COLUMNS.keys()),
        index=3)
    opacity = st.sidebar.slider("Cell opacity", 0.1, 1.0, 0.6, 0.05)
    
    value_col = YEAR_COLUMNS[sel_year]
    st.write("This indicator shows change in gross residential density over time (2012-2022) as captured in the City’s General Valuation Rolls (formal dwellings only). Data is aggregated by using a square grid comprised of 4ha (200m x 200m) grid cells.")
    st.write("Please select a year from the navigation bar to update the data shown in the map and chart below.")
     # Create two equal-width columns
    col1, col2 = st.columns(2)
    with col1:
        
        st.subheader("Table and Charts")
        # Show first few records
        #st.dataframe(gv_dens_data[[value_col,"lat","lon"]].head(20))
        #st.write(gv_dens_data[["dens_grs_duha_2022","dens_grs_duha_2018","dens_grs_duha_2015","dens_grs_duha_2012"]].describe())
    
        
        # === Compute means per configured year columns ===
        year_order = list(YEAR_COLUMNS.keys())
        cols = [YEAR_COLUMNS[y] for y in year_order]
        
        means = (
            gv_dens_data[cols]
            .apply(pd.to_numeric, errors="coerce")
            .mean(skipna=True)
        )
        
        df_mean = pd.DataFrame({
            "year": year_order,
            "mean_density": [means.get(YEAR_COLUMNS[y], np.nan) for y in year_order],
        })
        
        
        # --- % change calculations ---
        # % change vs previous year (ignores NaNs properly)
        df_mean["pct_change_prev"] = df_mean["mean_density"].pct_change()
        
        # % change vs baseline: first available (non-null) year in the sequence
        first_valid_idx = df_mean["mean_density"].first_valid_index()
        if first_valid_idx is not None:
            base_val = df_mean.loc[first_valid_idx, "mean_density"]
            if pd.notna(base_val) and base_val != 0:
                df_mean["pct_change_base"] = (df_mean["mean_density"] / base_val) - 1.0
            else:
                df_mean["pct_change_base"] = np.nan
        else:
            df_mean["pct_change_base"] = np.nan
        
        # Optional: a helper column to mark the selected year (useful for plotting/logic)
        df_mean["highlight"] = np.where(df_mean["year"] == sel_year, "Selected", "Other")
        
        # --- Pretty formatting for display ---
        def _format_pct(x):
            return "" if pd.isna(x) else f"{x*100:.1f}%"
        
        display_df = df_mean.copy()
        display_df["mean_density"] = display_df["mean_density"].round(2)
        display_df["pct_change_prev"] = display_df["pct_change_prev"].apply(_format_pct)
        display_df["pct_change_base"] = display_df["pct_change_base"].apply(_format_pct)
        
        # Reorder/limit columns for display
        display_cols = ["year", "mean_density", "pct_change_prev", "pct_change_base"]
        
        # --- Styling: highlight the selected year row or text ---
        def highlight_selected_row(row):
            if row["year"] == sel_year:
                # Subtle background highlight; tweak colors to taste
                return ["background-color: #ffe8e8; color: #7a0a0a; font-weight: 600;"] * len(row)
            return [""] * len(row)
        
        styler = (
            display_df[display_cols]
            .style.apply(highlight_selected_row, axis=1)
            .set_properties(**{"text-align": "left"})
        )
        
        # Optional number alignment: set a monospaced font for numeric cols
        styler = styler.set_table_styles([
            {"selector": "th", "props": [("text-align", "left")]},
            {"selector": "td", "props": [("vertical-align", "middle")]},
        ])
        
        st.subheader("Mean Density Summary")
        st.dataframe(styler, use_container_width=True)

        
        
        #st.write(df_mean)
        

        fig = px.line(
            df_mean,
            x="year",
            y="mean_density",
            markers=True,
            #color="highlight",
            #color_discrete_map=color_map,
            labels={"mean_density": "Mean density (du/ha)", "year": "Year"},
            title="Mean Gross Residential Density by Year",
            #template="plotly_white",
            )

        #Enlarge the selected marker
        sel_idx = df_mean.index[df_mean["year"] == sel_year][0]
        fig.add_scatter(
            x=[df_mean.loc[sel_idx, "year"]],
            y=[df_mean.loc[sel_idx, "mean_density"]],
            mode="markers",
            marker=dict(size=13, line=dict(width=2, color="rgba(120,0,0,1)")),
            hoverinfo="skip",
            showlegend=False
            )

        # Add a horizontal reference line at selected year's mean
        sel_mean = float(df_mean.loc[df_mean["year"] == sel_year, "mean_density"].iloc[0])
        fig.add_hline(
            y=sel_mean,
            line_dash="dash",
            annotation_text=f"{sel_year} mean: {sel_mean:.2f} du/ha",
            annotation_position="top left",
            opacity=0.6
            )
            
            
        st.plotly_chart(fig, use_container_width=True)

    
    with col2:
        
        st.subheader(f"Map showing average Gross Res Density for {sel_year} per 4ha grid cell")
        
        # Ensure numeric type and compute robust min/max (ignore NaNs)
        dens = pd.to_numeric(gv_dens_data[value_col], errors="coerce")
        dmin = float(np.nanpercentile(dens, 2)) if np.isfinite(dens).any() else 0.0
        dmax = float(np.nanpercentile(dens, 98)) if np.isfinite(dens).any() else 1.0
        if not np.isfinite(dmin) or not np.isfinite(dmax) or dmin == dmax:
            dmin, dmax = 0.0, 1.0

        # Normalize 0..1
        t = (dens - dmin) / (dmax - dmin + 1e-12)
        t = t.clip(lower=0, upper=1)

        # Blue (low) → Red (high), fixed green channel for contrast
        # Handle NaNs separately as grey
        r = (255 * t).round().astype("Int64")
        g = pd.Series(50, index=dens.index, dtype="Int64")
        b = (255 * (1 - t)).round().astype("Int64")

        # Grey for NaNs
        is_nan = dens.isna()
        r[is_nan] = 180
        g[is_nan] = 180
        b[is_nan] = 180

        # Pack color as [r,g,b, alpha]
        color_rgba = pd.concat([r, g, b, pd.Series(index=dens.index).round().astype("Int64")], axis=1)
        color_rgba.columns = ["r", "g", "b", "a"]

        # Build a working DataFrame for the layer
        df_map = gv_dens_data.copy()
        df_map["_elev"] = dens.fillna(0)  # elevation uses 0 for NaN
        df_map["_color"] = color_rgba.values.tolist()
        
        # Simple pydeck map
        st.pydeck_chart(
            pdk.Deck(
                map_style=None,
                    initial_view_state=pdk.ViewState(
                    latitude=float(gv_dens_data["lat"].median()),
                    longitude=float(gv_dens_data["lon"].median()),
                    zoom=9,
                    pitch=50
                ),
                layers=[
                    pdk.Layer(
                        "GridCellLayer",
                        data=df_map,
                        get_position=["lon", "lat"],
                        #get_fill_color=value_col,
                        cell_size=200,
                        get_fill_color="_color",
                        get_elevation="_elev",
                        elevation_scale=10,
                        elevation_range=[0, 1000],
                        pickable=True,
                        extruded=True,
                        opacity=opacity,
                    )
                ],
                tooltip={"text": f"Gross Res Density {sel_year}: {{{value_col}}} du/ha"},
            )
        )                  

        render_legend(dmin, dmax, sel_year)

elif menu == "Contact":
    # Add a contact section
    st.header("Contact Information")
    email = "upd.data@capetown.gov.za"
    st.write(f"For further information or if you would like to be in touch with us email: {email}.")



