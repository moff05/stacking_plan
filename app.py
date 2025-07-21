import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from io import BytesIO
from PIL import Image

# -----------------------------------
# Initialize session state & handle reset
# -----------------------------------

DEFAULTS = {
    'start_color': "#FF0000",
    'end_color': "#00FF00",
    'fig_width': 25,
    'fig_height': 14,
    'logo_x': 50,
    'logo_y': 50,
    'logo_size': 150,
    'uploaded_file_data': None, # Add this to store file content
    'uploaded_file_name': None # Add this to store file name
}

if 'reset_triggered' not in st.session_state:
    st.session_state.reset_triggered = False

if st.session_state.reset_triggered:
    for k, v in DEFAULTS.items():
        st.session_state[k] = v
    st.session_state.reset_triggered = False

# -----------------------------------
# UI Start
# -----------------------------------

st.title("Stacking Plan Generator")

# Template download
with open("stacking_plan_template.xlsx", "rb") as f:
    template_data = f.read()

st.download_button(
    label="📥 Download Excel Template",
    data=template_data,
    file_name="stacking_plan_template.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# -----------------------------------
# Sidebar Controls
# -----------------------------------

with st.sidebar:
    st.header("Settings")

    if st.button("🔄 Reset All Settings"):
        st.session_state.reset_triggered = True
        st.rerun()

    st.subheader("Chart Size")
    fig_width = st.slider(
        "Figure Width (inches)",
        min_value=5, max_value=40,
        value=st.session_state.get('fig_width', DEFAULTS['fig_width']),
        step=1, key="fig_width"
    )
    fig_height = st.slider(
        "Figure Height (inches)",
        min_value=5, max_value=25,
        value=st.session_state.get('fig_height', DEFAULTS['fig_height']),
        step=1, key="fig_height"
    )

    st.subheader("Colors")
    start_color = st.color_picker(
        "Start color (earliest year)",
        value=st.session_state.get('start_color', DEFAULTS['start_color']),
        key="start_color"
    )
    end_color = st.color_picker(
        "End color (latest year)",
        value=st.session_state.get('end_color', DEFAULTS['end_color']),
        key="end_color"
    )

    st.subheader("Logo")
    # Handle logo file separately if it needs to persist
    logo_file_uploader = st.file_uploader("Upload logo (PNG/JPG)", type=["png", "jpg", "jpeg"])

    # If a new logo is uploaded, store it
    if logo_file_uploader is not None:
        st.session_state['logo_file_data'] = logo_file_uploader.getvalue()
        st.session_state['logo_file_type'] = logo_file_uploader.type
    
    # Use the stored logo data if available, otherwise None
    logo_file = None
    if 'logo_file_data' in st.session_state and st.session_state['logo_file_data'] is not None:
        logo_file = BytesIO(st.session_state['logo_file_data'])
        logo_file.name = "uploaded_logo." + st.session_state['logo_file_type'].split('/')[-1] # give it a name for PIL

    if logo_file is not None:
        logo_x = st.slider(
            "Logo X position (px from left)", 0, 2000,
            value=st.session_state.get('logo_x', DEFAULTS['logo_x']),
            step=10, key="logo_x"
        )
        logo_y = st.slider(
            "Logo Y position (px from bottom)", 0, 2000,
            value=st.session_state.get('logo_y', DEFAULTS['logo_y']),
            step=10, key="logo_y"
        )
        logo_size = st.slider(
            "Logo max size (pixels)", 50, 500,
            value=st.session_state.get('logo_size', DEFAULTS['logo_size']),
            step=10, key="logo_size"
        )
    else:
        logo_x = st.session_state.get('logo_x', DEFAULTS['logo_x'])
        logo_y = st.session_state.get('logo_y', DEFAULTS['logo_y'])
        logo_size = st.session_state.get('logo_size', DEFAULTS['logo_size'])

# -----------------------------------
# Building name input
# -----------------------------------

building_name = st.text_input(
    "🏢 Enter building name or address for this stacking plan",
    "My Building"
)

# -----------------------------------
# File Upload & Processing
# -----------------------------------

# File uploader widget
new_uploaded_file = st.file_uploader("Upload your Excel file here (.xlsx)")

# If a new file is uploaded, update session state
if new_uploaded_file is not None:
    st.session_state['uploaded_file_data'] = new_uploaded_file.getvalue()
    st.session_state['uploaded_file_name'] = new_uploaded_file.name

# Use the file from session state, if available
if st.session_state.get('uploaded_file_data') is not None:
    # Recreate the BytesIO object from stored data
    uploaded_file_content = BytesIO(st.session_state['uploaded_file_data'])
    uploaded_file_content.name = st.session_state['uploaded_file_name'] # Important for pandas to read it
    uploaded_file = uploaded_file_content
else:
    uploaded_file = None # No file, so no plot

required_columns = ['Floor', 'Suite Number', 'Tenant Name', 'Square Footage', 'Expiration Date']

if uploaded_file is not None:
    try:
        data = pd.read_excel(uploaded_file)
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            st.error(f"❌ Uploaded file is missing required columns: {', '.join(missing_cols)}")
            st.stop()
    except Exception as e:
        st.error(f"❌ Error reading Excel file: {e}")
        st.stop()

    # Matplotlib style
    plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 8})

    data['Expiration Date'] = pd.to_datetime(data['Expiration Date'])
    data['Expiration Year'] = data['Expiration Date'].dt.year
    data = data.sort_values(by=['Floor', 'Suite Number'], ascending=[False, True])

    years = data.loc[~data['Tenant Name'].str.upper().str.contains('VACANT'), 'Expiration Year'].dropna().unique()
    years = np.sort(years) if len(years) > 0 else np.array([2025, 2030])

    cmap = LinearSegmentedColormap.from_list("custom_gradient", [start_color, end_color])
    norm = mcolors.Normalize(vmin=years.min(), vmax=years.max())

    def get_color(row):
        tenant_upper = str(row['Tenant Name']).upper()
        if 'VACANT' in tenant_upper:
            return '#d3d3d3'
        year = row['Expiration Year']
        if pd.isna(year):
            return '#1f77b4'
        return mcolors.to_hex(cmap(norm(year)))

    # Occupancy summary
    year_totals = data.loc[~data['Tenant Name'].str.upper().str.contains('VACANT')].groupby('Expiration Year')['Square Footage'].sum()
    no_expiry_total = data.loc[data['Expiration Year'].isna() & ~data['Tenant Name'].str.upper().str.contains('VACANT'), 'Square Footage'].sum()
    vacant_total = data.loc[data['Tenant Name'].str.upper().str.contains('VACANT'), 'Square Footage'].sum()

    occupancy_summary = [f"{int(year)}: {int(sf):,} SF" for year, sf in year_totals.items()]
    if vacant_total > 0:
        occupancy_summary.append(f"VACANT: {int(vacant_total):,} SF")
    if no_expiry_total > 0:
        occupancy_summary.append(f"No Expiry: {int(no_expiry_total):,} SF")
    occupancy_text = " | ".join(occupancy_summary)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    y_pos = 0
    height = 1
    plot_width = 10

    floors = sorted(data['Floor'].unique(), reverse=True)

    for floor in floors:
        floor_data = data[data['Floor'] == floor]
        floor_sum = floor_data['Square Footage'].sum()
        x_pos = 0

        ax.text(-0.5, y_pos, f"Floor {floor}\n{floor_sum} SF", ha='right', va='center', fontsize=8, fontweight='bold')

        for _, row in floor_data.iterrows():
            suite_sf = row['Square Footage']
            tenant = row['Tenant Name']
            suite = row['Suite Number']
            width = suite_sf / floor_sum * plot_width
            color = get_color(row)
            expiry = row['Expiration Date'].strftime('%Y-%m-%d') if pd.notna(row['Expiration Date']) else 'No Expiry'

            ax.barh(y=y_pos, width=width, height=height, left=x_pos, color=color, edgecolor='black')
            ax.text(x=x_pos + width/2, y=y_pos, s=f"{tenant}\nSuite {suite}\n{suite_sf} SF\n{expiry}", ha='center', va='center', fontsize=6)

            x_pos += width
        y_pos += height

    ax.set_xlabel('Proportional Suite Width (normalized per floor)', fontsize=10)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title(f'Stacking Plan - {building_name}', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(bottom=False)
    plt.tight_layout()

    if logo_file is not None:
        logo = Image.open(logo_file)
        logo.thumbnail((logo_size, logo_size))
        fig.figimage(logo, xo=logo_x, yo=logo_y, alpha=1, zorder=10)

    legend_elements = [mpatches.Patch(facecolor=mcolors.to_hex(cmap(norm(y))), edgecolor='black', label=str(int(y))) for y in years]
    legend_elements.append(mpatches.Patch(facecolor='#d3d3d3', edgecolor='black', label='VACANT'))
    legend_elements.append(mpatches.Patch(facecolor='#1f77b4', edgecolor='black', label='No Expiry Date'))

    ax.text(0.5, -0.1, f"Total SF by Expiration Year: {occupancy_text}", transform=ax.transAxes, ha='center', va='top', fontsize=10, fontweight='bold')
    ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=len(legend_elements), fontsize=8)

    st.pyplot(fig)

    pdf_buf = BytesIO()
    fig.savefig(pdf_buf, format="pdf", bbox_inches="tight")
    pdf_buf.seek(0)
    st.download_button("Download Stacking Plan as PDF", pdf_buf, file_name=f"{building_name}_stacking_plan.pdf", mime="application/pdf")

    png_buf = BytesIO()
    fig.savefig(png_buf, format="png", bbox_inches="tight")
    png_buf.seek(0)
    st.download_button("Download Stacking Plan as PNG", png_buf, file_name=f"{building_name}_stacking_plan.png", mime="application/png")

    st.success("✅ Stacking plan generated!")
