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

# Define ALL default settings for UI elements that can be reset
DEFAULTS = {
    'start_color': "#FF0000",
    'end_color': "#00FF00",
    'fig_width': 25,
    'fig_height': 14,
    'logo_x': 50,
    'logo_y': 50,
    'logo_size': 150
}

# Initialize session state variables for UI settings if they don't exist
# This ensures that when the app first runs, the defaults are applied
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Initialize file content session states if they don't exist
# These should NOT be part of the DEFAULTS for resetting because
# we want the uploaded file to persist even after a settings reset.
if 'excel_file_content' not in st.session_state:
    st.session_state.excel_file_content = None
if 'excel_file_name' not in st.session_state:
    st.session_state.excel_file_name = None
if 'logo_file_content' not in st.session_state:
    st.session_state.logo_file_content = None
if 'logo_file_type' not in st.session_state:
    st.session_state.logo_file_type = None

# --- Reset Logic ---
# This block specifically handles the "Reset All Settings" button click.
# It only clears the UI-related settings, keeping the file data intact.
if 'reset_triggered' not in st.session_state:
    st.session_state.reset_triggered = False

if st.session_state.reset_triggered:
    for k, v in DEFAULTS.items():
        st.session_state[k] = v # Reset only the UI settings
    st.session_state.reset_triggered = False # Turn off the flag
    st.rerun() # Immediately rerun to apply the reset UI settings

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

    # The reset button and its effect
    if st.button("🔄 Reset All Settings"):
        st.session_state.reset_triggered = True
        # The st.rerun() is handled in the top-level reset logic after setting the flag.

    st.subheader("Chart Size")
    # Pass the value from session_state directly. When session_state[k] is updated
    # due to reset_triggered, the widget will pick up the new value on rerun.
    fig_width = st.slider(
        "Figure Width (inches)",
        min_value=5, max_value=40,
        value=st.session_state.fig_width, # Directly use session_state value
        step=1, key="fig_width_slider" # Unique key for the widget
    )
    # Update session_state ONLY if the slider is interacted with by the user
    # This ensures that the slider's current value is always what's in session_state
    st.session_state.fig_width = fig_width

    fig_height = st.slider(
        "Figure Height (inches)",
        min_value=5, max_value=25,
        value=st.session_state.fig_height, # Directly use session_state value
        step=1, key="fig_height_slider" # Unique key for the widget
    )
    st.session_state.fig_height = fig_height

    st.subheader("Colors")
    start_color = st.color_picker(
        "Start color (earliest year)",
        value=st.session_state.start_color, # Directly use session_state value
        key="start_color_picker" # Unique key for the widget
    )
    st.session_state.start_color = start_color

    end_color = st.color_picker(
        "End color (latest year)",
        value=st.session_state.end_color, # Directly use session_state value
        key="end_color_picker" # Unique key for the widget
    )
    st.session_state.end_color = end_color

    st.subheader("Logo")
    # Logo upload widget
    new_logo_file_uploader = st.file_uploader("Upload logo (PNG/JPG)", type=["png", "jpg", "jpeg"], key="logo_uploader")

    # If a new logo is uploaded, store its content in session state
    if new_logo_file_uploader is not None:
        st.session_state['logo_file_content'] = new_logo_file_uploader.getvalue()
        st.session_state['logo_file_type'] = new_logo_file_uploader.type
    
    # Use the logo content from session state if available
    logo_file_to_display = None
    if st.session_state.get('logo_file_content') is not None:
        logo_file_to_display = BytesIO(st.session_state['logo_file_content'])
        # Give it a name for PIL. Use a placeholder if original type isn't available.
        logo_extension = st.session_state.get('logo_file_type', 'image/png').split('/')[-1]
        logo_file_to_display.name = f"uploaded_logo.{logo_extension}"

    # Display logo settings only if there's a logo in session state or a new one is uploaded
    if logo_file_to_display is not None:
        logo_x = st.slider(
            "Logo X position (px from left)", 0, 2000,
            value=st.session_state.logo_x,
            step=10, key="logo_x_slider"
        )
        st.session_state.logo_x = logo_x

        logo_y = st.slider(
            "Logo Y position (px from bottom)", 0, 2000,
            value=st.session_state.logo_y,
            step=10, key="logo_y_slider"
        )
        st.session_state.logo_y = logo_y

        logo_size = st.slider(
            "Logo max size (pixels)", 50, 500,
            value=st.session_state.logo_size,
            step=10, key="logo_size_slider"
        )
        st.session_state.logo_size = logo_size
    # No else block needed here as default values are already in session_state from global initialization

# -----------------------------------
# Building name input
# -----------------------------------

building_name = st.text_input(
    "🏢 Enter building name or address for this stacking plan",
    value=st.session_state.get('building_name', "My Building"), # Persist building name
    key="building_name_input"
)
st.session_state.building_name = building_name
# Add 'building_name': "My Building" to DEFAULTS if you want it to reset

# -----------------------------------
# File Upload & Processing
# -----------------------------------

# Excel file uploader widget
new_excel_file_uploader = st.file_uploader("Upload your Excel file here (.xlsx)", key="excel_uploader")

# If a new Excel file is uploaded, store its content in session state
if new_excel_file_uploader is not None:
    st.session_state['excel_file_content'] = new_excel_file_uploader.getvalue()
    st.session_state['excel_file_name'] = new_excel_file_uploader.name

# Determine which file content to use for plotting:
# Prefer content from session state if available (persisted file).
excel_file_to_process = None
if st.session_state.get('excel_file_content') is not None:
    excel_file_to_process = BytesIO(st.session_state['excel_file_content'])
    excel_file_to_process.name = st.session_state['excel_file_name'] # Important for pandas

required_columns = ['Floor', 'Suite Number', 'Tenant Name', 'Square Footage', 'Expiration Date']

# Only proceed with plotting if an Excel file is available (either newly uploaded or from session state)
if excel_file_to_process is not None:
    try:
        data = pd.read_excel(excel_file_to_process)
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            st.error(f"❌ Uploaded file is missing required columns: {', '.join(missing_cols)}")
            # Clear the session state file if it's invalid, so it doesn't keep trying to process it
            st.session_state['excel_file_content'] = None
            st.session_state['excel_file_name'] = None
            st.stop()
    except Exception as e:
        st.error(f"❌ Error reading Excel file: {e}")
        # Clear the session state file if it's invalid
        st.session_state['excel_file_content'] = None
        st.session_state['excel_file_name'] = None
        st.stop()

    # Matplotlib style
    plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 8})

    data['Expiration Date'] = pd.to_datetime(data['Expiration Date'])
    data['Expiration Year'] = data['Expiration Date'].dt.year
    data = data.sort_values(by=['Floor', 'Suite Number'], ascending=[False, True])

    # Ensure years array is not empty for cmap normalization
    years_data = data.loc[~data['Tenant Name'].str.upper().str.contains('VACANT'), 'Expiration Year'].dropna()
    if years_data.empty:
        years = np.array([pd.Timestamp.now().year, pd.Timestamp.now().year + 5]) # Use current year + 5 as sensible defaults
    else:
        years = np.sort(years_data.unique())
        if len(years) == 1: # Handle case with only one unique year
            years = np.array([years[0], years[0] + 1])


    cmap = LinearSegmentedColormap.from_list("custom_gradient", [st.session_state.start_color, st.session_state.end_color])
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

    fig, ax = plt.subplots(figsize=(st.session_state.fig_width, st.session_state.fig_height))
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

    if logo_file_to_display is not None:
        logo = Image.open(logo_file_to_display)
        # Ensure logo_size is an integer, as thumbnail expects integers
        logo.thumbnail((int(st.session_state.logo_size), int(st.session_state.logo_size)))
        fig.figimage(logo, xo=int(st.session_state.logo_x), yo=int(st.session_state.logo_y), alpha=1, zorder=10)

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
else:
    st.info("⬆️ Please upload an Excel file to generate the stacking plan.")
