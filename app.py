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
# Initialize session state for persistence of ALL changeable values
# -----------------------------------

# Define initial default values for settings
# These will be applied ONLY if the session_state variable doesn't exist yet.
# Once a user changes a setting, its value in session_state persists.
DEFAULTS = {
    'start_color': "#FF0000",
    'end_color': "#00FF00",
    'fig_width': 25,
    'fig_height': 14,
    'logo_x': 50,
    'logo_y': 50,
    'logo_size': 150,
    'building_name': "My Building",
    'excel_file_content': None, # To store the binary content of the Excel file
    'excel_file_name': None,   # To store the name of the Excel file
    'logo_file_content': None, # To store the binary content of the logo
    'logo_file_type': None     # To store the type of the logo
}

# Initialize ALL session state variables with their defaults if they don't exist
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# -----------------------------------
# Reset Settings Function
# -----------------------------------
def reset_settings():
    """Reset all settings to their default values while preserving uploaded files"""
    # Settings to reset (exclude file-related keys)
    settings_to_reset = ['start_color', 'end_color', 'fig_width', 'fig_height', 
                        'logo_x', 'logo_y', 'logo_size', 'building_name']
    
    # Reset the actual session state values
    for setting in settings_to_reset:
        st.session_state[setting] = DEFAULTS[setting]
    
    # Also reset the widget keys to force UI update
    widget_keys_to_reset = ['fig_width_slider', 'fig_height_slider', 'start_color_picker', 
                           'end_color_picker', 'logo_x_slider', 'logo_y_slider', 
                           'logo_size_slider', 'building_name_input']
    
    for key in widget_keys_to_reset:
        if key in st.session_state:
            if 'color' in key:
                # Reset color picker keys
                if key == 'start_color_picker':
                    st.session_state[key] = DEFAULTS['start_color']
                elif key == 'end_color_picker':
                    st.session_state[key] = DEFAULTS['end_color']
            elif 'slider' in key:
                # Reset slider keys
                if key == 'fig_width_slider':
                    st.session_state[key] = DEFAULTS['fig_width']
                elif key == 'fig_height_slider':
                    st.session_state[key] = DEFAULTS['fig_height']
                elif key == 'logo_x_slider':
                    st.session_state[key] = DEFAULTS['logo_x']
                elif key == 'logo_y_slider':
                    st.session_state[key] = DEFAULTS['logo_y']
                elif key == 'logo_size_slider':
                    st.session_state[key] = DEFAULTS['logo_size']
            elif key == 'building_name_input':
                st.session_state[key] = DEFAULTS['building_name']
    
    # Force a rerun to update the UI with reset values
    st.rerun()

# -----------------------------------
# UI Start
# -----------------------------------

st.title("Stacking Plan Generator")

# Template download (from existing file)
with open("stacking_plan_template.xlsx", "rb") as f:
    template_data = f.read()

st.download_button(
    label="📥 Download Excel Template",
    data=template_data,
    file_name="stacking_plan_template.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# Move all controls into sidebar
with st.sidebar:
    st.header("Settings")
    
    # Add Reset Settings button at the top of the sidebar
    if st.button("🔄 Reset All Settings", type="secondary", use_container_width=True):
        reset_settings()

    # Chart size sliders
    st.subheader("Chart Size")
    # Use the widget keys directly and sync with session state
    fig_width = st.slider(
        "Figure Width (inches)",
        min_value=5, max_value=40,
        value=st.session_state.fig_width,
        step=1, key="fig_width_slider"
    )
    st.session_state.fig_width = fig_width
    
    fig_height = st.slider(
        "Figure Height (inches)",
        min_value=5, max_value=25,
        value=st.session_state.fig_height,
        step=1, key="fig_height_slider"
    )
    st.session_state.fig_height = fig_height

    # Color pickers
    st.subheader("Colors")
    start_color = st.color_picker(
        "Start color (earliest year)",
        value=st.session_state.start_color,
        key="start_color_picker"
    )
    st.session_state.start_color = start_color
    
    end_color = st.color_picker(
        "End color (latest year)",
        value=st.session_state.end_color,
        key="end_color_picker"
    )
    st.session_state.end_color = end_color

    # Logo upload + controls
    st.subheader("Logo")
    # File uploader gets its own key. Its output is handled to persist content.
    new_logo_file_uploader = st.file_uploader("Upload logo (PNG/JPG)", type=["png", "jpg", "jpeg"], key="logo_uploader")

    # If a new logo is uploaded, store its content in session state
    if new_logo_file_uploader is not None:
        st.session_state['logo_file_content'] = new_logo_file_uploader.getvalue()
        st.session_state['logo_file_type'] = new_logo_file_uploader.type
    
    # Use the logo content from session state if available
    logo_file_to_display = None
    if st.session_state.get('logo_file_content') is not None:
        logo_file_to_display = BytesIO(st.session_state['logo_file_content'])
        logo_extension = st.session_state.get('logo_file_type', 'image/png').split('/')[-1]
        logo_file_to_display.name = f"uploaded_logo.{logo_extension}"

    # Display logo settings only if there's a logo in session state (meaning a file has been uploaded)
    if logo_file_to_display is not None:
        logo_x = st.slider(
            "Logo X position (pixels from left)", 0, 2000,
            value=st.session_state.logo_x, step=10, key="logo_x_slider"
        )
        st.session_state.logo_x = logo_x
        
        logo_y = st.slider(
            "Logo Y position (pixels from bottom)", 0, 2000,
            value=st.session_state.logo_y, step=10, key="logo_y_slider"
        )
        st.session_state.logo_y = logo_y
        
        logo_size = st.slider(
            "Logo max size (pixels)", 50, 500,
            value=st.session_state.logo_size, step=10, key="logo_size_slider"
        )
        st.session_state.logo_size = logo_size
    # else:
    #     # If no logo is uploaded/persisted, these values will remain at their defaults from session_state initialization
    #     pass

# Building name input stays in main UI for better visibility
building_name = st.text_input(
    "🏢 Enter building name or address for this stacking plan",
    value=st.session_state.building_name,
    key="building_name_input"
)
st.session_state.building_name = building_name

# File upload for stacking data
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
    excel_file_to_process.name = st.session_state['excel_file_name'] # Important for pandas to read it

required_columns = ['Floor', 'Suite Number', 'Tenant Name', 'Square Footage', 'Expiration Date']

if excel_file_to_process is not None:
    try:
        data = pd.read_excel(excel_file_to_process)
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            st.error(f"❌ Uploaded file is missing required columns: {', '.join(missing_cols)}")
            # Clear the invalid file from session state so it doesn't keep trying to process it
            st.session_state['excel_file_content'] = None
            st.session_state['excel_file_name'] = None
            st.stop()
    except Exception as e:
        st.error(f"❌ Error reading Excel file: {e}")
        # Clear the invalid file from session state
        st.session_state['excel_file_content'] = None
        st.session_state['excel_file_name'] = None
        st.stop()

    # Matplotlib style
    plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 8})

    data['Expiration Date'] = pd.to_datetime(data['Expiration Date'])
    data['Expiration Year'] = data['Expiration Date'].dt.year

    data = data.sort_values(by=['Floor', 'Suite Number'], ascending=[False, True])

    years_data = data.loc[~data['Tenant Name'].str.upper().str.contains('VACANT'), 'Expiration Year'].dropna()
    if years_data.empty:
        years = np.array([2025, 2030]) # Default years if no valid expiration dates
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
        color = cmap(norm(year))
        return mcolors.to_hex(color)

    year_totals = data.loc[~data['Tenant Name'].str.upper().str.contains('VACANT')].groupby('Expiration Year')['Square Footage'].sum()
    no_expiry_total = data.loc[data['Expiration Year'].isna() & ~data['Tenant Name'].str.upper().str.contains('VACANT'), 'Square Footage'].sum()
    vacant_total = data.loc[data['Tenant Name'].str.upper().str.contains('VACANT'), 'Square Footage'].sum()

    occupancy_summary = []
    for year, total_sf in year_totals.items():
        occupancy_summary.append(f"{int(year)}: {int(total_sf):,} SF")
    if no_expiry_total > 0:
        occupancy_summary.append(f"No Expiry: {int(no_expiry_total):,} SF")
    if vacant_total > 0:
        occupancy_summary.append(f"VACANT: {int(vacant_total):,} SF")
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

        ax.text(-0.5, y_pos, f"Floor {floor}\n{floor_sum} SF",
                        ha='right', va='center', fontsize=8, fontweight='bold')

        for i, row in floor_data.iterrows():
            suite_sf = row['Square Footage']
            tenant = row['Tenant Name']
            suite = row['Suite Number']
            width = suite_sf / floor_sum * plot_width
            color = get_color(row)

            ax.barh(y=y_pos, width=width, height=height, left=x_pos,
                            color=color, edgecolor='black')

            expiry = row['Expiration Date'].strftime('%Y-%m-%d') if pd.notna(row['Expiration Date']) else 'No Expiry'

            # Two-line format: Suite + Tenant (bold) | SF + Expiry
            line1 = f"Suite {suite} | {tenant}"
            line2 = f"{suite_sf} SF | {expiry}"
            
            ax.text(x=x_pos + width/2, y=y_pos,
                            s=f"{line1}\n{line2}",
                            ha='center', va='center', fontsize=6, weight='bold')

            x_pos += width

        y_pos += height

    ax.set_xlabel('Proportional Suite Width (normalized per floor)')
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title(f'Stacking Plan - {st.session_state.building_name}', fontsize=14, fontweight='bold')
    ax.invert_yaxis()

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.tick_params(bottom=False)

    plt.tight_layout()

    if logo_file_to_display is not None:
        logo = Image.open(logo_file_to_display)
        logo.thumbnail((int(st.session_state.logo_size), int(st.session_state.logo_size)))
        fig.figimage(logo, xo=int(st.session_state.logo_x), yo=int(st.session_state.logo_y), alpha=1, zorder=10)

    legend_elements = []
    for year in years:
        color = mcolors.to_hex(cmap(norm(year)))
        legend_elements.append(mpatches.Patch(facecolor=color, edgecolor='black', label=str(int(year))))
    legend_elements.append(mpatches.Patch(facecolor='#d3d3d3', edgecolor='black', label='VACANT'))
    legend_elements.append(mpatches.Patch(facecolor='#1f77b4', edgecolor='black', label='No Expiry Date'))

    ax.text(0.5, -0.1, f"Total SF by Expiration Year: {occupancy_text}",
                            transform=ax.transAxes,
                            ha='center', va='top', fontsize=10, fontweight='bold')

    ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.2),
                            ncol=len(legend_elements), fontsize=8)

    st.pyplot(fig)

    pdf_buf = BytesIO()
    fig.savefig(pdf_buf, format="pdf", bbox_inches="tight")
    pdf_buf.seek(0)

    st.download_button(
        label="Download Stacking Plan as PDF",
        data=pdf_buf,
        file_name=f"{st.session_state.building_name}_stacking_plan.pdf",
        mime="application/pdf"
    )

    png_buf = BytesIO()
    fig.savefig(png_buf, format="png", bbox_inches="tight")
    png_buf.seek(0)

    st.download_button(
        label="Download Stacking Plan as PNG",
        data=png_buf,
        file_name=f"{st.session_state.building_name}_stacking_plan.png",
        mime="application/png"
    )

    st.success("✅ Stacking plan generated!")
else:
    st.info("⬆️ Please upload an Excel file to generate the stacking plan.")
