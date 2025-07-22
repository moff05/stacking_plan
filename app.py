import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import matplotlib.patches as mpatches
from io import BytesIO
from PIL import Image
from datetime import datetime

# -----------------------------------
# Helper function to determine contrasting text color
# -----------------------------------
def get_contrasting_text_color(hex_color):
    """
    Determines if 'black' or 'white' text is more readable on a given hex background color.
    Uses the W3C luminance algorithm for perceived brightness.
    """
    if not isinstance(hex_color, str) or not hex_color.startswith('#'):
        # Default for non-hex or invalid input
        return 'black'

    hex_color = hex_color.lstrip('#')
    # Handle shorthand hex codes (e.g., #FFF)
    if len(hex_color) == 3:
        hex_color = ''.join([c*2 for c in hex_color])

    try:
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    except ValueError:
        # Fallback for invalid hex codes
        return 'black'

    # Calculate luminance (perceived brightness)
    # Formula: L = 0.2126 * R + 0.7152 * G + 0.0722 * B (for sRGB)
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    
    # Use a threshold to decide between black and white
    return 'black' if luminance > 0.5 else 'white'

# -----------------------------------
# Initialize session state for persistence of ALL changeable values
# -----------------------------------

# Get current year
CURRENT_YEAR = datetime.now().year

# Define initial default colors for each year offset
YEAR_COLOR_DEFAULTS = {
    0: "#FF0000",    # Current year = Red
    1: "#FFA500",    # +1 = Orange
    2: "#FFFF00",    # +2 = Yellow
    3: "#00FF00",    # +3 = Green
    4: "#ADD8E6",    # +4 = Light Blue
    5: "#800080",    # +5 = Purple
    6: "#000080",    # +6 = Dark Blue
    7: "#A52A2A",    # +7 = Brown
    8: "#808080"     # +8 or more = Gray
}

# Define initial default values for settings
DEFAULTS = {
    'fig_width': 16,
    'fig_height': 9,
    'logo_size': 200, # Still adjustable, but position fixed
    'building_name': "My Building",
    'vacant_color': '#d3d3d3',  # Light gray for vacant
    'no_expiry_color': '#1f77b4',  # Blue for no expiry
    'excel_file_content': None,
    'excel_file_name': None,
    'logo_file_content': None,
    'logo_file_type': None,
    **{f'year_{i}_color': color for i, color in YEAR_COLOR_DEFAULTS.items()},
    # Initialize text colors based on contrast for toggles
    **{f'year_{i}_text_color': get_contrasting_text_color(color) for i, color in YEAR_COLOR_DEFAULTS.items()},
    'vacant_text_color': get_contrasting_text_color('#d3d3d3'),
    'no_expiry_text_color': get_contrasting_text_color('#1f77b4'),
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
    # Settings to reset
    settings_to_reset = (['fig_width', 'fig_height', 'logo_size', 'building_name',
                          'vacant_color', 'no_expiry_color'] +
                         [f'year_{i}_color' for i in range(9)] +
                         [f'year_{i}_text_color' for i in range(9)] +
                         ['vacant_text_color', 'no_expiry_text_color'])

    # Reset the actual session state values
    for setting in settings_to_reset:
        if setting in DEFAULTS: # Ensure default exists for the setting
            st.session_state[setting] = DEFAULTS[setting]
        else: # Handle text colors that might not have a direct default in DEFAULTS
            if 'year_' in setting and '_text_color' in setting:
                year_num = int(setting.split('_')[1])
                st.session_state[setting] = get_contrasting_text_color(YEAR_COLOR_DEFAULTS[year_num])
            elif setting == 'vacant_text_color':
                st.session_state[setting] = get_contrasting_text_color(DEFAULTS['vacant_color'])
            elif setting == 'no_expiry_text_color':
                st.session_state[setting] = get_contrasting_text_color(DEFAULTS['no_expiry_color'])


    # Also reset the widget keys to force UI update
    widget_keys_to_reset = (['fig_width_slider', 'fig_height_slider', 'logo_size_slider',
                             'building_name_input', 'vacant_color_picker',
                             'no_expiry_color_picker',
                             'vacant_text_color_toggle', 'no_expiry_text_color_toggle'] +
                            [f'year_{i}_color_picker' for i in range(9)] +
                            [f'year_{i}_text_color_toggle' for i in range(9)])

    for key in widget_keys_to_reset:
        if key in st.session_state:
            if 'color_picker' in key:
                if key == 'vacant_color_picker':
                    st.session_state[key] = DEFAULTS['vacant_color']
                elif key == 'no_expiry_color_picker':
                    st.session_state[key] = DEFAULTS['no_expiry_color']
                else:
                    year_num = int(key.split('_')[1])
                    st.session_state[key] = YEAR_COLOR_DEFAULTS[year_num]
            elif 'slider' in key:
                if key == 'fig_width_slider':
                    st.session_state[key] = DEFAULTS['fig_width']
                elif key == 'fig_height_slider':
                    st.session_state[key] = DEFAULTS['fig_height']
                elif key == 'logo_size_slider':
                    st.session_state[key] = DEFAULTS['logo_size']
            elif key == 'building_name_input':
                st.session_state[key] = DEFAULTS['building_name']
            elif 'text_color_toggle' in key:
                if key == 'vacant_text_color_toggle':
                    st.session_state[key] = get_contrasting_text_color(DEFAULTS['vacant_color'])
                elif key == 'no_expiry_text_color_toggle':
                    st.session_state[key] = get_contrasting_text_color(DEFAULTS['no_expiry_color'])
                else: # For year_X_text_color_toggle
                    year_num = int(key.split('_')[1])
                    st.session_state[key] = get_contrasting_text_color(YEAR_COLOR_DEFAULTS[year_num])

    # Force a rerun to update the UI with reset values
    st.rerun()

def get_year_offset_color(expiration_year):
    """Get color based on year offset from current year"""
    if pd.isna(expiration_year):
        return st.session_state.no_expiry_color
    
    year_offset = int(expiration_year) - CURRENT_YEAR
    
    if year_offset < 0:
        year_offset = 0
    elif year_offset > 8:
        year_offset = 8
    
    return st.session_state[f'year_{year_offset}_color']

def get_year_offset_text_color(expiration_year):
    """Get text color based on year offset from current year's background color"""
    if pd.isna(expiration_year):
        return st.session_state.no_expiry_text_color
    
    year_offset = int(expiration_year) - CURRENT_YEAR
    
    if year_offset < 0:
        year_offset = 0
    elif year_offset > 8:
        year_offset = 8
    
    return st.session_state[f'year_{year_offset}_text_color']


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

    # Year-based color pickers with text color toggles
    st.subheader("Year Colors & Text")
    st.write(f"**Base Year: {CURRENT_YEAR}**")
    
    year_labels = {
        0: f"Current ({CURRENT_YEAR})",
        1: f"+1 Year ({CURRENT_YEAR + 1})",
        2: f"+2 Years ({CURRENT_YEAR + 2})",
        3: f"+3 Years ({CURRENT_YEAR + 3})",
        4: f"+4 Years ({CURRENT_YEAR + 4})",
        5: f"+5 Years ({CURRENT_YEAR + 5})",
        6: f"+6 Years ({CURRENT_YEAR + 6})",
        7: f"+7 Years ({CURRENT_YEAR + 7})",
        8: f"+8+ Years ({CURRENT_YEAR + 8}+)"
    }
    
    for i in range(9):
        col1, col2 = st.columns([0.6, 0.4])
        with col1:
            color = st.color_picker(
                year_labels[i],
                value=st.session_state[f'year_{i}_color'],
                key=f'year_{i}_color_picker'
            )
            st.session_state[f'year_{i}_color'] = color
        with col2:
            # Re-add text color radio for each year
            text_color = st.radio(
                f"Text for {year_labels[i]}",
                options=['black', 'white'],
                index=0 if st.session_state[f'year_{i}_text_color'] == 'black' else 1,
                key=f'year_{i}_text_color_toggle',
                horizontal=True
            )
            st.session_state[f'year_{i}_text_color'] = text_color


    # Special category colors with text color toggles
    st.subheader("Special Categories & Text")
    
    col1, col2 = st.columns([0.6, 0.4])
    with col1:
        vacant_color = st.color_picker(
            "Vacant Units Background",
            value=st.session_state.vacant_color,
            key='vacant_color_picker'
        )
        st.session_state.vacant_color = vacant_color
    with col2:
        # Re-add text color radio for vacant
        vacant_text_color = st.radio(
            "Vacant Text",
            options=['black', 'white'],
            index=0 if st.session_state.vacant_text_color == 'black' else 1,
            key='vacant_text_color_toggle',
            horizontal=True
        )
        st.session_state.vacant_text_color = vacant_text_color

    col1, col2 = st.columns([0.6, 0.4])
    with col1:
        no_expiry_color = st.color_picker(
            "No Expiry Background",
            value=st.session_state.no_expiry_color,
            key='no_expiry_color_picker'
        )
        st.session_state.no_expiry_color = no_expiry_color
    with col2:
        # Re-add text color radio for no expiry
        no_expiry_text_color = st.radio(
            "No Expiry Text",
            options=['black', 'white'],
            index=0 if st.session_state.no_expiry_text_color == 'black' else 1,
            key='no_expiry_text_color_toggle',
            horizontal=True
        )
        st.session_state.no_expiry_text_color = no_expiry_text_color


    # Logo upload + controls
    st.subheader("Logo")
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

    # Display logo size setting only if there's a logo in session state
    if logo_file_to_display is not None:
        st.write("**Logo Size**") # Only size is adjustable now

        logo_size = st.slider(
            "Logo max size (pixels)", 50, 800,
            value=st.session_state.logo_size, step=10, key="logo_size_slider",
            help="Maximum width or height of the logo."
        )
        st.session_state.logo_size = logo_size

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

# Determine which file content to use for plotting
excel_file_to_process = None
if st.session_state.get('excel_file_content') is not None:
    excel_file_to_process = BytesIO(st.session_state['excel_file_content'])
    excel_file_to_process.name = st.session_state['excel_file_name']

required_columns = ['Floor', 'Suite Number', 'Tenant Name', 'Square Footage', 'Expiration Date']

if excel_file_to_process is not None:
    try:
        data = pd.read_excel(excel_file_to_process)
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            st.error(f"❌ Uploaded file is missing required columns: {', '.join(missing_cols)}")
            st.session_state['excel_file_content'] = None
            st.session_state['excel_file_name'] = None
            st.stop()
    except Exception as e:
        st.error(f"❌ Error reading Excel file: {e}")
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
        years = np.array([CURRENT_YEAR, CURRENT_YEAR + 1])
    else:
        years = np.sort(years_data.unique())
        if len(years) == 1:
            years = np.array([years[0], years[0] + 1])

    def get_color(row):
        tenant_upper = str(row['Tenant Name']).upper()
        if 'VACANT' in tenant_upper:
            return st.session_state.vacant_color
        return get_year_offset_color(row['Expiration Year'])

    def get_text_color_for_unit(row):
        tenant_upper = str(row['Tenant Name']).upper()
        if 'VACANT' in tenant_upper:
            return st.session_state.vacant_text_color
        return get_year_offset_text_color(row['Expiration Year'])


    year_totals = data.loc[~data['Tenant Name'].str.upper().str.contains('VACANT')].groupby('Expiration Year')['Square Footage'].sum()
    no_expiry_total = data.loc[data['Expiration Year'].isna() & ~data['Tenant Name'].str.upper().str.contains('VACANT'), 'Square Footage'].sum()
    vacant_total = data.loc[data['Tenant Name'].str.upper().str.contains('VACANT'), 'Square Footage'].sum()

    # Calculate Total Occupied SF and Total Available SF
    total_occupied_sf = data.loc[~data['Tenant Name'].str.upper().str.contains('VACANT'), 'Square Footage'].sum()
    total_available_sf = data['Square Footage'].sum()

    occupancy_percentage = (total_occupied_sf / total_available_sf) * 100 if total_available_sf > 0 else 0
    occupancy_percent_text = f"{occupancy_percentage:.1f}% ({int(total_occupied_sf):,} / {int(total_available_sf):,} SF)"

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
                        ha='right', va='center', fontsize=8, fontweight='bold',
                        color='black') # Hardcoded to black

        for i, row in floor_data.iterrows():
            suite_sf = row['Square Footage']
            tenant = row['Tenant Name']
            suite = row['Suite Number']

            width = suite_sf / floor_sum * plot_width if floor_sum > 0 else 0
            color = get_color(row)
            text_color = get_text_color_for_unit(row) # Get specific text color for this unit

            ax.barh(y=y_pos, width=width, height=height, left=x_pos,
                            color=color, edgecolor='black')

            expiry = row['Expiration Date'].strftime('%Y-%m-%d') if pd.notna(row['Expiration Date']) else 'No Expiry'

            line1 = f"Suite {suite}"
            line2_text = f"{tenant}"
            line3 = f"{suite_sf:,} SF | {expiry}"

            ax.text(x=x_pos + width/2, y=y_pos - 0.2,
                            s=line1, ha='center', va='center', fontsize=6,
                            color=text_color)

            ax.text(x=x_pos + width/2, y=y_pos,
                            s=line2_text, ha='center', va='center', fontsize=6,
                            fontweight='bold', color=text_color)

            ax.text(x=x_pos + width/2, y=y_pos + 0.2,
                            s=line3, ha='center', va='center', fontsize=6,
                            color=text_color)

            x_pos += width

        y_pos += height

    ax.set_xlabel('Proportional Suite Width (normalized per floor)', color='black') # Hardcoded to black
    ax.set_yticks([])
    ax.set_xticks([])
    
    # Dynamic title based on building_name
    display_title = st.session_state.building_name
    if st.session_state.building_name == DEFAULTS['building_name']:
        display_title = f'Stacking Plan - {st.session_state.building_name}'

    ax.set_title(display_title, fontsize=14,
                  fontweight='bold', color='black') # Hardcoded to black
    ax.invert_yaxis()

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.tick_params(bottom=False)

    # Adjust layout to make space for the legend on the right
    plt.tight_layout(rect=[0, 0, 0.82, 1])


    # Add logo permanently in the bottom-left corner
    if logo_file_to_display is not None:
        logo = Image.open(logo_file_to_display)

        logo_size_px = int(st.session_state.logo_size)
        aspect_ratio = logo.size[0] / logo.size[1]
        if logo.size[0] > logo.size[1]:
            new_width = logo_size_px
            new_height = int(logo_size_px / aspect_ratio)
        else:
            new_height = logo_size_px
            new_width = int(logo_size_px * aspect_ratio)

        logo = logo.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Fixed bottom-left position with a small pixel margin
        margin_px = 10 # Small margin from the edges
        logo_x_px = margin_px
        logo_y_px = margin_px

        # Add logo to figure
        fig.figimage(logo, xo=int(logo_x_px), yo=int(logo_y_px), alpha=1, zorder=10)


    # Add Occupancy Percentage Text
    ax.text(0.5, -0.05, f"Occupancy: {occupancy_percent_text}",
            transform=ax.transAxes, ha='center', va='top', fontsize=12,
            fontweight='bold', color='black') # Hardcoded to black

    # Create legend with year-based colors
    legend_elements = []
    
    # Group years by offset from current year for legend
    year_groups = {}
    for year in years:
        offset = int(year) - CURRENT_YEAR
        if offset < 0:
            offset = 0
        elif offset > 8:
            offset = 8
        
        if offset not in year_groups:
            year_groups[offset] = []
        year_groups[offset].append(int(year))
    
    # Add legend entries for each year group
    for offset in sorted(year_groups.keys()):
        years_in_group = sorted(year_groups[offset])
        color = st.session_state[f'year_{offset}_color']
        total_sf = sum(year_totals.get(year, 0) for year in years_in_group)
        
        if offset == 0:
            label = f"{CURRENT_YEAR}"
        elif offset == 8:
            # For 8+, make sure it truly represents 8 and greater years in the data
            actual_years_8_plus = [y for y in years if y >= CURRENT_YEAR + 8]
            if actual_years_8_plus:
                 label = f"{CURRENT_YEAR + 8}+"
            else:
                 label = f"Years > {CURRENT_YEAR + 7}" # Fallback if no data beyond 7
        else:
            years_str = ", ".join(str(y) for y in years_in_group)
            label = years_str
        
        if total_sf > 0:
            label += f" ({int(total_sf):,} SF)"
        
        legend_elements.append(mpatches.Patch(facecolor=color, edgecolor='black', label=label))

    # Add VACANT and No Expiry with adjustable colors
    if vacant_total > 0:
        legend_elements.append(mpatches.Patch(facecolor=st.session_state.vacant_color, edgecolor='black', label=f'VACANT ({int(vacant_total):,} SF)'))
    else:
        legend_elements.append(mpatches.Patch(facecolor=st.session_state.vacant_color, edgecolor='black', label='VACANT'))

    if no_expiry_total > 0:
        legend_elements.append(mpatches.Patch(facecolor=st.session_state.no_expiry_color, edgecolor='black', label=f'No Expiry ({int(no_expiry_total):,} SF)'))
    else:
        legend_elements.append(mpatches.Patch(facecolor=st.session_state.no_expiry_color, edgecolor='black', label='No Expiry'))

    # Modified legend placement: to the right of the chart
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5),
              fontsize=10,
              facecolor='none', edgecolor='none',
              labelcolor='black') # Hardcoded to black

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
