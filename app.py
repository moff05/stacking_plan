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
    'logo_x_percent': 98,
    'logo_y_percent': 98,
    'logo_size': 200,
    'building_name': "My Building",
    'font_color': 'black',  # Default to black font
    'vacant_color': '#d3d3d3',  # Light gray for vacant
    'no_expiry_color': '#1f77b4',  # Blue for no expiry
    'excel_file_content': None,
    'excel_file_name': None,
    'logo_file_content': None,
    'logo_file_type': None,
    **{f'year_{i}_color': color for i, color in YEAR_COLOR_DEFAULTS.items()}
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
    settings_to_reset = (['fig_width', 'fig_height', 'logo_x_percent', 'logo_y_percent', 
                        'logo_size', 'building_name', 'font_color', 'vacant_color', 'no_expiry_color'] + 
                       [f'year_{i}_color' for i in range(9)])

    # Reset the actual session state values
    for setting in settings_to_reset:
        st.session_state[setting] = DEFAULTS[setting]

    # Also reset the widget keys to force UI update
    widget_keys_to_reset = (['fig_width_slider', 'fig_height_slider', 'logo_x_slider', 
                           'logo_y_slider', 'logo_size_slider', 'building_name_input', 
                           'font_color_toggle', 'vacant_color_picker', 'no_expiry_color_picker'] + 
                          [f'year_{i}_color_picker' for i in range(9)])

    for key in widget_keys_to_reset:
        if key in st.session_state:
            if 'color_picker' in key:
                if key == 'vacant_color_picker':
                    st.session_state[key] = DEFAULTS['vacant_color']
                elif key == 'no_expiry_color_picker':
                    st.session_state[key] = DEFAULTS['no_expiry_color']
                else:
                    # Extract year number from key
                    year_num = int(key.split('_')[1])
                    st.session_state[key] = YEAR_COLOR_DEFAULTS[year_num]
            elif 'slider' in key:
                if key == 'fig_width_slider':
                    st.session_state[key] = DEFAULTS['fig_width']
                elif key == 'fig_height_slider':
                    st.session_state[key] = DEFAULTS['fig_height']
                elif key == 'logo_x_slider':
                    st.session_state[key] = DEFAULTS['logo_x_percent']
                elif key == 'logo_y_slider':
                    st.session_state[key] = DEFAULTS['logo_y_percent']
                elif key == 'logo_size_slider':
                    st.session_state[key] = DEFAULTS['logo_size']
            elif key == 'building_name_input':
                st.session_state[key] = DEFAULTS['building_name']
            elif key == 'font_color_toggle':
                st.session_state[key] = DEFAULTS['font_color']

    # Force a rerun to update the UI with reset values
    st.rerun()

def get_year_offset_color(expiration_year):
    """Get color based on year offset from current year"""
    if pd.isna(expiration_year):
        return st.session_state.no_expiry_color  # Use adjustable no expiry color
    
    year_offset = int(expiration_year) - CURRENT_YEAR
    
    if year_offset < 0:
        year_offset = 0  # Past years use current year color
    elif year_offset > 8:
        year_offset = 8  # 8+ years use the 8+ color
    
    return st.session_state[f'year_{year_offset}_color']

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

    # Year-based color pickers
    st.subheader("Year Colors")
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
        color = st.color_picker(
            year_labels[i],
            value=st.session_state[f'year_{i}_color'],
            key=f'year_{i}_color_picker'
        )
        st.session_state[f'year_{i}_color'] = color

    # Special category colors
    st.subheader("Special Categories")
    
    vacant_color = st.color_picker(
        "Vacant Units",
        value=st.session_state.vacant_color,
        key='vacant_color_picker'
    )
    st.session_state.vacant_color = vacant_color

    no_expiry_color = st.color_picker(
        "No Expiry Date",
        value=st.session_state.no_expiry_color,
        key='no_expiry_color_picker'
    )
    st.session_state.no_expiry_color = no_expiry_color

    # Font color toggle
    st.subheader("Text Settings")
    font_color = st.selectbox(
        "Font Color",
        options=['black', 'white'],
        index=0 if st.session_state.font_color == 'black' else 1,
        key='font_color_toggle'
    )
    st.session_state.font_color = font_color

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

    # Display logo settings only if there's a logo in session state
    if logo_file_to_display is not None:
        st.write("**Logo Position & Size**")

        # Position presets for easier adjustment
        col1, col2 = st.columns(2)
        with col1:
            if st.button("↖️ Top Left", use_container_width=True):
                st.session_state.logo_x_percent = 2
                st.session_state.logo_y_percent = 98
                st.rerun()
            if st.button("↙️ Bottom Left", use_container_width=True):
                st.session_state.logo_x_percent = 2
                st.session_state.logo_y_percent = 2
                st.rerun()

        with col2:
            if st.button("↗️ Top Right", use_container_width=True):
                st.session_state.logo_x_percent = 98
                st.session_state.logo_y_percent = 98
                st.rerun()
            if st.button("↘️ Bottom Right", use_container_width=True):
                st.session_state.logo_x_percent = 98
                st.session_state.logo_y_percent = 2
                st.rerun()

        if st.button("🎯 Center", use_container_width=True):
            st.session_state.logo_x_percent = 50
            st.session_state.logo_y_percent = 50
            st.rerun()

        st.write("**Fine-tune Position:**")

        logo_x_percent = st.slider(
            "Horizontal position (%)", 0, 100,
            value=st.session_state.logo_x_percent, step=1, key="logo_x_slider",
            help="0% = far left, 100% = far right"
        )
        st.session_state.logo_x_percent = logo_x_percent

        logo_y_percent = st.slider(
            "Vertical position (%)", 0, 100,
            value=st.session_state.logo_y_percent, step=1, key="logo_y_slider",
            help="0% = bottom, 100% = top"
        )
        st.session_state.logo_y_percent = logo_y_percent

        logo_size = st.slider(
            "Logo max size (pixels)", 50, 800,
            value=st.session_state.logo_size, step=10, key="logo_size_slider",
            help="Maximum width or height of the logo"
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
            return st.session_state.vacant_color  # Use adjustable vacant color
        return get_year_offset_color(row['Expiration Year'])

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
                color=st.session_state.font_color)

        for i, row in floor_data.iterrows():
            suite_sf = row['Square Footage']
            tenant = row['Tenant Name']
            suite = row['Suite Number']

            width = suite_sf / floor_sum * plot_width if floor_sum > 0 else 0
            color = get_color(row)

            ax.barh(y=y_pos, width=width, height=height, left=x_pos,
                    color=color, edgecolor='black')

            expiry = row['Expiration Date'].strftime('%Y-%m-%d') if pd.notna(row['Expiration Date']) else 'No Expiry'

            line1 = f"Suite {suite}"
            line2_text = f"{tenant}"
            line3 = f"{suite_sf:,} SF | {expiry}"

            ax.text(x=x_pos + width/2, y=y_pos - 0.2,
                    s=line1, ha='center', va='center', fontsize=6, 
                    color=st.session_state.font_color)

            ax.text(x=x_pos + width/2, y=y_pos,
                    s=line2_text, ha='center', va='center', fontsize=6, 
                    fontweight='bold', color=st.session_state.font_color)

            ax.text(x=x_pos + width/2, y=y_pos + 0.2,
                    s=line3, ha='center', va='center', fontsize=6,
                    color=st.session_state.font_color)

            x_pos += width

        y_pos += height

    ax.set_xlabel('Proportional Suite Width (normalized per floor)', color=st.session_state.font_color)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title(f'Stacking Plan - {st.session_state.building_name}', fontsize=14, 
                 fontweight='bold', color=st.session_state.font_color)
    ax.invert_yaxis()

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.tick_params(bottom=False)

    plt.tight_layout()

    # Add logo with corrected percentage-based positioning
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

        # Get figure dimensions in pixels
        fig_width_px = fig.get_figwidth() * fig.dpi
        fig_height_px = fig.get_figheight() * fig.dpi

        # Calculate logo position based on percentages
        # X position (0% = left edge, 100% = right edge)
        if st.session_state.logo_x_percent <= 5:
            # Left edge
            logo_x_px = 10  # Small margin from edge
        elif st.session_state.logo_x_percent >= 95:
            # Right edge
            logo_x_px = fig_width_px - logo.size[0] - 10  # Small margin from edge
        else:
            # Percentage-based positioning
            logo_x_px = (st.session_state.logo_x_percent / 100) * fig_width_px - logo.size[0] / 2

        # Y position (0% = bottom, 100% = top)
        # Note: figimage uses bottom-left origin, so we need to flip Y
        if st.session_state.logo_y_percent <= 5:
            # Bottom edge
            logo_y_px = 10  # Small margin from edge
        elif st.session_state.logo_y_percent >= 95:
            # Top edge
            logo_y_px = fig_height_px - logo.size[1] - 10  # Small margin from edge
        else:
            # Percentage-based positioning
            logo_y_px = (st.session_state.logo_y_percent / 100) * fig_height_px - logo.size[1] / 2
        
        # Ensure logo stays within figure bounds
        logo_x_px = max(0, min(logo_x_px, fig_width_px - logo.size[0]))
        logo_y_px = max(0, min(logo_y_px, fig_height_px - logo.size[1]))

        # Add logo to figure
        fig.figimage(logo, xo=int(logo_x_px), yo=int(logo_y_px), alpha=1, zorder=10)

    # Add Occupancy Percentage Text
    ax.text(0.5, -0.05, f"Occupancy: {occupancy_percent_text}",
            transform=ax.transAxes, ha='center', va='top', fontsize=12, 
            fontweight='bold', color=st.session_state.font_color)

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
            years_str = ", ".join(str(y) for y in years_in_group)
            label = f"{years_str}+"
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

    ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.15),
              ncol=len(legend_elements), fontsize=12, 
              facecolor='none', edgecolor='none', 
              labelcolor=st.session_state.font_color)

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
