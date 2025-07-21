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
    'logo_file_type': None,    # To store the type of the logo
    'auto_size': True,         # New: Enable auto-sizing by default
    'min_bar_height': 1.2,     # New: Minimum height per floor for readability
    'text_padding_factor': 1.5 # New: Extra space factor for text comfort
}

# Initialize ALL session state variables with their defaults if they don't exist
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# -----------------------------------
# Auto-sizing calculation function
# -----------------------------------
def calculate_optimal_dimensions(data, building_name, auto_size=True, min_bar_height=1.2, text_padding_factor=1.5):
    """
    Calculate optimal figure dimensions based on content.
    
    Args:
        data: DataFrame with the stacking plan data
        building_name: String for the building name
        auto_size: Boolean to enable/disable auto-sizing
        min_bar_height: Minimum height per floor for readability
        text_padding_factor: Factor to add extra space for text comfort
    
    Returns:
        tuple: (optimal_width, optimal_height)
    """
    if not auto_size:
        return st.session_state.fig_width, st.session_state.fig_height
    
    # Calculate number of floors
    num_floors = len(data['Floor'].unique())
    
    # Calculate maximum text length per floor to determine width needs
    max_text_length = 0
    floors = data['Floor'].unique()
    
    for floor in floors:
        floor_data = data[data['Floor'] == floor]
        for _, row in floor_data.iterrows():
            tenant = str(row['Tenant Name'])
            suite = str(row['Suite Number'])
            suite_sf = row['Square Footage']
            expiry = row['Expiration Date'].strftime('%Y-%m-%d') if pd.notna(row['Expiration Date']) else 'No Expiry'
            
            # Calculate text length for both lines
            line1 = f"Suite {suite} | {tenant}"
            line2 = f"{suite_sf} SF | {expiry}"
            
            max_line_length = max(len(line1), len(line2))
            max_text_length = max(max_text_length, max_line_length)
    
    # Calculate width based on content
    # Base width calculation: consider building name length and maximum suite text
    building_name_length = len(building_name) if building_name else 20
    title_width_needed = building_name_length * 0.15  # Rough character-to-inch conversion
    
    # Calculate width needed for suite text (accounting for proportional layout)
    content_width_needed = max_text_length * 0.08  # Character-to-inch for suite text
    
    # Minimum width for readability and proportions
    min_width = 12
    calculated_width = max(min_width, title_width_needed + 8, content_width_needed + 6)
    
    # Apply text padding factor
    optimal_width = calculated_width * text_padding_factor
    
    # Cap maximum width for practical display
    optimal_width = min(optimal_width, 35)
    
    # Calculate height based on number of floors
    base_height_per_floor = max(min_bar_height, 1.0)
    calculated_height = num_floors * base_height_per_floor
    
    # Add extra space for title, legend, and padding
    title_space = 2
    legend_space = 2
    padding_space = 2
    
    optimal_height = calculated_height + title_space + legend_space + padding_space
    
    # Apply text padding factor to height as well
    optimal_height *= text_padding_factor
    
    # Set reasonable bounds
    optimal_height = max(8, min(optimal_height, 25))
    optimal_width = max(10, optimal_width)
    
    return optimal_width, optimal_height

# -----------------------------------
# Reset Settings Function
# -----------------------------------
def reset_settings():
    """Reset all settings to their default values while preserving uploaded files"""
    # Settings to reset (exclude file-related keys)
    settings_to_reset = ['start_color', 'end_color', 'fig_width', 'fig_height', 
                        'logo_x', 'logo_y', 'logo_size', 'building_name', 
                        'auto_size', 'min_bar_height', 'text_padding_factor']
    
    # Reset the actual session state values
    for setting in settings_to_reset:
        st.session_state[setting] = DEFAULTS[setting]
    
    # Also reset the widget keys to force UI update
    widget_keys_to_reset = ['fig_width_slider', 'fig_height_slider', 'start_color_picker', 
                           'end_color_picker', 'logo_x_slider', 'logo_y_slider', 
                           'logo_size_slider', 'building_name_input', 'auto_size_toggle',
                           'min_bar_height_slider', 'text_padding_slider']
    
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
                elif key == 'min_bar_height_slider':
                    st.session_state[key] = DEFAULTS['min_bar_height']
                elif key == 'text_padding_slider':
                    st.session_state[key] = DEFAULTS['text_padding_factor']
            elif key == 'building_name_input':
                st.session_state[key] = DEFAULTS['building_name']
            elif key == 'auto_size_toggle':
                st.session_state[key] = DEFAULTS['auto_size']
    
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

    # Auto-sizing toggle
    st.subheader("🤖 Auto-Sizing")
    auto_size = st.toggle(
        "Enable automatic chart sizing",
        value=st.session_state.auto_size,
        key="auto_size_toggle",
        help="When enabled, the chart dimensions will automatically adjust based on your data content for optimal readability."
    )
    st.session_state.auto_size = auto_size
    
    # Auto-sizing parameters (only show when auto-sizing is enabled)
    if auto_size:
        min_bar_height = st.slider(
            "Minimum floor height",
            min_value=0.8, max_value=2.5, 
            value=st.session_state.min_bar_height,
            step=0.1, key="min_bar_height_slider",
            help="Minimum height per floor for text readability"
        )
        st.session_state.min_bar_height = min_bar_height
        
        text_padding_factor = st.slider(
            "Text comfort factor",
            min_value=1.0, max_value=2.5,
            value=st.session_state.text_padding_factor,
            step=0.1, key="text_padding_slider",
            help="Increases spacing around text for better readability (1.0 = tight, 2.0 = very spacious)"
        )
        st.session_state.text_padding_factor = text_padding_factor

    # Manual chart size sliders (only show when auto-sizing is disabled)
    if not auto_size:
        st.subheader("📏 Manual Chart Size")
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
    else:
        st.info("📐 Chart size is automatically calculated based on your data when auto-sizing is enabled.")

    # Color pickers
    st.subheader("🎨 Colors")
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
    st.subheader("🖼️ Logo")
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

    # Calculate optimal dimensions
    optimal_width, optimal_height = calculate_optimal_dimensions(
        data, 
        st.session_state.building_name,
        st.session_state.auto_size,
        st.session_state.min_bar_height,
        st.session_state.text_padding_factor
    )
    
    # Display current dimensions info
    if st.session_state.auto_size:
        st.info(f"🤖 Auto-calculated chart size: {optimal_width:.1f}\" × {optimal_height:.1f}\" (W×H)")

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

    # Use the calculated optimal dimensions
    fig, ax = plt.subplots(figsize=(optimal_width, optimal_height))

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

            # Two-line format: Suite + Tenant | SF + Expiry
            line1 = f"Suite {suite} | {tenant}"
            line2 = f"{suite_sf} SF | {expiry}"
            
            # Dynamic font size based on chart size and bar width
            font_size = max(4, min(8, optimal_height * 0.4))
            
            ax.text(x=x_pos + width/2, y=y_pos,
                            s=f"{line1}\n{line2}",
                            ha='center', va='center', fontsize=font_size)

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
    # Add expiration years with their square footage in the legend labels
    for year in years:
        color = mcolors.to_hex(cmap(norm(year)))
        year_sf = year_totals.get(year, 0)
        label = f"{int(year)} ({int(year_sf):,} SF)" if year_sf > 0 else str(int(year))
        legend_elements.append(mpatches.Patch(facecolor=color, edgecolor='black', label=label))
    
    # Add VACANT with square footage if any
    if vacant_total > 0:
        legend_elements.append(mpatches.Patch(facecolor='#d3d3d3', edgecolor='black', label=f'VACANT ({int(vacant_total):,} SF)'))
    else:
        legend_elements.append(mpatches.Patch(facecolor='#d3d3d3', edgecolor='black', label='VACANT'))
    
    # Add No Expiry with square footage if any
    if no_expiry_total > 0:
        legend_elements.append(mpatches.Patch(facecolor='#1f77b4', edgecolor='black', label=f'No Expiry ({int(no_expiry_total):,} SF)'))
    else:
        legend_elements.append(mpatches.Patch(facecolor='#1f77b4', edgecolor='black', label='No Expiry'))

    # Remove the separate square footage summary text since it's now in the legend
    ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.08),
                            ncol=len(legend_elements), fontsize=12)

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
