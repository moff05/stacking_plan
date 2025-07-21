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
    'logo_x_percent': 98,  # Changed to percentage-based positioning, default top right
    'logo_y_percent': 98,  # Changed to percentage-based positioning, default top right
    'logo_size': 200,      # Increased default size
    'building_name': "My Building",
    'excel_file_content': None, # To store the binary content of the Excel file
    'excel_file_name': None,    # To store the name of the Excel file
    'logo_file_content': None, # To store the binary content of the logo
    'logo_file_type': None      # To store the type of the logo
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
                          'logo_x_percent', 'logo_y_percent', 'logo_size', 'building_name']

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
                    st.session_state[key] = DEFAULTS['logo_x_percent']
                elif key == 'logo_y_slider':
                    st.session_state[key] = DEFAULTS['logo_y_percent']
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

    # Calculate Total Occupied SF and Total Available SF
    total_occupied_sf = data.loc[~data['Tenant Name'].str.upper().str.contains('VACANT'), 'Square Footage'].sum()
    total_available_sf = data['Square Footage'].sum() # Sum of all square footage in the building

    occupancy_percentage = (total_occupied_sf / total_available_sf) * 100 if total_available_sf > 0 else 0

    # Format the occupancy text
    occupancy_percent_text = f"{occupancy_percentage:.1f}% ({int(total_occupied_sf):,} / {int(total_available_sf):,} SF)"

    occupancy_summary = []
    for year, total_sf in year_totals.items():
        occupancy_summary.append(f"{int(year)}: {int(total_sf):,} SF")
    if no_expiry_total > 0:
        occupancy_summary.append(f"No Expiry: {int(no_expiry_total):,} SF")
    if vacant_total > 0:
        occupancy_summary.append(f"VACANT: {int(vacant_total):,} SF")

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

            # Handle division by zero for floors with 0 SF total (though unlikely with valid data)
            width = suite_sf / floor_sum * plot_width if floor_sum > 0 else 0

            color = get_color(row)

            ax.barh(y=y_pos, width=width, height=height, left=x_pos,
                                color=color, edgecolor='black')

            expiry = row['Expiration Date'].strftime('%Y-%m-%d') if pd.notna(row['Expiration Date']) else 'No Expiry'

            # Three-line format: Suite \n Tenant \n SF + Expiry
            line1 = f"Suite {suite}"
            line2_text = f"{tenant}" # Tenant name without bolding escape codes
            line3 = f"{suite_sf:,} SF | {expiry}" # Added comma formatting

            # Add the first and third lines
            ax.text(x=x_pos + width/2, y=y_pos - 0.2, # Adjust y position slightly for 3 lines
                    s=line1,
                    ha='center', va='center', fontsize=6)

            # Add the second line (tenant name) with bold fontweight
            ax.text(x=x_pos + width/2, y=y_pos,
                    s=line2_text,
                    ha='center', va='center', fontsize=6, **{'fontweight': 'bold'}) # Apply bold here

            # Add the third line
            ax.text(x=x_pos + width/2, y=y_pos + 0.2, # Adjust y position slightly for 3 lines
                    s=line3,
                    ha='center', va='center', fontsize=6)

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

    # Add logo with percentage-based positioning
    # Add logo with percentage-based positioning relative to the chart area
    if logo_file_to_display is not None:
        logo = Image.open(logo_file_to_display)

        # Resize logo to exact size instead of using thumbnail
        logo_size_px = int(st.session_state.logo_size)
        # Calculate aspect ratio to maintain proportions
        aspect_ratio = logo.size[0] / logo.size[1]
        if logo.size[0] > logo.size[1]:  # Wider than tall
            new_width = logo_size_px
            new_height = int(logo_size_px / aspect_ratio)
        else:  # Taller than wide or square
            new_height = logo_size_px
            new_width = int(logo_size_px * aspect_ratio)

        logo = logo.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Get figure dimensions in pixels
        # Get the axes position in figure coordinates
        bbox = ax.get_position()  # Returns Bbox object with x0, y0, x1, y1 in figure coordinates (0-1)
        fig_width_px = fig.get_figwidth() * fig.dpi
        fig_height_px = fig.get_figheight() * fig.dpi

        # Calculate logo position based on percentage of entire figure
        # For corner positioning, align logo edges with figure edges
        if st.session_state.logo_x_percent <= 5:  # Left edge
            logo_x_px = 0
        elif st.session_state.logo_x_percent >= 95:  # Right edge
            logo_x_px = int(fig_width_px - logo.size[0])
        else:  # Anywhere in between, center the logo on the percentage point
            logo_x_px = int((st.session_state.logo_x_percent / 100) * fig_width_px - logo.size[0] / 2)
        # Calculate actual chart area in pixels
        chart_left_px = bbox.x0 * fig_width_px
        chart_right_px = bbox.x1 * fig_width_px
        chart_bottom_px = bbox.y0 * fig_height_px
        chart_top_px = bbox.y1 * fig_height_px

        if st.session_state.logo_y_percent <= 5:  # Bottom edge
            logo_y_px = 0
        elif st.session_state.logo_y_percent >= 95:  # Top edge
            logo_y_px = int(fig_height_px - logo.size[1])
        else:  # Anywhere in between, center the logo on the percentage point
            logo_y_px = int((st.session_state.logo_y_percent / 100) * fig_height_px - logo.size[1] / 2)
        chart_width_px = chart_right_px - chart_left_px
        chart_height_px = chart_top_px - chart_bottom_px

        # Ensure logo stays within figure bounds
        logo_x_px = max(0, min(logo_x_px, int(fig_width_px - logo.size[0])))
        logo_y_px = max(0, min(logo_y_px, int(fig_height_px - logo.size[1])))
        # Calculate logo position based on percentage within the chart area
        if st.session_state.logo_x_percent <= 5:  # Left edge - align left edge of logo with left edge of chart
            logo_x_px = int(chart_left_px)
        elif st.session_state.logo_x_percent >= 95:  # Right edge - align right edge of logo with right edge of chart
            logo_x_px = int(chart_right_px - logo.size[0])
        else:  # Anywhere in between, position relative to chart area
            logo_x_px = int(chart_left_px + (st.session_state.logo_x_percent / 100) * chart_width_px - logo.size[0] / 2)
        
        if st.session_state.logo_y_percent <= 5:  # Bottom edge - align bottom edge of logo with bottom edge of chart
            logo_y_px = int(chart_bottom_px)
        elif st.session_state.logo_y_percent >= 95:  # Top edge - align top edge of logo with top edge of chart
            logo_y_px = int(chart_top_px - logo.size[1])
        else:  # Anywhere in between, position relative to chart area
            logo_y_px = int(chart_bottom_px + (st.session_state.logo_y_percent / 100) * chart_height_px - logo.size[1] / 2)
        
        # Ensure logo stays within chart bounds
        logo_x_px = max(int(chart_left_px), min(logo_x_px, int(chart_right_px - logo.size[0])))
        logo_y_px = max(int(chart_bottom_px), min(logo_y_px, int(chart_top_px - logo.size[1])))

        fig.figimage(logo, xo=logo_x_px, yo=logo_y_px, alpha=1, zorder=10)

    # --- Add Occupancy Percentage Text ---
    # Position it relative to the axes for consistent placement above the legend
    # bbox_to_anchor controls the position in axes coordinates (0,0 is bottom-left, 1,1 is top-right)
    # The first value (0.5) centers it horizontally.
    # The second value (-0.15) places it below the main plot area but above the legend.
    # Adjust this value as needed based on your fig_height and desired spacing.
    ax.text(0.5, -0.05, f"Occupancy: {occupancy_percent_text}",
            transform=ax.transAxes, ha='center', va='top', fontsize=12, fontweight='bold')
    # --- End Occupancy Percentage Text ---

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

    # Adjusted bbox_to_anchor for the legend to make space for the new text
    # The y-coordinate might need slight tweaking depending on overall chart size and desired spacing.
    ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.15),
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
