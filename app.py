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

# Define required columns for the Excel file
REQUIRED_COLUMNS = ['Floor', 'Suite Number', 'Tenant Name', 'Square Footage', 'Expiration Date']

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
                    st.session_state[key] = DEULTS['fig_height']
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
# Plotting Function
# -----------------------------------
def generate_stacking_plan_plot(data, building_name, start_color, end_color,
                                fig_width, fig_height, logo_file_to_display,
                                logo_x, logo_y, logo_size,
                                occupancy_percent_text, year_totals, vacant_total, no_expiry_total):
    
    plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 8})

    data['Expiration Date'] = pd.to_datetime(data['Expiration Date'])
    data['Expiration Year'] = data['Expiration Date'].dt.year
    data = data.sort_values(by=['Floor', 'Suite Number'], ascending=[False, True])

    years_data = data.loc[~data['Tenant Name'].str.upper().str.contains('VACANT'), 'Expiration Year'].dropna()
    if years_data.empty:
        years = np.array([pd.Timestamp.now().year, pd.Timestamp.now().year + 5]) # Use current year + 5 as sensible defaults
    else:
        years = np.sort(years_data.unique())
        if len(years) == 1:
            years = np.array([years[0], years[0] + 1])

    cmap = LinearSegmentedColormap.from_list("custom_gradient", [start_color, end_color])
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

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    y_pos = 0
    height = 1
    plot_width = 10

    floors = sorted(data['Floor'].unique(), reverse=True)

    for floor in floors:
        floor_data = data[data['Floor'] == floor]
        floor_sum = floor_data['Square Footage'].sum()
        x_pos = 0

        ax.text(-0.5, y_pos, f"Floor {floor}\n{floor_sum:,} SF",
                                ha='right', va='center', fontsize=8, fontweight='bold')

        for i, row in floor_data.iterrows():
            suite_sf = row['Square Footage']
            tenant = row['Tenant Name']
            suite = row['Suite Number']
            
            width = suite_sf / floor_sum * plot_width if floor_sum > 0 else 0
            
            color = get_color(row)

            ax.barh(y=y_pos, width=width, height=height, left=x_pos,
                                color=color, edgecolor='black')

            expiry = row['Expiration Date'].strftime('%Y-%m-%d') if pd.notna(row['Expiration Date']) else 'No Expiry'

            line1 = f"Suite {suite} | {tenant}"
            line2 = f"{suite_sf:,} SF | {expiry}" # Added comma formatting
            
            ax.text(x=x_pos + width/2, y=y_pos,
                                s=f"{line1}\n{line2}",
                                ha='center', va='center', fontsize=6)

            x_pos += width

        y_pos += height

    ax.set_xlabel('Proportional Suite Width (normalized per floor)')
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
        # Convert logo_size to int for thumbnail
        logo.thumbnail((int(logo_size), int(logo_size)))
        fig.figimage(logo, xo=int(logo_x), yo=int(logo_y), alpha=1, zorder=10)

    # Occupancy Percentage Text
    ax.text(0.5, -0.05, f"Occupancy: {occupancy_percent_text}",
            transform=ax.transAxes, ha='center', va='top', fontsize=12, fontweight='bold')

    legend_elements = []
    for year in years:
        color = mcolors.to_hex(cmap(norm(year)))
        year_sf = year_totals.get(year, 0)
        label = f"{int(year)} ({int(year_sf):,} SF)" if year_sf > 0 else str(int(year))
        legend_elements.append(mpatches.Patch(facecolor=color, edgecolor='black', label=label))
    
    if vacant_total > 0:
        legend_elements.append(mpatches.Patch(facecolor='#d3d3d3', edgecolor='black', label=f'VACANT ({int(vacant_total):,} SF)'))
    else:
        legend_elements.append(mpatches.Patch(facecolor='#d3d3d3', edgecolor='black', label='VACANT'))
    
    if no_expiry_total > 0:
        legend_elements.append(mpatches.Patch(facecolor='#1f77b4', edgecolor='black', label=f'No Expiry ({int(no_expiry_total):,} SF)'))
    else:
        legend_elements.append(mpatches.Patch(facecolor='#1f77b4', edgecolor='black', label='No Expiry'))

    ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.15),
                                ncol=len(legend_elements), fontsize=12)
    
    return fig

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
st.info("Download an Excel template to easily format your data for the stacking plan.")

# Move all controls into sidebar
with st.sidebar:
    st.header("Settings")
    
    # Add Reset Settings button at the top of the sidebar
    if st.button("🔄 Reset All Settings", type="secondary", use_container_width=True):
        st.info("Click to reset all chart and logo settings to their default values.")
        reset_settings()

    # Chart size sliders
    st.subheader("Chart Size")
    fig_width = st.slider(
        "Figure Width (inches)",
        min_value=5, max_value=40,
        value=st.session_state.fig_width,
        step=1, key="fig_width_slider"
    )
    st.markdown("💡 **Figure Width**: Adjusts the overall width of the generated stacking plan image.")
    
    fig_height = st.slider(
        "Figure Height (inches)",
        min_value=5, max_value=25,
        value=st.session_state.fig_height,
        step=1, key="fig_height_slider"
    )
    st.markdown("💡 **Figure Height**: Adjusts the overall height of the generated stacking plan image.")

    # Color pickers
    st.subheader("Colors")
    start_color = st.color_picker(
        "Start color (earliest year)",
        value=st.session_state.start_color,
        key="start_color_picker"
    )
    st.markdown("💡 **Start Color**: Select the color for the earliest lease expiration year. The gradient will transition from this color.")
    
    end_color = st.color_picker(
        "End color (latest year)",
        value=st.session_state.end_color,
        key="end_color_picker"
    )
    st.markdown("💡 **End Color**: Select the color for the latest lease expiration year. The gradient will transition to this color.")

    # Logo upload + controls
    st.subheader("Logo")
    new_logo_file_uploader = st.file_uploader(
        "Upload logo (PNG/JPG)",
        type=["png", "jpg", "jpeg"],
        key="logo_uploader"
    )
    st.info("Upload your company logo to be displayed on the stacking plan. Recommended formats: PNG, JPG.")

    if new_logo_file_uploader is not None:
        # Only update session state if a new file is indeed uploaded or content changed
        if (st.session_state.get('logo_file_content') != new_logo_file_uploader.getvalue() or
            st.session_state.get('logo_file_type') != new_logo_file_uploader.type):
            st.session_state['logo_file_content'] = new_logo_file_uploader.getvalue()
            st.session_state['logo_file_type'] = new_logo_file_uploader.type
            st.success("✅ Logo uploaded!") # Visual feedback for logo upload
    
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
        st.markdown("💡 **Logo X Position**: Adjusts the horizontal position of the logo on the chart. 0 is the left edge of the plot.")
        
        logo_y = st.slider(
            "Logo Y position (pixels from bottom)", 0, 2000,
            value=st.session_state.logo_y, step=10, key="logo_y_slider"
        )
        st.markdown("💡 **Logo Y Position**: Adjusts the vertical position of the logo on the chart. 0 is the bottom edge of the plot.")
        
        logo_size = st.slider(
            "Logo max size (pixels)", 50, 500,
            value=st.session_state.logo_size, step=10, key="logo_size_slider"
        )
        st.markdown("💡 **Logo Size**: Sets the maximum width/height for the uploaded logo. The logo will scale down if larger.")

# Building name input stays in main UI for better visibility
building_name = st.text_input(
    "🏢 Enter building name or address for this stacking plan",
    value=st.session_state.building_name,
    key="building_name_input"
)
st.markdown("💡 **Building Name**: This text will appear as the main title of your stacking plan chart.")
st.session_state.building_name = building_name

# File upload for stacking data
new_excel_file_uploader = st.file_uploader(
    "Upload your Excel file here (.xlsx)",
    key="excel_uploader"
)
st.info(f"Upload an Excel file containing your building's unit data. **Required columns**: `{', '.join(REQUIRED_COLUMNS)}`.")
st.markdown("If you don't have a file, you can download the **Excel Template** above for proper formatting.")


# If a new Excel file is uploaded, store its content in session state
if new_excel_file_uploader is not None:
    # Only update session state if a new file is indeed uploaded or content changed
    if (st.session_state.get('excel_file_content') != new_excel_file_uploader.getvalue() or
        st.session_state.get('excel_file_name') != new_excel_file_uploader.name):
        st.session_state['excel_file_content'] = new_excel_file_uploader.getvalue()
        st.session_state['excel_file_name'] = new_excel_file_uploader.name
        st.success(f"✅ File '{new_excel_file_uploader.name}' uploaded successfully!")
elif st.session_state.get('excel_file_content') is None:
    pass # Initial state, message already shown by st.info and st.markdown above


# Determine which file content to use for plotting:
# Prefer content from session state if available (persisted file).
excel_file_to_process = None
if st.session_state.get('excel_file_content') is not None:
    excel_file_to_process = BytesIO(st.session_state['excel_file_content'])
    excel_file_to_process.name = st.session_state['excel_file_name'] # Important for pandas to read it

if excel_file_to_process is not None:
    with st.spinner("📊 Generating stacking plan..."): # Add a spinner
        try:
            data = pd.read_excel(excel_file_to_process)
            
            # Validate required columns
            missing_cols = [col for col in REQUIRED_COLUMNS if col not in data.columns]
            if missing_cols:
                st.error(f"❌ Uploaded file is missing required columns: {', '.join(missing_cols)}. Please check the template.")
                st.session_state['excel_file_content'] = None
                st.session_state['excel_file_name'] = None
                st.stop()

            # Validate numeric columns
            for col in ['Square Footage', 'Floor']:
                if not pd.api.types.is_numeric_dtype(data[col]):
                    st.error(f"❌ Column '{col}' must contain numeric values. Please correct your Excel file.")
                    st.session_state['excel_file_content'] = None
                    st.session_state['excel_file_name'] = None
                    st.stop()
            
            # Check for empty data after initial processing
            if data.empty:
                st.warning("⚠️ The uploaded Excel file contains no data or no valid rows after initial processing. Please ensure your file is not empty.")
                st.session_state['excel_file_content'] = None
                st.session_state['excel_file_name'] = None
                st.stop()

        except Exception as e:
            st.error(f"❌ Error reading Excel file: {e}. Please ensure it's a valid .xlsx file and not corrupted.")
            st.session_state['excel_file_content'] = None
            st.session_state['excel_file_name'] = None
            st.stop()

        # Calculate Total Occupied SF and Total Available SF before plotting
        total_available_sf = data['Square Footage'].sum()
        if total_available_sf == 0:
            st.warning("⚠️ Total square footage in the building is zero. Cannot calculate occupancy percentage or plot effectively.")
            total_occupied_sf = 0
            occupancy_percentage = 0
            occupancy_percent_text = "N/A (Total SF is 0)"
        else:
            total_occupied_sf = data.loc[~data['Tenant Name'].str.upper().str.contains('VACANT'), 'Square Footage'].sum()
            occupancy_percentage = (total_occupied_sf / total_available_sf) * 100
            occupancy_percent_text = f"{occupancy_percentage:.1f}% ({int(total_occupied_sf):,} / {int(total_available_sf):,} SF)"

        # Calculate year totals, no expiry, and vacant totals for the legend
        year_totals = data.loc[~data['Tenant Name'].str.upper().str.contains('VACANT')].groupby('Expiration Year')['Square Footage'].sum()
        no_expiry_total = data.loc[data['Expiration Year'].isna() & ~data['Tenant Name'].str.upper().str.contains('VACANT'), 'Square Footage'].sum()
        vacant_total = data.loc[data['Tenant Name'].str.upper().str.contains('VACANT'), 'Square Footage'].sum()


        # Generate the plot using the function
        fig = generate_stacking_plan_plot(
            data,
            st.session_state.building_name,
            st.session_state.start_color,
            st.session_state.end_color,
            st.session_state.fig_width,
            st.session_state.fig_height,
            logo_file_to_display,
            st.session_state.logo_x,
            st.session_state.logo_y,
            st.session_state.logo_size,
            occupancy_percent_text,
            year_totals,
            vacant_total,
            no_expiry_total
        )

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
