import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from io import BytesIO
from PIL import Image

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

# 🏢 Building name input
building_name = st.text_input("🏢 Enter building name or address for this stacking plan", "My Building")

# 🎨 Color pickers for gradient
start_color = st.color_picker("🎨 Choose start color (earliest expiration year)", "#FF0000")
end_color = st.color_picker("🎨 Choose end color (latest expiration year)", "#00FF00")

# 🖼️ Logo upload
logo_file = st.file_uploader("Upload a logo image (PNG or JPG) to include on the stacking plan", type=["png", "jpg", "jpeg"])

# 🖼️ Logo position adjustment sliders
if logo_file is not None:
    st.subheader("🔧 Adjust Logo Position")

    logo_x = st.slider("Logo X position (pixels from left)", min_value=0, max_value=1000, value=50, step=10)
    logo_y = st.slider("Logo Y position (pixels from bottom)", min_value=0, max_value=1000, value=50, step=10)

    # Optional: Logo size slider
    logo_size = st.slider("Logo max size (pixels)", min_value=50, max_value=500, value=150, step=10)


# File upload for stacking data
uploaded_file = st.file_uploader("Upload your Excel file here (.xlsx)")

if uploaded_file is not None:
    # Read data
    data = pd.read_excel(uploaded_file)

    # Parse expiration dates as datetime
    data['Expiration Date'] = pd.to_datetime(data['Expiration Date'])
    data['Expiration Year'] = data['Expiration Date'].dt.year

    # Sort data
    data = data.sort_values(by=['Floor', 'Suite Number'], ascending=[False, True])

    # Prepare colormap with user-selected colors
    years = data.loc[~data['Tenant Name'].str.upper().str.contains('VACANT'), 'Expiration Year'].dropna().unique()
    years = np.sort(years)

    if len(years) == 0:
        years = np.array([2025, 2030])

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

    # -------------------------
    # Calculate occupancy totals
    # -------------------------
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

    # -------------------------
    # Create plot
    # -------------------------
    fig, ax = plt.subplots(figsize=(25, 14))
    y_pos = 0
    height = 1
    plot_width = 10

    floors = sorted(data['Floor'].unique(), reverse=True)

    for floor in floors:
        floor_data = data[data['Floor'] == floor]
        floor_sum = floor_data['Square Footage'].sum()
        x_pos = 0

        # Floor label + total SF on the left
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

            ax.text(x=x_pos + width/2, y=y_pos,
                    s=f"{tenant}\nSuite {suite}\n{suite_sf} SF\n{expiry}",
                    ha='center', va='center', fontsize=6)

            x_pos += width

        y_pos += height

    # Formatting
    ax.set_xlabel('Proportional Suite Width (normalized per floor)')
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title(f'Stacking Plan - {building_name}', fontsize=14, fontweight='bold')
    ax.invert_yaxis()

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.tick_params(bottom=False)

    plt.tight_layout()

    # Add logo to figure if uploaded
    if logo_file is not None:
        logo = Image.open(logo_file)
        logo.thumbnail((logo_size, logo_size))  # Resize dynamically based on user input
        fig.figimage(logo, xo=logo_x, yo=logo_y, alpha=1, zorder=10)


    # Legend for colors at bottom
    legend_elements = []

    for year in years:
        color = mcolors.to_hex(cmap(norm(year)))
        legend_elements.append(
            mpatches.Patch(facecolor=color, edgecolor='black', label=str(int(year)))
        )

    legend_elements.append(
        mpatches.Patch(facecolor='#d3d3d3', edgecolor='black', label='VACANT')
    )

    legend_elements.append(
        mpatches.Patch(facecolor='#1f77b4', edgecolor='black', label='No Expiry Date')
    )

    ax.text(0.5, -0.1, f"Total SF by Expiration Year: {occupancy_text}",
            transform=ax.transAxes,
            ha='center', va='top', fontsize=10, fontweight='bold')

    ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.2),
              ncol=len(legend_elements), fontsize=8)

    # Show plot in Streamlit
    st.pyplot(fig)

    # Save figure to buffer for download
    pdf_buf = BytesIO()
    fig.savefig(pdf_buf, format="pdf", bbox_inches="tight")
    pdf_buf.seek(0)

    st.download_button(
        label="Download Stacking Plan as PDF",
        data=pdf_buf,
        file_name=f"{building_name}_stacking_plan.pdf",
        mime="application/pdf"
    )

    png_buf = BytesIO()
    fig.savefig(png_buf, format="png", bbox_inches="tight")
    png_buf.seek(0)

    st.download_button(
        label="Download Stacking Plan as PNG",
        data=png_buf,
        file_name=f"{building_name}_stacking_plan.png",
        mime="application/png"
    )

    st.success("✅ Stacking plan generated!")
