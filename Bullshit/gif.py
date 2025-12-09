import os
import imageio
import matplotlib.pyplot as plt
from PIL import Image

# ==========================
# IESTATI CEĻUS
# ==========================

PNG_FOLDER = r"C:\Users\deniss.boka\Desktop\interpolacija_Qgis\rezultati"
OUTPUT_GIF = os.path.join(PNG_FOLDER, "temperature.gif")
OUTPUT_SLIDER = os.path.join(PNG_FOLDER, "slider.html")

# ==========================
# MĒNEŠU SECĪBA
# ==========================

month_order = [
    "Janvāris", "Februāris", "Marts", "Aprīlis", "Maijs", "Jūnijs",
    "Julijs", "Augusts", "Septembris", "Oktobris", "Novembris", "Decembris"
]

# ==========================
# SAMEKLĒ PNG FAILUS
# ==========================

png_files = [p for p in os.listdir(PNG_FOLDER) if p.lower().endswith(".png")]

month_to_file = {}
for p in png_files:
    name = p.lower()
    for m in month_order:
        if m.lower() in name:
            month_to_file[m] = p

ordered_files = [month_to_file[m] for m in month_order if m in month_to_file]


# ==========================
# ZIEMEĻU BULTA
# ==========================

def add_north_arrow(ax):
    ax.annotate(
        "N",
        xy=(0.97, 0.97),
        xytext=(0.97, 0.87),
        xycoords="axes fraction",
        fontsize=30,
        ha="center",
        arrowprops=dict(facecolor="black", width=6, headwidth=20)
    )


# ==========================
# MĒROGA JOSLA
# ==========================

def add_scale_bar(ax, length_km=50, dpi=150, map_scale=1727608):

    km_per_cm = map_scale / 100000.0  
    cm_length = length_km / km_per_cm
    px_length = cm_length * (dpi / 2.54)

    fig = ax.get_figure()
    fig_w_px = fig.get_size_inches()[0] * dpi
    rel = px_length / fig_w_px

    y = 0.03

    ax.plot([0.10, 0.10 + rel], [y, y], transform=ax.transAxes,
            color="black", linewidth=5)

    ax.text(0.10, y - 0.04, f"{length_km} km",
            transform=ax.transAxes, fontsize=12, color="black")


# ==========================
# GENERĒ FRAME PNG
# ==========================

frames = []
DPI = 150   # IMPORTANT

for file, month in zip(ordered_files, month_order):

    img_path = os.path.join(PNG_FOLDER, file)
    img = Image.open(img_path)

    fig, ax = plt.subplots(figsize=(14, 8))

    plt.subplots_adjust(top=0.92, bottom=0.20, left=0.05, right=0.95)

    ax.imshow(img)
    ax.axis("off")

    ax.text(0.02, 0.96, month, transform=ax.transAxes,
            fontsize=28, color="white",
            ha="left", va="top",
            bbox=dict(facecolor="black", alpha=0.6, pad=10))

    add_north_arrow(ax)

    add_scale_bar(ax, dpi=DPI)

    frame_path = img_path.replace(".png", "_frame.png")
    plt.savefig(frame_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    frames.append(imageio.imread(frame_path))


# ==========================
# GIF
# ==========================

imageio.mimsave(OUTPUT_GIF, frames, duration=5.0)
print("GIF izveidots:", OUTPUT_GIF)


# ==========================
# HTML autoplay + slider
# ==========================

html = f"""
<html>
<head>
<meta charset="UTF-8">
<title>Temperature Slider</title>
<style>
body {{
    background:#111;
    text-align:center;
    color:white;
    font-family:Arial;
}}
img {{
    width:90%;
    margin-top:20px;
    border-radius:12px;
}}
input[type=range]{{
    width:60%;
    margin-top:40px;
}}
</style>
</head>
<body>

<h2>EXTERNAL DRIFT Krigings</h2>

<input type="range" min="1" max="12" value="1" id="slider">
<br>
<img id="img" src="{month_order[0]}_frame.png">

<script>
const names = {month_order};

let slider = document.getElementById("slider");
let img = document.getElementById("img");

let index = 0;

let timer = setInterval(nextImage, 3000);

function nextImage() {{
    index = (index + 1) % names.length;
    slider.value = index + 1;
    img.src = names[index] + "_frame.png";
}}

slider.oninput = function() {{
    index = this.value - 1;
    img.src = names[index] + "_frame.png";
    clearInterval(timer);
    timer = setInterval(nextImage, 3000);
}};
</script>

</body>
</html>
"""

with open(OUTPUT_SLIDER, "w", encoding="utf-8") as f:
    f.write(html)

print("HTML slider izveidots:", OUTPUT_SLIDER)
