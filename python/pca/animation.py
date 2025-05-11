import pathlib

import numpy as np
import plotly.io as pio
from golubreinsch import golub_reinsch_svd_snapshots
from plotmatrix import plot_b_history


def export_b_plot_for_readme(
    B_hist,
    *,
    out_dir="assets",
    html_name="svd_interactive.html",
    png_name="svd_thumb.png",
    bar_depth=0.9,
    colorscale="Turbo",
    frame_ms=750,
    transition_ms=150,
    png_width=900,
    png_height=700,
    png_scale=2,
):
    """
    • Builds the interactive cuboid figure (no browser pop-up).
    • Saves it as HTML and as a PNG thumbnail.
    • Prints a Markdown snippet that links thumbnail → interactive plot.
    """
    # build figure
    fig: go.Figure = plot_b_history(
        B_hist,
        bar_depth=bar_depth,
        colorscale=colorscale,
        frame_ms=frame_ms,
        transition_ms=transition_ms,
        show=False,  # headless
    )

    # strip controls so thumb & HTML look clean
    fig.update_layout(sliders=[], updatemenus=[])

    # ensure output folder exists
    out_path = pathlib.Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # full paths
    html_path = out_path / html_name
    png_path = out_path / png_name

    # save interactive HTML (self-contained, JS via CDN)
    fig.write_html(html_path, include_plotlyjs="cdn", full_html=True)

    # save static PNG thumbnail
    pio.write_image(
        fig,
        png_path,
        format="png",
        width=png_width,
        height=png_height,
        scale=png_scale,
        engine="kaleido",
    )

    print("✔ Saved:")
    print("  •", html_path.resolve())
    print("  •", png_path.resolve())
    print()

    # Markdown snippet ----------------------------------------------------
    # GitHub READMEs cannot run JS, so we link the thumbnail to the HTML.
    # If you publish on GitHub-Pages, swap the `html_preview_link` with the
    # final URL (e.g. https://USERNAME.github.io/REPO/assets/svd_interactive.html)

    html_preview_link = (
        f"https://htmlpreview.github.io/?"
        f"https://raw.githubusercontent.com/"
        f"YOUR_GH_USERNAME/YOUR_REPO_NAME/MAIN_BRANCH/{html_path.as_posix()}"
    )

    md = f"[![SVD cuboids]({png_path.as_posix()})]({html_preview_link})"

    print("--- Copy-paste this into your README.md ---")
    print(md)
    print("-" * 60)


if __name__ == "__main__":
    A = np.random.rand(10, 10)
    B_hist, *_ = golub_reinsch_svd_snapshots(A)
    export_b_plot_for_readme(B_hist)
