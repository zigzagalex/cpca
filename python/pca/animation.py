import io
import os

import imageio.v2 as imageio
import numpy as np
import plotly.io as pio
from golubreinsch import golub_reinsch_svd_snapshots
from plotmatrix import plot_b_history


def b_history_to_gif(
    B_hist,
    gif_path="svd.gif",
    *,
    bar_depth=0.9,
    colorscale="Viridis",
    frame_ms=500,
    transition_ms=150,
    width=900,
    height=700,
    scale=2,
    show_plot_interactive=False,
):
    """
    Render the cuboid animation with plot_b_history(), grab every frame
    as PNG via Kaleido, and write them out as an animated GIF.
    """
    # build the interactive animation without showing it —
    fig = plot_b_history(
        B_hist,
        bar_depth=bar_depth,
        colorscale=colorscale,
        frame_ms=frame_ms,
        transition_ms=transition_ms,
        show=show_plot_interactive,
    )
    # grab the first (static) scene
    png_bytes = pio.to_image(fig, format="png", width=width, height=height, scale=scale)
    frames = [imageio.imread(io.BytesIO(png_bytes))]

    # iterate over the remaining animation frames
    for fr in fig.frames:  # skips t=0
        fig.update(data=fr.data)  # draw this frame
        png_bytes = pio.to_image(
            fig, format="png", width=width, height=height, scale=scale
        )
        frames.append(imageio.imread(io.BytesIO(png_bytes)))

    # write the GIF
    print("Writing GIF")
    fps = 30 / frame_ms  # frame_ms → frames / sec
    imageio.mimsave(gif_path, frames, fps=fps)
    print(f"✔ GIF saved → {os.path.abspath(gif_path)}")


if __name__ == "__main__":
    A = np.random.rand(10, 10)
    B_hist, V_hist, U_hist = golub_reinsch_svd_snapshots(A)
    b_history_to_gif(B_hist, gif_path="svd.gif", colorscale="Turbo")
