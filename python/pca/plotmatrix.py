import numpy as np
import plotly.graph_objects as go
from plotly.colors import sample_colorscale


#  One Mesh3d per entry so each bar gets its own colour
def _gradient_cuboids(M, *, bar_depth=1.0, colorscale="Viridis"):
    """Return a list of Mesh3d cuboids for matrix M, coloured by |value|."""
    m, n = M.shape
    abs_max = np.max(np.abs(M)) or 1.0  # avoid DIV/0
    traces = []

    def add_cube(i, j, z_val):
        x0, x1 = j, j + bar_depth
        y0, y1 = i, i + bar_depth
        z0, z1 = 0.0, z_val

        verts = np.array(
            [
                [x0, y0, z0],
                [x1, y0, z0],
                [x1, y1, z0],
                [x0, y1, z0],  # bottom
                [x0, y0, z1],
                [x1, y0, z1],
                [x1, y1, z1],
                [x0, y1, z1],  # top
            ]
        )
        faces = np.array(
            [
                [0, 1, 2],
                [0, 2, 3],  # bottom
                [4, 5, 6],
                [4, 6, 7],  # top
                [0, 1, 5],
                [0, 5, 4],  # front
                [1, 2, 6],
                [1, 6, 5],  # right
                [2, 3, 7],
                [2, 7, 6],  # back
                [3, 0, 4],
                [3, 4, 7],  # left
            ]
        )
        colour = sample_colorscale(colorscale, abs(z_val) / abs_max)[0]

        traces.append(
            go.Mesh3d(
                x=verts[:, 0],
                y=verts[:, 1],
                z=verts[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color=colour,
                flatshading=True,
                showscale=False,
                hoverinfo="skip",
            )
        )

    for i in range(m):
        for j in range(n):
            if M[i, j] != 0:
                add_cube(i, j, M[i, j])

    return traces


#  Cuboids-only animation
def plot_b_history(
    B_hist,
    *,
    bar_depth=1.0,
    colorscale="Viridis",
    frame_ms=500,
    transition_ms=150,
    show=False,
):
    """
    Animate the evolution of B only (filled 3-D cuboids, value-gradient).
    """
    steps = len(B_hist)
    if steps == 0:
        raise ValueError("B_hist is empty - nothing to plot.")

    # base figure: ONE scene
    fig = go.Figure()

    # frame 0
    for tr in _gradient_cuboids(B_hist[0], bar_depth=bar_depth, colorscale=colorscale):
        fig.add_trace(tr)

    # animation frames
    frames = []
    for t in range(1, steps):
        frames.append(
            go.Frame(
                name=f"step {t}",
                data=_gradient_cuboids(
                    B_hist[t], bar_depth=bar_depth, colorscale=colorscale
                ),
            )
        )
    fig.frames = frames

    # layout + controls
    fig.update_layout(
        autosize=True,
        margin=dict(l=20, r=20, t=40, b=20),
        scene=dict(
            xaxis_title="column j",
            yaxis_title="row i",
            zaxis_title="value",
            aspectmode="data",
            camera=dict(eye=dict(x=1.5, y=1.3, z=0.9)),
        ),
        sliders=[
            {
                "pad": {"b": 10},
                "len": 0.9,
                "currentvalue": {"prefix": "step = "},
                "steps": [
                    dict(
                        label=str(k),
                        method="animate",
                        args=[
                            [f"step {k}"],
                            {
                                "mode": "immediate",
                                "frame": {"duration": 0},
                                "transition": {"duration": 0},
                            },
                        ],
                    )
                    for k in range(steps)
                ],
            }
        ],
        updatemenus=[
            {
                "type": "buttons",
                "buttons": [
                    {
                        "label": "▶︎ Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "fromcurrent": True,
                                "frame": {"duration": frame_ms},
                                "transition": {"duration": transition_ms},
                            },
                        ],
                    },
                    {
                        "label": "⏹ Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "mode": "immediate",
                                "frame": {"duration": 0},
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                ],
                "direction": "left",
                "pad": {"t": 5},
                "showactive": False,
            }
        ],
        title="S-matrix evolution through Householder and Golub-Reinsch Algorithms",
    )

    if show:
        fig.show(config={"responsive": True})

    return fig
