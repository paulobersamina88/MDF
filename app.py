import math
from typing import Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="MDOF Building Dynamics Explorer", layout="wide")

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
G = 9.81  # m/s^2


def ensure_positive(values: np.ndarray, fallback: float = 1.0) -> np.ndarray:
    arr = np.array(values, dtype=float)
    arr[arr <= 0] = fallback
    return arr


def build_mass_matrix(weights_kN: np.ndarray) -> np.ndarray:
    """Convert floor weights in kN to lumped masses in kN*s^2/m (= kN/g)."""
    masses = weights_kN / G
    return np.diag(masses)


def build_stiffness_matrix(story_stiffness_kN_per_m: np.ndarray) -> np.ndarray:
    n = len(story_stiffness_kN_per_m)
    K = np.zeros((n, n), dtype=float)

    for i in range(n):
        ki = story_stiffness_kN_per_m[i]
        kip1 = story_stiffness_kN_per_m[i + 1] if i + 1 < n else 0.0

        K[i, i] = ki + kip1
        if i < n - 1:
            K[i, i + 1] = -kip1
            K[i + 1, i] = -kip1

    return K


def solve_modes(M: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Solve generalized eigenvalue problem K phi = w^2 M phi."""
    A = np.linalg.solve(M, K)
    eigvals, eigvecs = np.linalg.eig(A)
    eigvals = np.real_if_close(eigvals)
    eigvecs = np.real_if_close(eigvecs)

    eigvals = np.asarray(eigvals, dtype=float)
    eigvecs = np.asarray(eigvecs, dtype=float)

    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    eigvals[eigvals < 0] = 0.0
    omega = np.sqrt(eigvals)

    # normalize each mode so max abs displacement = 1
    for i in range(eigvecs.shape[1]):
        mode = eigvecs[:, i]
        max_abs = np.max(np.abs(mode))
        if max_abs > 0:
            eigvecs[:, i] = mode / max_abs
        if eigvecs[-1, i] < 0:
            eigvecs[:, i] *= -1

    return omega, eigvecs


def modal_properties(M: np.ndarray, phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ones = np.ones((M.shape[0], 1))
    gamma = []
    modal_mass = []
    eff_modal_mass = []

    for i in range(phi.shape[1]):
        mode = phi[:, [i]]

        num = (mode.T @ M @ ones).item()
        den = (mode.T @ M @ mode).item()

        g_i = num / den if abs(den) > 1e-12 else 0.0
        m_i = den
        m_eff_i = (num ** 2) / den if abs(den) > 1e-12 else 0.0

        gamma.append(g_i)
        modal_mass.append(m_i)
        eff_modal_mass.append(m_eff_i)

    gamma = np.array(gamma, dtype=float)
    modal_mass = np.array(modal_mass, dtype=float)
    eff_modal_mass = np.array(eff_modal_mass, dtype=float)
    return gamma, modal_mass, eff_modal_mass


def compute_periods(omega: np.ndarray) -> np.ndarray:
    periods = np.zeros_like(omega)
    mask = omega > 0
    periods[mask] = 2 * np.pi / omega[mask]
    return periods


def approximate_code_base_shear(total_weight_kN: float, T1: float, SDS: float, R: float, Ie: float) -> float:
    """Simplified teaching expression inspired by ELF concepts."""
    T_use = max(T1, 0.05)
    Cs = SDS / max(R / Ie, 1e-6)

    period_factor = min(1.0, max(0.4, 1.0 / math.sqrt(T_use)))
    Cs_adj = Cs * period_factor

    Cs_adj = min(max(Cs_adj, 0.01), 0.30)
    return Cs_adj * total_weight_kN


def vertical_distribution_exponent(T: float) -> float:
    if T <= 0.5:
        return 1.0
    if T >= 2.5:
        return 2.0
    return 1.0 + (T - 0.5) / 2.0


def distribute_lateral_forces(weights_kN: np.ndarray, heights_m: np.ndarray, V_kN: float, T1: float) -> np.ndarray:
    k = vertical_distribution_exponent(T1)
    Cvx = weights_kN * (heights_m ** k)
    s = np.sum(Cvx)
    if s <= 0:
        return np.zeros_like(weights_kN)
    return V_kN * Cvx / s


def story_shear_from_floor_forces(floor_forces_kN: np.ndarray) -> np.ndarray:
    n = len(floor_forces_kN)
    Vstory = np.zeros(n)
    running = 0.0
    for i in range(n - 1, -1, -1):
        running += floor_forces_kN[i]
        Vstory[i] = running
    return Vstory


def compute_modal_floor_force_shape(M: np.ndarray, phi: np.ndarray, gamma: np.ndarray, mode_idx: int) -> np.ndarray:
    mode = phi[:, mode_idx]
    mass_diag = np.diag(M)
    shape = gamma[mode_idx] * mass_diag * mode
    if np.max(np.abs(shape)) > 0:
        shape = shape / np.max(np.abs(shape))
    return shape


def frame_plot(n_story: int, story_h: float, lateral: np.ndarray | None = None, title: str = "Frame View"):
    x_left = 0.0
    x_right = 6.0
    y = np.arange(0, n_story + 1) * story_h

    dx = np.zeros(n_story + 1)
    if lateral is not None:
        dx[1:] = lateral

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=x_left + dx, y=y, mode="lines+markers", name="Left Column"))
    fig.add_trace(go.Scatter(x=x_right + dx, y=y, mode="lines+markers", name="Right Column"))

    for i in range(1, n_story + 1):
        fig.add_trace(
            go.Scatter(
                x=[x_left + dx[i], x_right + dx[i]],
                y=[y[i], y[i]],
                mode="lines",
                showlegend=False,
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Horizontal Position / Relative Sway",
        yaxis_title="Height (m)",
        height=520,
    )
    return fig


def force_plot(floor_forces_kN: np.ndarray):
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=floor_forces_kN,
            y=[f"Floor {i+1}" for i in range(len(floor_forces_kN))],
            orientation="h",
            text=np.round(floor_forces_kN, 2),
            textposition="outside",
            name="Fx",
        )
    )
    fig.update_layout(title="Equivalent Lateral Force per Floor", xaxis_title="Force (kN)", height=420)
    return fig


def mode_shape_plot(phi: np.ndarray, story_h: float, mode_number: int):
    n = phi.shape[0]
    y = np.arange(1, n + 1) * story_h
    x = phi[:, mode_number - 1]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=np.r_[0, x],
            y=np.r_[0, y],
            mode="lines+markers",
            name=f"Mode {mode_number}",
        )
    )
    fig.update_layout(
        title=f"Mode Shape {mode_number}",
        xaxis_title="Normalized Lateral Displacement",
        yaxis_title="Height (m)",
        height=420,
    )
    return fig


# ---------------------------------------------------------
# UI
# ---------------------------------------------------------
st.title("🏢 MDOF Building Dynamics Explorer for 2 to 5 Storeys")
st.caption(
    "Interactive classroom app for modal properties, mass participation, and base shear distribution. "
    "This is a teaching tool and does not replace a full NSCP/IBC code design workflow."
)

with st.sidebar:
    st.header("Building Inputs")
    n_story = st.selectbox("Number of storeys", [2, 3, 4, 5], index=2)
    story_h = st.number_input("Typical storey height (m)", min_value=2.5, max_value=6.0, value=3.0, step=0.1)

    st.subheader("Dynamic / Seismic Inputs")
    code_basis = st.selectbox("Teaching basis", ["NSCP / IBC simplified ELF", "Modal dynamics only"])
    SDS = st.number_input("SDS", min_value=0.05, max_value=2.50, value=0.75, step=0.05)
    R = st.number_input("Response modification factor R", min_value=1.0, max_value=12.0, value=8.0, step=0.5)
    Ie = st.number_input("Importance factor Ie", min_value=0.5, max_value=2.0, value=1.0, step=0.1)

    st.subheader("Visualization")
    scale_factor = st.slider("Mode/frame exaggeration", min_value=0.5, max_value=30.0, value=8.0, step=0.5)
    selected_mode = st.selectbox("Mode to visualize", list(range(1, n_story + 1)), index=0)

st.markdown("### Floor Input Table")
default_weights = [900.0, 900.0, 900.0, 900.0, 900.0][:n_story]
default_k = [35000.0, 32000.0, 29000.0, 26000.0, 23000.0][:n_story]

input_df = pd.DataFrame(
    {
        "Floor": [f"Floor {i+1}" for i in range(n_story)],
        "Weight_kN": default_weights,
        "Story_Stiffness_kN_per_m": default_k,
    }
)

edited_df = st.data_editor(input_df, use_container_width=True, num_rows="fixed", key="floor_input_editor")
weights_kN = ensure_positive(edited_df["Weight_kN"].to_numpy(dtype=float), fallback=100.0)
stiffness_kNpm = ensure_positive(edited_df["Story_Stiffness_kN_per_m"].to_numpy(dtype=float), fallback=1000.0)
heights_m = np.arange(1, n_story + 1, dtype=float) * story_h

# ---------------------------------------------------------
# Analysis
# ---------------------------------------------------------
M = build_mass_matrix(weights_kN)
K = build_stiffness_matrix(stiffness_kNpm)
omega, phi = solve_modes(M, K)
periods = compute_periods(omega)
gamma, modal_mass, eff_modal_mass = modal_properties(M, phi)

mass_diag = np.diag(M)
total_mass = np.sum(mass_diag)
eff_mass_ratio = eff_modal_mass / total_mass if total_mass > 0 else np.zeros_like(eff_modal_mass)
cum_eff_mass_ratio = np.cumsum(eff_mass_ratio)

T1 = periods[0] if len(periods) > 0 else 0.0
W_total = float(np.sum(weights_kN))
Vbase = approximate_code_base_shear(W_total, T1, SDS, R, Ie) if code_basis == "NSCP / IBC simplified ELF" else 0.0
floor_forces = distribute_lateral_forces(weights_kN, heights_m, Vbase, T1) if Vbase > 0 else np.zeros(n_story)
story_shear = story_shear_from_floor_forces(floor_forces)

modal_force_shape = compute_modal_floor_force_shape(M, phi, gamma, selected_mode - 1)
visual_lateral = modal_force_shape * scale_factor

# ---------------------------------------------------------
# Results
# ---------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Fundamental Period T1 (s)", f"{T1:.4f}")
col2.metric("Total Weight W (kN)", f"{W_total:,.2f}")
col3.metric("Approx. Base Shear V (kN)", f"{Vbase:,.2f}")
col4.metric("1st Mode Mass Participation", f"{eff_mass_ratio[0]*100:.2f}%")

res_tab1, res_tab2, res_tab3, res_tab4 = st.tabs(
    ["Visualization", "Modal Properties", "Seismic Forces", "Matrices & Downloads"]
)

with res_tab1:
    a, b = st.columns([1.2, 1])

    with a:
        st.plotly_chart(
            frame_plot(n_story, story_h, title="Undeformed Frame"),
            use_container_width=True,
            key="undeformed_frame_chart",
        )
        st.plotly_chart(
            frame_plot(n_story, story_h, lateral=visual_lateral, title=f"Mode {selected_mode} Visualization"),
            use_container_width=True,
            key=f"mode_visualization_chart_{selected_mode}",
        )

    with b:
        st.plotly_chart(
            mode_shape_plot(phi, story_h, selected_mode),
            use_container_width=True,
            key=f"selected_mode_shape_chart_{selected_mode}",
        )

        viz_df = pd.DataFrame(
            {
                "Floor": [f"Floor {i+1}" for i in range(n_story)],
                "Height_m": heights_m,
                f"Mode_{selected_mode}_Shape": np.round(phi[:, selected_mode - 1], 5),
                "Scaled_Display_Displacement": np.round(visual_lateral, 5),
            }
        )
        st.dataframe(viz_df, use_container_width=True, key="viz_df_table")

with res_tab2:
    modal_df = pd.DataFrame(
        {
            "Mode": np.arange(1, n_story + 1),
            "Omega_rad_per_s": np.round(omega, 5),
            "Period_s": np.round(periods, 5),
            "Gamma": np.round(gamma, 5),
            "Modal_Mass": np.round(modal_mass, 5),
            "Effective_Modal_Mass": np.round(eff_modal_mass, 5),
            "Eff_Mass_Ratio_%": np.round(eff_mass_ratio * 100, 3),
            "Cumulative_%": np.round(cum_eff_mass_ratio * 100, 3),
        }
    )
    st.dataframe(modal_df, use_container_width=True, key="modal_df_table")

    mode_cols = st.columns(min(n_story, 4))
    for i in range(n_story):
        with mode_cols[i % len(mode_cols)]:
            fig_mode = mode_shape_plot(phi, story_h, i + 1)
            st.plotly_chart(
                fig_mode,
                use_container_width=True,
                key=f"mode_shape_chart_{i+1}",
            )

    mode_shape_df = pd.DataFrame(phi, columns=[f"Mode {i+1}" for i in range(n_story)])
    mode_shape_df.insert(0, "Floor", [f"Floor {i+1}" for i in range(n_story)])
    st.markdown("#### Mode Shape Matrix")
    st.dataframe(np.round(mode_shape_df, 5), use_container_width=True, key="mode_shape_df_table")

with res_tab3:
    if code_basis == "NSCP / IBC simplified ELF":
        c1, c2 = st.columns([1, 1])

        with c1:
            st.plotly_chart(
                force_plot(floor_forces),
                use_container_width=True,
                key="seismic_force_plot",
            )

        with c2:
            seismic_df = pd.DataFrame(
                {
                    "Floor": [f"Floor {i+1}" for i in range(n_story)],
                    "Height_m": heights_m,
                    "Weight_kN": np.round(weights_kN, 3),
                    "Fx_kN": np.round(floor_forces, 3),
                    "Story_Shear_kN": np.round(story_shear, 3),
                }
            )
            st.dataframe(seismic_df, use_container_width=True, key="seismic_df_table")

        st.info(
            "The base shear and vertical distribution shown here are simplified for classroom demonstration. "
            "For a strict NSCP or IBC design check, include the exact edition, full spectral parameters, "
            "period limits, regularity checks, and all required code minimums and maximums."
        )
    else:
        st.warning("Seismic floor force distribution is disabled in 'Modal dynamics only' mode.")

with res_tab4:
    mcol1, mcol2 = st.columns(2)

    with mcol1:
        st.markdown("#### Mass Matrix M")
        st.dataframe(pd.DataFrame(np.round(M, 5)), use_container_width=True, key="mass_matrix_table")

    with mcol2:
        st.markdown("#### Stiffness Matrix K")
        st.dataframe(pd.DataFrame(np.round(K, 5)), use_container_width=True, key="stiffness_matrix_table")

    export_modal = modal_df.copy()
    export_input = edited_df.copy()
    export_mode_shapes = mode_shape_df.copy()
    export_seismic = pd.DataFrame(
        {
            "Floor": [f"Floor {i+1}" for i in range(n_story)],
            "Height_m": heights_m,
            "Weight_kN": weights_kN,
            "Floor_Force_kN": floor_forces,
            "Story_Shear_kN": story_shear,
        }
    )

    st.download_button(
        "Download modal_properties.csv",
        export_modal.to_csv(index=False).encode("utf-8"),
        file_name="modal_properties.csv",
        mime="text/csv",
        key="download_modal_csv",
    )
    st.download_button(
        "Download mode_shapes.csv",
        export_mode_shapes.to_csv(index=False).encode("utf-8"),
        file_name="mode_shapes.csv",
        mime="text/csv",
        key="download_mode_shapes_csv",
    )
    st.download_button(
        "Download seismic_forces.csv",
        export_seismic.to_csv(index=False).encode("utf-8"),
        file_name="seismic_forces.csv",
        mime="text/csv",
        key="download_seismic_csv",
    )
    st.download_button(
        "Download input_table.csv",
        export_input.to_csv(index=False).encode("utf-8"),
        file_name="input_table.csv",
        mime="text/csv",
        key="download_input_csv",
    )

st.markdown("---")
st.markdown(
    "**How students can use this app:** change floor weight, storey stiffness, or seismic parameters and observe how "
    "the periods, mode shapes, mass participation, base shear, and floor force distribution change immediately."
)
