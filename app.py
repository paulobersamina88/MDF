
import math
from typing import Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="MDOF Building Dynamics Explorer", layout="wide")

G = 9.81

def build_mass_matrix(weights_kN):
    masses = weights_kN / G
    return np.diag(masses)

def build_stiffness_matrix(k):
    n = len(k)
    K = np.zeros((n,n))
    for i in range(n):
        K[i,i] = k[i] + (k[i+1] if i+1<n else 0)
        if i<n-1:
            K[i,i+1] = -k[i+1]
            K[i+1,i] = -k[i+1]
    return K

def solve_modes(M,K):
    A = np.linalg.inv(M) @ K
    w2,phi = np.linalg.eig(A)
    idx = np.argsort(w2)
    w = np.sqrt(np.maximum(w2[idx],0))
    phi = phi[:,idx]
    return w,phi

def compute_periods(w):
    return np.where(w>0, 2*np.pi/w, 0)

def frame_plot(n,h):
    y = np.arange(0,n+1)*h
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0]*(n+1),y=y,mode="lines+markers"))
    fig.add_trace(go.Scatter(x=[6]*(n+1),y=y,mode="lines+markers"))
    for i in range(1,n+1):
        fig.add_trace(go.Scatter(x=[0,6],y=[y[i],y[i]],mode="lines"))
    return fig

st.title("MDOF Building Explorer")

n = st.sidebar.selectbox("Storeys",[2,3,4,5])
h = st.sidebar.number_input("Storey height",value=3.0)

w_input = st.sidebar.text_input("Weights kN","900,900,900,900,900")
k_input = st.sidebar.text_input("Stiffness kN/m","30000,30000,30000,30000,30000")

w = np.array([float(x) for x in w_input.split(",")[:n]])
k = np.array([float(x) for x in k_input.split(",")[:n]])

M = build_mass_matrix(w)
K = build_stiffness_matrix(k)
omega,phi = solve_modes(M,K)
T = compute_periods(omega)

st.write("Periods (s):",T)
st.plotly_chart(frame_plot(n,h))
