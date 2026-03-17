import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOSS LANDSCAPE DEFINITIONS
#    We define two environments: 'Bumpy' for local minima and 'Canyon' for
#    ill-conditioned curvature (high Hessian condition number).
# ─────────────────────────────────────────────────────────────────────────────

def bumpy_loss(x, y):
    """
    Non-convex loss surface used to stress-test optimizers.

    L(x, y) = sin(x)cos(y) + 0.05(x^2 + y^2)

    - The trigonometric term creates many local minima/maxima.
    - The quadratic term keeps values bounded and adds a global bowl tendency.
    """
    return np.sin(x) * np.cos(y) + 0.05 * (x**2 + y**2)

def bumpy_grads(x, y):
    """
    Analytical gradient of bumpy_loss.

    dL/dx = cos(x)cos(y) + 0.1x
    dL/dy = -sin(x)sin(y) + 0.1y
    """
    dx = np.cos(x) * np.cos(y) + 0.1 * x
    dy = -np.sin(x) * np.sin(y) + 0.1 * y
    return np.array([dx, dy])

def canyon_loss(x, y):
    """
    Convex anisotropic quadratic:

    L(x, y) = 0.5x^2 + 0.05y^2

    Curvature in x is 10x larger than y, so one coordinate is stiff and the
    other is flat. This reveals optimizer sensitivity to ill-conditioning.
    """
    return (0.5 * x**2) + (0.05 * y**2)

def canyon_grads(x, y):
    """Gradient of canyon_loss: [dL/dx, dL/dy] = [x, 0.1y]."""
    dx = x 
    dy = 0.1 * y
    return np.array([dx, dy])

# ─────────────────────────────────────────────────────────────────────────────
# 2. OPTIMIZER UPDATE ENGINE
#    Centralized logic for all 8 optimizers using era-based buffers.
# ─────────────────────────────────────────────────────────────────────────────

def step_optimizer(opt_name, pos, grads, buffers, step, params):
    """
    Core update engine for all optimizers in this demo.

    Inputs:
    - pos: current parameter vector theta_t in R^2
    - grads: gradient g_t = grad L(theta_t)
    - buffers: state variables (m, v, h) for momentum/variance/Hessian proxy
    - step: iteration index t (1-based), needed for bias correction
    - params: hyperparameter dict

    Returns:
    - new position theta_{t+1}
    - updated buffers
    """
    lr = params.get('lr', 0.2)
    
    # --- FOUNDATION ERA ---
    if opt_name == "SGD":
        # theta_{t+1} = theta_t - eta * g_t
        return pos - lr * grads, buffers

    elif opt_name == "SGD + Momentum":
        # m_t = gamma*m_{t-1} + (1-gamma)g_t
        # theta_{t+1} = theta_t - eta*m_t
        gamma = params.get('gamma', 0.9)
        buffers['m'] = gamma * buffers['m'] + (1 - gamma) * grads
        return pos - lr * buffers['m'], buffers

    elif opt_name == "AdaGrad":
        # v_t = v_{t-1} + g_t^2 (element-wise accumulation)
        # theta_{t+1} = theta_t - eta * g_t / (sqrt(v_t) + eps)
        buffers['v'] += grads**2
        return pos - lr * grads / (np.sqrt(buffers['v']) + 1e-8), buffers

    elif opt_name == "RMSProp":
        # v_t = rho*v_{t-1} + (1-rho)g_t^2
        # theta_{t+1} = theta_t - eta * g_t / (sqrt(v_t) + eps)
        rho = params.get('rho', 0.9)
        buffers['v'] = rho * buffers['v'] + (1 - rho) * (grads**2)
        return pos - lr * grads / (np.sqrt(buffers['v']) + 1e-8), buffers

    elif opt_name == "Adam":
        # m_t = b1*m_{t-1} + (1-b1)g_t
        # v_t = b2*v_{t-1} + (1-b2)g_t^2
        # m_hat = m_t/(1-b1^t), v_hat = v_t/(1-b2^t)
        # theta_{t+1} = theta_t - eta * m_hat / (sqrt(v_hat) + eps)
        b1, b2 = params.get('b1', 0.9), params.get('b2', 0.999)
        buffers['m'] = b1 * buffers['m'] + (1 - b1) * grads
        buffers['v'] = b2 * buffers['v'] + (1 - b2) * (grads**2)
        # Bias Correction: Essential for Adam's stability in early steps
        m_h, v_h = buffers['m'] / (1 - b1**step), buffers['v'] / (1 - b2**step)
        return pos - lr * m_h / (np.sqrt(v_h) + 1e-8), buffers

    # --- MODERN ERA ---
    elif opt_name == "AdamW":
        # Decoupled weight decay then Adam-style adaptive step:
        # theta <- (1 - lambda*eta) * theta
        pos = pos * (1 - 0.01 * lr) 
        b1, b2 = params.get('b1_w', 0.9), params.get('b2_w', 0.999)
        buffers['m'] = b1 * buffers['m'] + (1 - b1) * grads
        buffers['v'] = b2 * buffers['v'] + (1 - b2) * (grads**2)
        m_h, v_h = buffers['m'] / (1 - b1**step), buffers['v'] / (1 - b2**step)
        return pos - lr * m_h / (np.sqrt(v_h) + 1e-8), buffers

    elif opt_name == "Lion":
        # Lion update uses sign of momentum-like direction:
        # u_t = sign(b1*m_{t-1} + (1-b1)g_t), theta_{t+1} = theta_t - eta*u_t
        b1_l, b2_l = params.get('b1_l', 0.9), params.get('b2_l', 0.99)
        update = np.sign(b1_l * buffers['m'] + (1 - b1_l) * grads)
        new_pos = pos - lr * update
        buffers['m'] = b2_l * buffers['m'] + (1 - b2_l) * grads 
        return new_pos, buffers

    elif opt_name == "Sophia":
        # Diagonal second-order proxy:
        # h_t tracks abs(grad) as curvature proxy; step scales by 1/max(h_t, floor)
        b1_s, b2_s = params.get('b1_s', 0.96), params.get('b2_s', 0.99)
        buffers['m'] = b1_s * buffers['m'] + (1 - b1_s) * grads
        buffers['h'] = b2_s * buffers['h'] + (1 - b2_s) * np.abs(grads)
        return pos - lr * buffers['m'] / (np.maximum(0.01, buffers['h'])), buffers

# ─────────────────────────────────────────────────────────────────────────────
# 3. GLOBAL APP CONFIGURATION & NAVIGATION
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(layout="wide", page_title="Neural Optimizer Lab")
st.title("🔬 Neural Optimizer Lab: The Evolutionary Race")

# Pre-calculate meshgrid so each frame reuses the same landscape surface values.
x_r, y_r = np.linspace(-6, 6, 100), np.linspace(-6, 6, 100)
X, Y = np.meshgrid(x_r, y_r)

tab1, tab2 = st.tabs(["🎮 Sandbox Experiment", "🏁 Multi-Scenario Benchmark"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: THE SANDBOX (Boxed Hyperparams & Distinct Markers)
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    col_ctrl, col_plot = st.columns([1, 3])
    
    with col_ctrl:
        st.subheader("Global Control")
        sb_lr = st.slider("Shared Learning Rate (η)", 0.01, 1.0, 0.2)
        sb_iter = st.slider("Iterations", 10, 150, 60)
        
        # ── FOUNDATION HYPERPARAMS (Boxed) ──
        # ── FOUNDATION HYPERPARAMS (Boxed, 2-col compact) ──
        with st.container(border=True):
            st.markdown("**🏛️ Foundation Parameters**")
            _c1, _c2 = st.columns(2)
            f_gamma = _c1.slider("Momentum γ", 0.0, 1.0, 0.9)
            f_rho   = _c2.slider("RMSProp ρ",  0.0, 1.0, 0.9)
            _c3, _c4 = st.columns(2)
            f_b1 = _c3.slider("Adam β1", 0.0, 0.99,  0.9)
            f_b2 = _c4.slider("Adam β2", 0.0, 0.999, 0.999)

        # ── ADVANCED HYPERPARAMS (Boxed, 2-col compact) ──
        with st.container(border=True):
            st.markdown("**🚀 Advanced Parameters**")
            _c5, _c6 = st.columns(2)
            a_b1_l = _c5.slider("Lion β1",   0.0, 0.99, 0.9)
            a_b2_l = _c6.slider("Lion β2",   0.0, 0.99, 0.99)
            _c7, _c8 = st.columns(2)
            a_b1_s = _c7.slider("Sophia β1", 0.0, 0.99, 0.96)
            a_b2_s = _c8.slider("Sophia β2", 0.0, 0.99, 0.99)

        era = st.radio("Active Set", ["Foundation Set", "Advanced Set"])

    plot_spot = col_plot.empty()
    Z_bumpy = bumpy_loss(X, Y)

    # Era definitions with high-contrast colors and unique markers
    if era == "Foundation Set":
        opts = ["SGD", "SGD + Momentum", "AdaGrad", "RMSProp", "Adam"]
        colors = ['red', 'white', 'yellow', 'cyan', 'magenta']
        markers = ['o', 's', 'v', '^', 'D'] # Circle, Square, Triangle Down, Triangle Up, Diamond
    else:
        opts = ["Adam", "AdamW", "Lion", "Sophia"]
        colors = ['white', 'cyan', 'orange', 'magenta']
        markers = ['D', '*', '+', 'x'] # Diamond, Star, Plus, X

    p_dict = {
        'lr': sb_lr, 'gamma': f_gamma, 'rho': f_rho, 'b1': f_b1, 'b2': f_b2,
        'b1_w': f_b1, 'b2_w': f_b2,
        'b1_l': a_b1_l, 'b2_l': a_b2_l, 'b1_s': a_b1_s, 'b2_s': a_b2_s
    }

    # All paths start at Upper Right (4,4)
    start_pt = np.array([4.0, 4.0])
    paths = {o: [start_pt.copy()] for o in opts}
    bufs = {o: {'m': np.zeros(2), 'v': np.zeros(2), 'h': np.zeros(2)} for o in opts}

    # Convergence proxy: first t where ||theta_{t+1} - theta_t||_2 < threshold.
    # This measures update size, not exact optimality (useful for visualization).
    CONV_THRESH = 1e-3
    convergence = {o: None for o in opts}

    for i in range(1, sb_iter + 1):
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.contourf(X, Y, Z_bumpy, levels=30, cmap='viridis', alpha=0.7)
        ax.axhline(0, color='white', linestyle='--', alpha=0.3)
        ax.axvline(0, color='white', linestyle='--', alpha=0.3)

        for idx, o in enumerate(opts):
            curr_pos = paths[o][-1]
            g = bumpy_grads(*curr_pos)
            new_p, new_b = step_optimizer(o, curr_pos, g, bufs[o], i, p_dict)
            paths[o].append(new_p)
            bufs[o] = new_b

            # Record convergence iteration (first time step size < threshold)
            if convergence[o] is None and np.linalg.norm(new_p - curr_pos) < CONV_THRESH:
                convergence[o] = i

            p_arr = np.array(paths[o])

            # Show convergence summary directly in final-frame legend for screenshots.
            is_final = (i == sb_iter)
            if is_final:
                conv_val = convergence[o]
                conv_tag = f"iter {conv_val}" if conv_val is not None else f">{sb_iter}"
                legend_label = f"{o}  [conv: {conv_tag}]"
            else:
                legend_label = o

            ax.plot(p_arr[:,0], p_arr[:,1], color=colors[idx], marker=markers[idx],
                    linestyle='-', label=legend_label, markersize=3, alpha=0.5, linewidth=1.5)

        if era == "Foundation Set":
            ax.set_xlim(0, 6); ax.set_ylim(0, 6)
        else:
            ax.set_xlim(-3, 6); ax.set_ylim(-1, 6)
        ax.set_xlabel("X (Zoomed)"); ax.set_ylabel("Y")
        ax.legend(); ax.set_title(f"Sandbox: {era} (Iteration {i})")
        plot_spot.pyplot(fig)
        plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: MULTI-SCENARIO BENCHMARK
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.info("The Benchmark tab applies this multi-marker logic across 4 stress scenarios.")
    col_l, col_r = st.columns([1, 3])
    with col_l:
        st.subheader("Benchmark Environment")
        l_choice = st.selectbox("Landscape", ["Bumpy Egg-Crate", "Elliptical Canyon"], key="bl")
        e_bench = st.radio("Benchmark Set", ["Foundation", "Modern"])
        st.divider()
        st.slider("Locked LR", 0.01, 1.0, 0.2, disabled=True, key="ll")
        st.slider("Max Steps", 10, 120, 120, disabled=True, key="lit")
        st.divider()
        with st.container(border=True):
            st.markdown("**🏛️ Foundation (fixed)**")
            _bc1, _bc2 = st.columns(2)
            _bc1.slider("Momentum γ", 0.0, 1.0,   0.9,   disabled=True, key="bl_gam")
            _bc2.slider("RMSProp ρ",  0.0, 1.0,   0.9,   disabled=True, key="bl_rho")
            _bc3, _bc4 = st.columns(2)
            _bc3.slider("Adam β1",    0.0, 0.99,  0.9,   disabled=True, key="bl_b1")
            _bc4.slider("Adam β2",    0.0, 0.999, 0.999, disabled=True, key="bl_b2")
        with st.container(border=True):
            st.markdown("**🚀 Advanced (fixed)**")
            _bc5, _bc6 = st.columns(2)
            _bc5.slider("Lion β1",   0.0, 0.99,  0.9,  disabled=True, key="bl_lb1")
            _bc6.slider("Lion β2",   0.0, 0.99,  0.99, disabled=True, key="bl_lb2")
            _bc7, _bc8 = st.columns(2)
            _bc7.slider("Sophia β1", 0.0, 0.99,  0.96, disabled=True, key="bl_sb1")
            _bc8.slider("Sophia β2", 0.0, 0.99,  0.99, disabled=True, key="bl_sb2")

    # Landscape switch determines both objective function and its gradient field.
    f_b, g_b = (bumpy_loss, bumpy_grads) if l_choice == "Bumpy Egg-Crate" else (canyon_loss, canyon_grads)
    b_plot_spot = col_r.empty()
    Z_bench = f_b(X, Y)

    opts_b = ["SGD", "SGD + Momentum", "AdaGrad", "RMSProp", "Adam"] if e_bench == "Foundation" else ["Adam", "AdamW", "Lion", "Sophia"]
    presets = {
        "1. Ideal": {"lr": 0.2, "noise": 0.0, "steps": 70},
        "2. Divergence": {"lr": 0.85, "noise": 0.0, "steps": 70},
        "3. Stochastic": {"lr": 0.15, "noise": 0.4, "steps": 70},
        "4. Vanishing": {"lr": 0.02, "noise": 0.0, "steps": 120}
    }

    # Each optimizer stores one trajectory per scenario (4 panels).
    # Shape idea: b_paths[optimizer][scenario][time] -> R^2 vector
    b_paths = {o: [ [start_pt.copy()] for _ in range(4) ] for o in opts_b}
    b_bufs = {o: [ {'m': np.zeros(2), 'v': np.zeros(2), 'h': np.zeros(2)} for _ in range(4) ] for o in opts_b}

    for i in range(1, 121):
        fig_m, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        for idx, (name, cfg) in enumerate(presets.items()):
            ax = axes[idx]
            ax.contourf(X, Y, Z_bench, levels=25, cmap='viridis', alpha=0.5)
            ax.axhline(0, color='white', linestyle='--', alpha=0.3)
            ax.axvline(0, color='white', linestyle='--', alpha=0.3)
            
            for o in opts_b:
                if i <= cfg['steps']:
                    cur = b_paths[o][idx][-1]
                    # Add isotropic Gaussian noise to mimic stochastic mini-batch gradients.
                    grads = g_b(*cur) + np.random.normal(0, cfg['noise'], 2)
                    new_p, new_b = step_optimizer(o, cur, grads, b_bufs[o][idx], i, cfg)
                    b_paths[o][idx].append(new_p)
                    b_bufs[o][idx] = new_b

                p_arr = np.array(b_paths[o][idx])
                ax.plot(p_arr[:,0], p_arr[:,1], '-', alpha=0.6, label=o if idx==0 and i==1 else "")

            ax.set_title(f"{name} (Step {min(i, cfg['steps'])})")
            ax.set_xlabel("X Axis"); ax.set_ylabel("Y Axis")
            ax.set_xlim(-6, 6); ax.set_ylim(-6, 6)

        b_plot_spot.pyplot(fig_m)
        plt.close(fig_m)