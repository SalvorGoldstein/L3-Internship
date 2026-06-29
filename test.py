import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp

# ─────────────────────────────────────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────────────────────────────────────
# We work in D=2 + E=1 dimensions for visualization:
#   full space: (x, y, theta)  where theta in [0, 2*pi) is the compact direction
#   D-dim space: (x, y)
#   internal:    theta
#
# The Kaluza-Klein metric (eq.8, 14a of Scherk-Schwarz) is:
#
#   g_hat_{mu nu}   = g_{mu nu} + A_mu A_nu * R^2    (D-dim part)
#   g_hat_{mu,int}  = R^2 * A_mu                     (off-diagonal = gauge field)
#   g_hat_{int,int} = R^2                             (internal)
#
# where A_mu(x,y) is the KK gauge field and R is the internal radius.
# A free particle (geodesic) in this metric looks like a CHARGED particle in D dims.
# The conserved momentum along theta = electric charge q.
#
# Geodesic equations from the KK metric:
#   d²x^i/dt² + Gamma^i_{jk} dx^j/dt dx^k/dt = 0
#
# For our metric with A_mu = A_mu(x,y) (a background gauge field in 2D):
#   D-dim equation of motion (Lorentz force!):
#     d²x^i/dt² = q * F^i_j * dx^j/dt
#   where q = R * dtheta/dt (conserved KK charge) and F_{ij} = d_i A_j - d_j A_i
#
# This is the KEY result: a free particle in D+E dims looks like
# a CHARGED particle in a magnetic field in D dims.
# ─────────────────────────────────────────────────────────────────────────────

R = 0.15    # compactification radius (small -> KK approximation valid)

# ── Choose a background gauge field A_mu(x,y) ────────────────────────────────
# Case 1: uniform "magnetic" field B in 2D  ->  A_x=0, A_y=B*x
# The field strength F_{xy} = dA_y/dx - dA_x/dy = B (constant)
B_field = 1.5   # effective magnetic field seen in D=2

def A_vec(x, y):
    """Gauge field: A = (0, B*x)  -> uniform magnetic field B in 2D."""
    return np.array([0.0, B_field * x])

def curl_A(x, y):
    """F_{xy} = dA_y/dx - dA_x/dy = B  (field strength)."""
    return B_field   # constant

# ── Equations of motion ───────────────────────────────────────────────────────
def eom(t, state, q):
    """
    state = [x, y, vx, vy]
    EOM: d²x^i/dt² = q * F^i_j * v^j
    In 2D: F_{xy} = B, so:
      ax = q * B * vy
      ay = -q * B * vx
    This is a Lorentz force: a = q * v × B
    """
    x, y, vx, vy = state
    B = curl_A(x, y)
    ax =  q * B * vy
    ay = -q * B * vx
    return [vx, vy, ax, ay]

# ── Integrate for several values of KK charge q ───────────────────────────────
q_values   = [0.0, 0.5, 1.0, 2.0, -1.0]
colors     = ['#888780', '#2a78d6', '#1baf7a', '#eda100', '#e34948']
labels     = ['q=0 (neutral)', 'q=0.5', 'q=1.0', 'q=2.0', 'q=−1.0 (opposite charge)']
linestyles = ['-', '-', '-', '-', '--']

t_span = (0, 8)
t_eval = np.linspace(*t_span, 1000)
ic     = [0.0, 0.0, 1.0, 0.0]   # start at origin, moving in x direction

solutions = []
for q in q_values:
    sol = solve_ivp(eom, t_span, ic, args=(q,), t_eval=t_eval,
                    method='RK45', rtol=1e-8, atol=1e-10)
    solutions.append(sol)

# ── FIGURE 1: static plot of all trajectories ────────────────────────────────
fig1, axes1 = plt.subplots(1, 2, figsize=(13, 5))

ax = axes1[0]
for sol, color, label, ls in zip(solutions, colors, labels, linestyles):
    ax.plot(sol.y[0], sol.y[1], color=color, lw=1.8, ls=ls, label=label)
ax.set_xlabel('x', fontsize=11)
ax.set_ylabel('y', fontsize=11)
ax.set_title('D-dim projection of free KK particle\n'
             '(looks like charged particle in B field)', fontsize=11)
ax.legend(fontsize=9)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.scatter([0], [0], color='black', s=40, zorder=5)
ax.text(0.05, 0.05, 'start', fontsize=8, color='gray')

# right panel: radius of curvature vs q  (r = v/(qB))
ax2 = axes1[1]
q_arr = np.linspace(0.1, 3.0, 200)
r_arr = 1.0 / (np.abs(q_arr) * B_field)   # v=1, so r = 1/(|q|B)
ax2.plot(q_arr, r_arr, '#2a78d6', lw=2)
ax2.set_xlabel('KK charge  q = R · dθ/dt', fontsize=11)
ax2.set_ylabel('Radius of curvature  r = v/(|q|B)', fontsize=11)
ax2.set_title('Heavier KK charge -> tighter circular orbit', fontsize=11)
ax2.grid(True, alpha=0.3)
for q, color in zip([0.5, 1.0, 2.0], ['#2a78d6','#1baf7a','#eda100']):
    r = 1.0/(q*B_field)
    ax2.scatter([q], [r], color=color, s=60, zorder=5)
    ax2.text(q+0.05, r+0.02, f'q={q}', fontsize=9, color=color)

plt.suptitle('Kaluza-Klein particle  —  free geodesic in (D+E) dims\n'
             'projects to a charged particle trajectory in D dims', fontsize=12)
plt.tight_layout()
plt.savefig('kk_trajectories.png', dpi=150, bbox_inches='tight')

# ── FIGURE 2: animation ───────────────────────────────────────────────────────
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
ax_2d = axes2[0]
ax_th = axes2[1]

ax_2d.set_xlim(-1.2, 1.2); ax_2d.set_ylim(-1.2, 1.2)
ax_2d.set_aspect('equal'); ax_2d.grid(True, alpha=0.3)
ax_2d.set_xlabel('x'); ax_2d.set_ylabel('y')
ax_2d.set_title('D-dim projection (x, y)', fontsize=11)
ax_2d.scatter([0], [0], color='black', s=30, zorder=5)

ax_th.set_xlim(0, 8); ax_th.set_ylim(-2*np.pi, 2*np.pi)
ax_th.set_xlabel('time t'); ax_th.set_ylabel('θ (internal coordinate)')
ax_th.set_title('Internal coordinate θ(t)\n(winds around compact dimension)', fontsize=11)
ax_th.grid(True, alpha=0.3)
ax_th.axhline(0,    color='gray', lw=0.5, ls='--')
ax_th.axhline( np.pi, color='gray', lw=0.5, ls=':')
ax_th.axhline(-np.pi, color='gray', lw=0.5, ls=':')

# Pick q=1.0 for the animation
q_anim = 1.0
sol_anim = solutions[2]   # q=1.0

# Also reconstruct theta(t):
# theta_dot = q/R = constant, so theta(t) = (q/R)*t
theta_t = (q_anim / R) * t_eval

lines_trail  = [ax_2d.plot([], [], color, lw=1.5, alpha=0.4, ls=ls)[0]
                for color, ls in zip(colors, linestyles)]
dots         = [ax_2d.plot([], [], 'o', color=color, ms=7)[0] for color in colors]
line_th,     = ax_th.plot([], [], '#2a78d6', lw=2)
dot_th,      = ax_th.plot([], [], 'o', color='#2a78d6', ms=8)
time_label   = ax_2d.text(0.05, 0.93, '', transform=ax_2d.transAxes, fontsize=10)
theta_label  = ax_th.text(0.05, 0.88, '', transform=ax_th.transAxes, fontsize=10)

SKIP = 4   # animate every SKIP-th frame for speed

def init():
    for l in lines_trail + dots: l.set_data([], [])
    line_th.set_data([], [])
    dot_th.set_data([], [])
    return lines_trail + dots + [line_th, dot_th, time_label, theta_label]

def update(frame):
    i = frame * SKIP
    if i >= len(t_eval): i = len(t_eval)-1
    for j, (sol, lt, dot) in enumerate(zip(solutions, lines_trail, dots)):
        lt.set_data(sol.y[0, :i], sol.y[1, :i])
        dot.set_data([sol.y[0, i]], [sol.y[1, i]])
    line_th.set_data(t_eval[:i], theta_t[:i])
    dot_th.set_data([t_eval[i]], [theta_t[i] % (2*np.pi) - np.pi])
    time_label.set_text(f't = {t_eval[i]:.2f}')
    theta_label.set_text(f'θ = {theta_t[i]:.2f} rad')
    return lines_trail + dots + [line_th, dot_th, time_label, theta_label]

n_frames = len(t_eval) // SKIP
ani = animation.FuncAnimation(fig2, update, frames=n_frames,
                               init_func=init, interval=30, blit=True)

plt.suptitle('Animation: free geodesic in (x, y, θ) space\n'
             'Left: D-dim projection  |  Right: internal winding', fontsize=11)
plt.tight_layout()
ani.save('kk_particle.gif', writer='pillow', fps=30)
plt.show()
print("Saved kk_trajectories.png and kk_particle.gif")
