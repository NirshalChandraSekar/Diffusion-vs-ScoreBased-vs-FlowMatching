import os
import numpy as np
import matplotlib.pyplot as plt

timesteps = 1000
beta1 = 0.1
beta2 = 50.0
dt = 1.0 / timesteps
means = np.array([1.0, -1.0])
stds = np.array([0.2, 0.2])
weights = np.array([0.1, 0.1])
weights /= np.sum(weights)
n_samples = 50
x_min = -4
x_max = 4
x_grid = np.linspace(x_min, x_max, num=1000)

def get_beta_t(t):
    ratio = float(t) / timesteps
    return ratio * beta2 + (1 - ratio) * beta1

def f(x, t):
    beta_t = get_beta_t(t)
    return -0.5 * beta_t * x

def g(t):
    beta_t = get_beta_t(t)
    return np.sqrt(beta_t)

def gaussian_pdf(x, mean, std):
    return (1.0 / (np.sqrt(2 * np.pi) * std)) * np.exp(-0.5 * ((x - mean) / std) ** 2)

def mixture_pdf(x):
    pdf = np.zeros_like(x)
    for i in range(len(means)):
        pdf += weights[i] * gaussian_pdf(x, means[i], stds[i])
    return pdf

def sample_mixture_gaussian(n_samples):
    components = np.random.choice(len(means), size=n_samples, p=weights / np.sum(weights))
    samples = np.random.normal(loc=means[components], scale=stds[components])
    return samples

def p_xt(x_t, t):
    ratio = float(t) / timesteps
    gamma = beta1 * ratio + (beta2 - beta1) * (ratio ** 2 / 2.0)
    alpha = np.exp(-gamma)
    sqrt_alpha = np.sqrt(alpha)
    p_xt_val = np.zeros_like(x_t)
    for i in range(len(means)):
        mean_t = sqrt_alpha * means[i]
        var_t = alpha * stds[i]**2 + (1 - alpha)
        std_t = np.sqrt(var_t)
        p_xt_val += weights[i] * gaussian_pdf(x_t, mean_t, std_t)
    return p_xt_val

def grad_log_p_xt(x_t, t):
    ratio = float(t) / timesteps
    gamma = beta1 * ratio + (beta2 - beta1) * (ratio ** 2 / 2.0)
    alpha = np.exp(-gamma)
    sqrt_alpha = np.sqrt(alpha)
    p_xt_val = p_xt(x_t, t)
    grad = np.zeros_like(x_t)
    for i in range(len(means)):
        mean_t = sqrt_alpha * means[i]
        var_t = alpha * stds[i]**2 + (1 - alpha)
        std_t = np.sqrt(var_t)
        pdf_i = weights[i] * gaussian_pdf(x_t, mean_t, std_t)
        pi_i = pdf_i / (p_xt_val + 1e-8)
        grad += pi_i * (-(x_t - mean_t) / var_t)
    return grad

def forward_sde(timesteps, n_samples, dt):
    x = np.zeros((timesteps, n_samples))
    x_pdf = np.zeros((timesteps, x_grid.shape[0]))

    x0 = sample_mixture_gaussian(n_samples)
    x[0] = x0
    
    x0_pdf = mixture_pdf(x_grid)
    x_pdf[0] = x0_pdf
    for t in range(1, timesteps):
        ############################
        # TODO: Implement the forward SDE
        ############################
        x_pdf[t] = p_xt(x_grid, t)

    return x, x_pdf

def reverse_sde(timesteps, n_samples, dt):
    x = np.zeros((timesteps, n_samples))
    x_pdf = np.zeros((timesteps, x_grid.shape[0]))
    
    xT = np.random.normal(0, 1, size=n_samples)
    x[-1] = xT
    
    x0_pdf = mixture_pdf(x_grid)
    x_pdf[0] = x0_pdf
    for t in range(timesteps - 1, 0, -1):
        ############################
        # TODO: Implement the reverse SDE
        ############################
        x_pdf[t-1] = p_xt(x_grid, t-1)

    return x, x_pdf

forward_x, forward_x_pdf = forward_sde(timesteps, n_samples, dt)
reverse_x, reverse_x_pdf = reverse_sde(timesteps, n_samples, dt)

fig, axes = plt.subplots(1, 2, figsize=(36, 12))

for i in range(n_samples):
    axes[0].plot(forward_x[:, i], lw=1)

time = np.arange(timesteps)
X, Y = np.meshgrid(time, x_grid)
pcm = axes[0].pcolormesh(X, Y, forward_x_pdf.T,
                         cmap='viridis', shading='auto', vmin=0.0, vmax=forward_x_pdf.max())
axes[0].set_title('Forward SDE')
axes[0].set_xlabel('Timesteps')
axes[0].set_ylabel('x')
axes[0].set_ylim([x_min, x_max])

for i in range(n_samples):
    axes[1].plot(reverse_x[::-1][:, i], lw=1)

X, Y = np.meshgrid(time, x_grid)
pcm = axes[1].pcolormesh(X, Y, reverse_x_pdf[::-1].T,
                         cmap='viridis', shading='auto', vmin=0.0, vmax=forward_x_pdf.max())
axes[1].set_title('Reverse SDE')
axes[1].set_xlabel('Timesteps')
axes[1].set_ylabel('x')

save_dir = "./step2_results"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, f"SDE.png")
plt.tight_layout()
plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
plt.close()