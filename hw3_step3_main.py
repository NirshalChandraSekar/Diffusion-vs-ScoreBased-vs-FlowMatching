import os
import torch
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt
from tqdm import tqdm

import torchdiffeq
from step3_utils.cfm_model import UNet

loss_history = []
def train_main_loop(train_loader, n_epochs, optimizer, FM, model, device, save_dir="./step3_results"):
  for epoch in range(n_epochs):
      print(f'epoch {epoch+1}')
      pbar = tqdm(train_loader)
      for i, data in enumerate(pbar):
          optimizer.zero_grad()
          x1 = data[0].to(device)
          x0 = torch.randn_like(x1)
          t, xt, cond_vec_field = FM.sample_location_and_conditional_flow(x0, x1)
          vt = model(xt, t)
          loss = (vt - cond_vec_field).pow(2).flatten(1).sum(1).mean() # TODO: Define the loss function given in HW description.
          loss_history.append(loss.item())
          loss.backward()
          optimizer.step()
          pbar.set_description(f"loss: {loss.item():.4f}")

      with torch.no_grad():
          traj = torchdiffeq.odeint(
              lambda t, x: model.forward(x, t),
              torch.randn(100, 1, 28, 28, device=device),
              torch.linspace(0, 1, 2, device=device),
              atol=1e-4,
              rtol=1e-4,
              method="dopri5",    # TODO: Choose your own ODE solver (see the torchdiffeq documentation on GitHub for available options).
          )

      grid = make_grid(
          traj[-1, :100].view([-1, 1, 28, 28]).clip(0, 1), value_range=(0, 1), padding=0, nrow=10
      )

      img = ToPILImage()(grid)
      plt.figure(figsize=(6, 6))
      plt.imshow(img)
      plt.axis("off")
      plt.tight_layout()

      os.makedirs(save_dir, exist_ok=True)
      save_path = os.path.join(save_dir, f"epoch_{epoch+1:02d}.png")
      plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
      plt.close()

def pad_t_like_x(t, x):
    if isinstance(t, (float, int)):
        return t
    return t.reshape(-1, *([1] * (x.dim() - 1)))

class ConditionalFlowMatcher:
    def __init__(self, sigma = 0.0):
        self.sigma = sigma

    def compute_mu_t(self, x1, t):
        ############################
        # TODO: Implement the function computing μ_t(x_1)
        ############################
        t_pad = pad_t_like_x(t, x1)
        mu_t = t_pad * x1
        return mu_t

    def compute_sigma_t(self, t):
        ############################
        # TODO: Implement the function computing σ_t
        ############################
        sigma_t = (1.0 - t) + t * self.sigma 
        return sigma_t

    def compute_conditional_velocity_field(self, x1, t, xt):
        ############################
        # TODO: Implement the function computing the conditional velocity field v(x_t, t; x_1)
        ############################
        mu_t = self.compute_mu_t(x1, t)
        sigma_t = self.compute_sigma_t(t)
        sigma_t = pad_t_like_x(sigma_t, xt)
        sigma_dt = self.sigma - 1.0
        sigma_dt = pad_t_like_x(sigma_dt, xt)
        cond_vec_field = x1 + (sigma_dt / (sigma_t + 1e-8)) * (xt - mu_t)
        return cond_vec_field

    def sample_xt(self, x0, x1, t, epsilon):
        ############################
        # TODO: Implement the function drawing a sample from the probability path N(x; μ_t(x1), σ_t^2 I)
        ############################
        mu_t = self.compute_mu_t(x1, t)
        sigma_t = self.compute_sigma_t(t)
        sigma_t = pad_t_like_x(sigma_t, x1)
        xt = mu_t + sigma_t * epsilon
        return xt

    def sample_noise_like(self, x):
        return torch.randn_like(x)

    def sample_location_and_conditional_flow(self, x0, x1):
        t = torch.rand(x0.shape[0]).type_as(x0)
        assert len(t) == x0.shape[0], "t has to have batch size dimension"

        eps = self.sample_noise_like(x0)
        xt = self.sample_xt(x0, x1, t, eps)
        cond_vec_field = self.compute_conditional_velocity_field(x1, t, xt)
        return t, xt, cond_vec_field

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    batch_size = 256
    sigma = 0.0
    n_feat = 128
    learning_rate = 5e-5
    num_epochs = 30   # TODO: Set an appropriate number of epochs. 30–50 should give you reasonable results

    tensor_transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root = "./data",	train = True, download = True,	transform = tensor_transform)    
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)    

    model = UNet(in_channels=1, n_feat=n_feat, n_classes=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    FM = ConditionalFlowMatcher(sigma=sigma)

    train_main_loop(train_loader, num_epochs, optimizer, FM, model, device)

    # plot loss history
    plt.plot(loss_history)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss History")
    plt.savefig("./step3_results/loss_history.png")

if __name__ == "__main__":
    main()