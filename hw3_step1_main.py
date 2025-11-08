import torch
import os
from step1_utils.models.unet import create_model
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import step1_utils.utils as utils
import argparse
import numpy as np

class Config:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.conf = None
        
        # hyperparameters for path & dataset
        self.parser.add_argument('--out_path', type=str, default='step1_results', help='results file directory')
        self.parser.add_argument('--dataset', type=str, default='ffhq', help='either choose ffhq or imagenet')
        
        # hyperparameters for sampling
        self.parser.add_argument('--total_instances', type=int, default=1, help='number of images you want to generate - 10 is ideal')
        self.parser.add_argument('--diff_timesteps', type=int, default=1000, help='Original number of steps from Ho et al. (2020) which is 1000 - do not change')
        self.parser.add_argument('--desired_timesteps', type=int, default=100, help='How many steps do you want?')
        self.parser.add_argument('--eta', type=float, default=1.0, help='Should be between [0.0, 1.0]')
        self.parser.add_argument('--schedule', type=str, default="100", help="regular/irregular schedule to use (jumps)")

    def parse(self, args=None):
        """Parse the configuration"""
        self.conf = self.parser.parse_args(args=args)
        return self.conf

class Sampler():
    def __init__(self):
        self.conf = Config().parse()
        scale = 1000 / self.conf.diff_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        self.betas = torch.linspace(beta_start, beta_end, self.conf.diff_timesteps, dtype=torch.float64)
        self.alpha_init()
       
    def alpha_init(self):
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = torch.cat((torch.tensor([1.0]), self.alphas_cumprod[:-1]))
    
    # def recreate_alphas(self):
    #     use_timesteps = utils.space_timesteps(self.conf.diff_timesteps, self.conf.schedule) # Selects a subset of timesteps according to the given schedule
    #     self.timestep_map = []
    #     last_alpha_cumprod = 1.0
        
    #     # TODO: Initialize an empty list to store the new beta values --> This is to collect the new betas corresponding to the chosen timesteps.
    #     new_betas = []
    #     for i, alpha_cumprod in enumerate(self.alphas_cumprod):
            
    #         # TODO: Check if the current timestep index 'i' is part of the selected timesteps (use_timesteps)
    #         if i in use_timesteps:
    #             # TODO: If so, compute the corresponding beta value and append it to the empty list
    #             new_beta = 1 - (alpha_cumprod / last_alpha_cumprod)
    #             new_betas.append(new_beta)
    #             # TODO: Update 'last_alpha_cumprod' to the current 'alpha_cumprod'
    #             last_alpha_cumprod = alpha_cumprod
    #             # TODO: Keep track of which original timesteps are being used by appending 'i' to 'self.timestep_map'
    #             self.timestep_map.append(i)

    #     # TODO: Convert 'new_betas' into a PyTorch tensor and store it in 'self.betas'
    #     self.betas = torch.tensor(new_betas, dtype=torch.float64)
    #     # TODO: After updating betas, Recompute the related alpha terms to refresh alpha values
    #     # Hint: A helper function is already implemented in this hw3_step1_main.py file to refresh the alpha values
    #     # Understand which function does that and use it here.
    #     self.alpha_init()
    #     return torch.tensor(self.timestep_map)

    def recreate_alphas(self):
        kept_indices = utils.space_timesteps(self.conf.diff_timesteps, self.conf.schedule)

        if isinstance(kept_indices, set):
            kept_indices = sorted(list(kept_indices))
        elif hasattr(kept_indices, 'tolist'):
            kept_indices = kept_indices.tolist()
        else:
            kept_indices = list(kept_indices)
            
        self.timestep_map = kept_indices

        new_betas = []
        last_alpha_cumprod = torch.tensor(1.0, dtype=self.alphas.dtype, device=self.alphas.device)

        alpha_bar = self.alphas_cumprod

        for idx in kept_indices:
            curr_alpha_cumprod = alpha_bar[idx]
            beta = 1.0 - (curr_alpha_cumprod / last_alpha_cumprod)

            beta = torch.clamp(beta, min=1e-12, max=1 - 1e-12)

            new_betas.append(beta)
            last_alpha_cumprod = curr_alpha_cumprod

        self.betas = torch.stack(new_betas).to(dtype=self.alphas.dtype, device=self.alphas.device)

        self.alpha_init()

        return torch.tensor(self.timestep_map, dtype=torch.long)

    
    def get_variance(self, x, t):
        posterior_variance = (self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        posterior_log_variance_clipped = torch.log(torch.cat((posterior_variance[1].unsqueeze(0), posterior_variance[1:])))
        posterior_variance = (self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        posterior_log_variance_clipped = torch.log(torch.cat((posterior_variance[1].unsqueeze(0), posterior_variance[1:])))
        model_var_values = x
        min_log = posterior_log_variance_clipped
        max_log = torch.log(self.betas)
        min_log = utils.extract_and_expand(min_log, t, x)
        max_log = utils.extract_and_expand(max_log, t, x)
        frac = (model_var_values + 1.0) / 2.0
        model_log_variance = frac * max_log + (1-frac) * min_log
        return model_log_variance
    
    def predict_x0_hat(self, x_t, t, model_output):
        ############################
        # TODO: Implement the function predicting the clean denoised estimate x_{0|t} --> step (c and d)
        ############################

        C_t = 1 / torch.sqrt(self.alphas_cumprod)
        D_t = (torch.sqrt(1 - self.alphas_cumprod)) / (torch.sqrt(self.alphas_cumprod))

        C_t = utils.extract_and_expand(C_t, t, x_t)
        D_t = utils.extract_and_expand(D_t, t, x_t)

        x0_hat = (C_t * x_t) - (D_t * model_output)

        return x0_hat
    
    def sample_ddpm(self, score_model, x_t, model_t, t):
        with torch.no_grad():
            model_output = score_model(x_t, model_t)
        model_output, model_var_values = torch.split(model_output, x_t.shape[1], dim=1)
        model_log_variance = self.get_variance(model_var_values, t)
        
        ############################
        # TODO: Implement DDPM sampling --> step (b, c, d, e)
        # Hint: Think about step (a) which you are expected to find --> At · xt + Bt · ˆϵθ(xt, t) + σtz (z = 0 if t = 0)
        # Now after you calculate At, you should perform << utils.extract_and_expand(At, t, x_t) >> before running At * x_t
        # Otherwise, you will face some dimensionality errors
        # Also note that self.betas = β, self.alphas = α, self.alphas_cumprod = \bar{α} and self.alphas_cumprod_prev[t] = \bar{α_t-1}
        # Note that extract_and_expand() function already selects the index you input to the function (always input t)
        ############################

        # A_t = 1 / torch.sqrt(self.alphas)
        # B_t = - (1 - self.alphas) / (torch.sqrt(1-self.alphas_cumprod) * torch.sqrt(self.alphas))

        # A_t = utils.extract_and_expand(A_t, t, x_t)
        # B_t = utils.extract_and_expand(B_t, t, x_t)

        # mean = A_t * x_t + B_t * model_output

        x_0_hat = self.predict_x0_hat(x_t, t, model_output)

        A_t = (torch.sqrt(self.alphas)*(1 - self.alphas_cumprod_prev)) / (1 - self.alphas_cumprod)
        B_t = (torch.sqrt(self.alphas_cumprod_prev)*(1 - self.alphas)) / (1 - self.alphas_cumprod)

        A_t = utils.extract_and_expand(A_t, t, x_t)
        B_t = utils.extract_and_expand(B_t, t, x_t)

        mean = A_t * x_t + B_t * x_0_hat

        # σ_t (log variance → variance → stddev)
        sigma = torch.exp(0.5 * model_log_variance) 

        if(t == 0):
            z = torch.zeros_like(x_t)
        else:
            # create random noise in numpy and convert to torch to see if choosing random Z makes the model stochastic
            # z = torch.from_numpy(np.random.randn(*x_t.shape)).to(x_t.device).to(x_t.dtype)

            z = torch.randn_like(x_t)

        sample = mean + sigma * z

        # if(t == 999 or t == 750 or t == 500 or t == 250 or t==50 or t==0):
        #     #save the intermediate images
        #     x0_hat = self.predict_x0_hat(x_t, t, model_output)
        #     image_filename = f"intermediate_image_t{t.item()}.png"
        #     image_path = os.path.join(self.conf.out_path, "intermediate_images", image_filename)
        #     os.makedirs(os.path.dirname(image_path), exist_ok=True)
        #     plt.imsave(image_path, utils.clear_color(x0_hat))

        return sample

    def sample_ddim(self, score_model, x_t, model_t, t):
        with torch.no_grad():
            model_output = score_model(x_t, model_t)
        model_output, _ = torch.split(model_output, x_t.shape[1], dim=1)
        
        ############################
        # TODO: Implement DDIM sampling --> step (f)
        ############################

        x_0_hat = self.predict_x0_hat(x_t, t, model_output)

        alpha_bar_t = utils.extract_and_expand(self.alphas_cumprod, t, x_t)
        alpha_bar_prev = utils.extract_and_expand(self.alphas_cumprod_prev, t, x_t)

        eta = self.conf.eta

        sigma_t = eta * torch.sqrt(
            ((1 - alpha_bar_prev) / (1 - alpha_bar_t)) * (1 - (alpha_bar_t / alpha_bar_prev))
        )
        sigma_t = torch.clamp(sigma_t, min=0.0)

        coeff_eps = torch.sqrt(
            torch.clamp((1 - alpha_bar_prev) - sigma_t**2, min=0.0)
        )

        if (t == 0).all():
            z = torch.zeros_like(x_t)
        else:
            z = torch.randn_like(x_t)

        sample = torch.sqrt(alpha_bar_prev) * x_0_hat + coeff_eps * model_output + sigma_t * z
                
        return sample
    
def main():
    
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    conf = Config().parse()
    
    algo = "DDIM"
    if algo == "DDIM":
        print('*' * 60 + f'\nSTARTED {algo} Sampling with eta = \"%.1f\" \n' %conf.eta)
    if algo == "DDPM":
        print('*' * 60 + f'\nSTARTED {algo} Sampling')
        
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Create and config model
    model_config = utils.load_yaml("step1_utils/models/" + conf.dataset + "_model_config.yaml")
    score_model = create_model(**model_config).to(device).eval()

    # Sampling
    for instance in range(conf.total_instances): 
        sampler_operator = Sampler()
        x_t = utils.get_noise_x_t(instance, conf.total_instances, device)
        
        pbar = (list(range(conf.desired_timesteps))[::-1])
        
        if conf.desired_timesteps == 1000:
            time_map = torch.tensor(list(utils.space_timesteps(conf.diff_timesteps, "1000"))).to(device)
        else:
            time_map = sampler_operator.recreate_alphas().to(device)
        
        print(f"\n********* image {instance+1}/{conf.total_instances}: *********\n")
        for idx in tqdm(pbar):
            time = torch.tensor([idx] * x_t.shape[0], device=device)
            if algo == "DDPM":
                x_t_prev_bar = sampler_operator.sample_ddpm(score_model, x_t, time_map[time], time)
            if algo == "DDIM":
                x_t_prev_bar = sampler_operator.sample_ddim(score_model, x_t, time_map[time], time)
            x_t = x_t_prev_bar
        image_filename = f"image_{instance+1}.png"
        
        image_path = os.path.join(conf.out_path, image_filename)
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        plt.imsave(image_path, utils.clear_color(x_t))
    print('\nFINISHED Sampling!\n' + '*' * 60)

if __name__ == '__main__':
    main()