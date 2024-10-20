import os
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
from utils import *
from modules import UNet
import logging
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

class Diffusion:
    def __init__(self, noise_steps=1000, beta_start = 1e-4, beta_end = 0.02, img_size = 64, device = "cuda"):
        
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device
        
        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        
    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
    
    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        eps = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * eps, eps
    
    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))
    
    def sample(self, model, n):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x
    

def train(args):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args)
    model = UNet().to(device)
    print(f"Les paramètres du model sont: {model.parameters()}")
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)
    logger = SummaryWriter(os.path.join(script_dir, "runs", args.run_name))
    l = len(dataloader)
    
    # on vérifie si un modele existe déjà
    path = r"D:\github\Academic-Projects\Diffusion_models\DDPM\models\DDPM_Uncondtional\ckpt.pt"
    if os.path.isfile(path):
        ckpt = torch.load(path, weights_only=True)
        model.load_state_dict(ckpt)
        logging.info(f"Loading latest model...")
        
    # on vérifie combien d'epochs ont été faites avant
    path2 = r"D:\github\Academic-Projects\Diffusion_models\DDPM\results\DDPM_Uncondtional"
    if os.path.isdir(path2):
        fichiers = os.listdir(path2)
        nombres = [int(f.split('.')[0]) for f in fichiers]
        if nombres == []:
            epoch_before = 0
        else:
            epoch_before = max(nombres)
    else:
        epoch_before = 0
        

    for epoch in range(epoch_before + 1, args.epochs + epoch_before + 1):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (images, _) in enumerate(pbar):
            images = images.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            predicted_noise = model(x_t, t)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        sampled_images = diffusion.sample(model, n=1)
        save_images(sampled_images, os.path.join(script_dir, "results", args.run_name, f"{epoch}.jpg"))
        torch.save(model.state_dict(), os.path.join(script_dir, "models", args.run_name, f"ckpt.pt"))
        
    

def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_Uncondtional"
    args.epochs = 100 #500
    args.batch_size = 8
    args.image_size = 64
    args.dataset_path = r"D:\data\Landscape_Pictures"
    args.device = "cuda"
    args.lr = 3e-4
    train(args)
    

if __name__ == '__main__':
    launch()
    # device = "cuda"
    # model = UNet().to(device)
    # ckpt = torch.load(r"D:\github\Academic-Projects\Diffusion_models\DDPM\models\DDPM_Uncondtional\ckpt.pt", weights_only=True)
    # model.load_state_dict(ckpt)
    # diffusion = Diffusion(img_size=64, device=device)
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # for _ in range(20):
    #     x = diffusion.sample(model, 1)
    #     y = int(torch.randint(low=1,high=100000,size=(1,)))
    #     save_images(x, os.path.join(script_dir, "sample", f"{y}.jpg"))
        
        
    # print(x.shape)
    # plt.figure(figsize=(32, 32))
    # plt.imshow(torch.cat([
    #     torch.cat([i for i in x.cpu()], dim=-1),
    # ], dim=-2).permute(1, 2, 0).cpu())
    # plt.show()
    
    