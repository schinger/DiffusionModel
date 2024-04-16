import argparse
import os
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.data import TensorDataset
from torchvision.datasets import MNIST
from torchvision import transforms

# animation
from celluloid import Camera
import imageio


def cartesian_heart_dataset(n=8000):
    rng = np.random.default_rng(42)
    theta = 2 * np.pi * rng.uniform(0, 1, n)
    x = (1 - np.sin(theta)) * np.cos(theta)
    y = (1 - np.sin(theta)) * np.sin(theta) + 0.9
    X = np.stack((x, y), axis=1)
    X *= 3
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))

def perfect_heart_dataset(n=8000):
    rng = np.random.default_rng(42)
    theta = 2 * np.pi * rng.uniform(0, 1, n)
    x = 16 * np.sin(theta)**3
    y = 13 * np.cos(theta) - 5 * np.cos(2*theta) - 2 * np.cos(3*theta) - np.cos(4*theta)
    X = np.stack((x, y), axis=1)
    X /= 5
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))

def mnist_dataset(train=True):
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = MNIST(
            "./data", download=True, train=train, transform=transform
        )
    data = train_dataset.data
    data = ((data / 255.0) * 2.0) - 1.0
    data = data.reshape(-1, 28*28)
    return data

def get_dataset(name, n=8000):
    if name == "heart":
        return perfect_heart_dataset(n)
    elif name == "mnist":
        return mnist_dataset()
    else:
        raise ValueError(f"Unknown dataset: {name}")

class Block(nn.Module):
    def __init__(self, size: int):
        super().__init__()

        self.ff = nn.Linear(size, size)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor):
        return x + self.act(self.ff(x))

class SinusoidalEmbedding(nn.Module):
    def __init__(self, size: int, scale: float = 1.0):
        super().__init__()
        self.size = size
        self.scale = scale
        half_size = self.size // 2
        emb = torch.log(torch.Tensor([10000.0])) / (half_size - 1)
        emb = torch.exp(-emb * torch.arange(half_size))
        self.emb = nn.Parameter(emb, requires_grad=False)

    def forward(self, x: torch.Tensor):
        x = x * self.scale
        emb = x * self.emb.unsqueeze(0)
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
        return emb

class MLP(nn.Module):
    def __init__(self, hidden_size: int = 128, hidden_layers: int = 3, emb_size: int = 128,
                 twoD_data: bool = True):
        super().__init__()
        self.twoD_data = twoD_data
        self.time_emb = SinusoidalEmbedding(emb_size)
        if twoD_data:
            self.input_emb1 = SinusoidalEmbedding(emb_size, scale=25.0)
            self.input_emb2 = SinusoidalEmbedding(emb_size, scale=25.0)
            self.concat_size = 2 * emb_size + emb_size # 2d concat time
            self.data_size = 2
        else:
            self.concat_size = 28 * 28 + emb_size # mnist is 28*28
            self.data_size = 28 * 28

        layers = [nn.Linear(self.concat_size, hidden_size), nn.GELU()]
        for _ in range(hidden_layers):
            layers.append(Block(hidden_size))
        layers.append(nn.Linear(hidden_size, self.data_size))
        self.joint_mlp = nn.Sequential(*layers)

    def forward(self, x, t):
        t_emb = self.time_emb(t.reshape(-1,1))
        if self.twoD_data:
            x1_emb = self.input_emb1(x[:, 0].reshape(-1,1))
            x2_emb = self.input_emb2(x[:, 1].reshape(-1,1))
            x = torch.cat((x1_emb, x2_emb), dim=-1)
        x = torch.cat((x, t_emb), dim=-1)
        x = self.joint_mlp(x)
        return x

class Diffusion():
    def __init__(self,
                 num_timesteps=1000,
                 beta_start=0.0001,
                 beta_end=0.02):

        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(
                beta_start, beta_end, num_timesteps, dtype=torch.float32)

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(
            self.alphas_cumprod[:-1], (1, 0), value=1.)

        # required for self.add_noise
        self.sqrt_alphas_cumprod = self.alphas_cumprod ** 0.5
        self.sqrt_one_minus_alphas_cumprod = (1 - self.alphas_cumprod) ** 0.5

        self.sqrt_inv_alphas = torch.sqrt(1. / self.alphas)
        self.noise_coef = self.betas / self.sqrt_one_minus_alphas_cumprod
        self.variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def add_noise(self, x_start, x_noise, timesteps):
        s1 = self.sqrt_alphas_cumprod[timesteps].reshape(-1, 1)
        s2 = self.sqrt_one_minus_alphas_cumprod[timesteps].reshape(-1, 1)
        return s1 * x_start + s2 * x_noise
    
    def sample_step(self, model_output, timestep, sample):
        s1 = self.sqrt_inv_alphas[timestep].reshape(-1, 1)
        s2 = self.noise_coef[timestep].reshape(-1, 1)
        s3 = self.variance[timestep].reshape(-1, 1) ** 0.5
        noise = torch.randn_like(model_output)
        return s1 * (sample - s2 * model_output) + s3 * noise

    def __len__(self):
        return self.num_timesteps


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="base")
    parser.add_argument("--dataset", type=str, default="heart", choices=["heart", "mnist"])
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=1000)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_timesteps", type=int, default=200)
    parser.add_argument("--embedding_size", type=int, default=32)
    parser.add_argument("--hidden_size", type=int, default=32)
    parser.add_argument("--hidden_layers", type=int, default=1)
    parser.add_argument("--show_image_step", type=int, default=1)
    parser.add_argument("--eval_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    config = parser.parse_args()
    outdir = f"exps/{config.experiment_name}"

    if config.device[:4] =="cuda":
        device = config.device
    else:
        device = config.device = "cpu"

    dataset = get_dataset(config.dataset)
    dataloader = DataLoader(
        dataset, batch_size=config.train_batch_size, shuffle=True, drop_last=True)

    model = MLP(
        hidden_size=config.hidden_size,
        hidden_layers=config.hidden_layers,
        emb_size=config.embedding_size,
        twoD_data=config.dataset!='mnist').to(device)

    diffusion = Diffusion(
        num_timesteps=config.num_timesteps)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 - step / (config.num_epochs * len(dataloader)))

    if config.eval_path is None:
        global_step = 0
        losses = []
        print("Training model...")
        for epoch in range(config.num_epochs):
            model.train()
            progress_bar = tqdm(total=len(dataloader))
            progress_bar.set_description(f"Epoch {epoch}")
            for step, batch in enumerate(dataloader):
                # print(batch.shape)
                # batch = batch[0]
                noise = torch.randn(batch.shape)
                timesteps = torch.randint(
                    0, diffusion.num_timesteps, (batch.shape[0],)
                ).long()

                noisy = diffusion.add_noise(batch, noise, timesteps)
                noise_pred = model(noisy.to(device), timesteps.to(device))
                loss = F.mse_loss(noise_pred, noise.to(device))
                loss.backward()

                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                progress_bar.update(1)
                logs = {"loss": loss.detach().item(), "step": global_step, "lr": scheduler.get_last_lr()[0]}
                losses.append(loss.detach().item())
                progress_bar.set_postfix(**logs)
                global_step += 1
            progress_bar.close()

        print("Saving model...")
        os.makedirs(outdir, exist_ok=True)
        torch.save(model.state_dict(), f"{outdir}/model.pth")
    else:
        model.load_state_dict(torch.load(config.eval_path))

    print("Evaluate/Save Animation")
    model.eval()
    sample = torch.randn(config.eval_batch_size, model.data_size)
    timesteps = list(range(config.num_timesteps))[::-1]
    reverse_samples = []
    reverse_samples.append(sample)
    for i, t in enumerate(tqdm(timesteps)):
        t = torch.from_numpy(np.repeat(t, config.eval_batch_size)).long()
        with torch.no_grad():
            residual = model(sample.to(device), t.to(device)).to("cpu")
        sample = diffusion.sample_step(residual, t[0], sample)
        if i % config.show_image_step == 0:
            reverse_samples.append(sample)
    n_hold_final = 30
    for _ in range(n_hold_final):
        reverse_samples.append(sample)

    if config.dataset != 'mnist':
        # also show forward samples
        dataset = get_dataset(config.dataset, n=config.eval_batch_size)
        x0 = dataset.tensors[0]
        forward_samples = []
        forward_samples.append(x0)
        for t in range(len(diffusion)):
            timesteps = np.repeat(t, len(x0))
            noise = torch.randn_like(x0)
            sample = diffusion.add_noise(x0, noise, timesteps)
            if i % config.show_image_step == 0:
                forward_samples.append(sample)

        xmin, xmax = -6, 6
        ymin, ymax = -6, 6
        fig, ax = plt.subplots()
        camera = Camera(fig)

        for i, sample in enumerate(forward_samples + reverse_samples):
            plt.scatter(sample[:, 0], sample[:, 1], alpha=0.5, s=15, color="red")
            steps = i if i < len(forward_samples) else i - len(forward_samples)
            ax.text(0.0, 0.95, f"step {steps+1: 4} / {config.num_timesteps}", transform=ax.transAxes)
            ax.text(0.0, 1.01, "Forward process" if i < len(forward_samples) else "Reverse process", transform=ax.transAxes, size=15)
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
            plt.axis('scaled')
            plt.axis("off")
            camera.snap()

        animation = camera.animate(blit=True, interval=200)
        animation.save(f"{outdir}/animation_2d.gif")

    else:
        # show reverse process only
        # dataset = get_dataset('mnist')
        # dataset = dataset[:10].unsqueeze(0).expand(20,-1,-1)
        # reverse_samples = dataset

        reverse_samples = torch.stack(reverse_samples, dim=0)
        reverse_samples = (reverse_samples.clamp(-1, 1) + 1) / 2
        reverse_samples = (reverse_samples * 255).type(torch.uint8)
        reverse_samples = reverse_samples.reshape(-1, config.eval_batch_size, 28, 28)
        reverse_samples = list(torch.split(reverse_samples, 1, dim=1))
        for i in range(len(reverse_samples)):
            reverse_samples[i] = reverse_samples[i].squeeze(1)
        reverse_samples = torch.cat(reverse_samples, dim=-1)
        print(len(reverse_samples))
        imageio.mimsave(
            f"{outdir}/animation_mnist.gif",
            list(reverse_samples),
            fps=5,
        )
