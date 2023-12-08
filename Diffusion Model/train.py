import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import PredictNoise
from dataloader import MnistDataset 
from util import plot_loss, plot_samples, save_models, build_dir

def train_step(model, x_batch, alpha_hat, loss_fn):
    global DEVICE
    model.train()
    batch_size = x_batch.size(0)
    x_batch = x_batch.to(DEVICE)
    t_batch = torch.randint(T, size=(batch_size,), device = DEVICE)
    eps_batch = torch.tensor(np.random.normal(size=x_batch.shape), 
                                    dtype=torch.float32, 
                                    device = DEVICE)
            
    alpha_batch = alpha_hat[t_batch].reshape(batch_size, x_batch.shape[1])
    alpha_batch = alpha_batch.unsqueeze(2).unsqueeze(3).expand(batch_size, 1, 28, 28)
    x_batch = torch.sqrt(alpha_batch)*x_batch + torch.sqrt(1-alpha_batch)*eps_batch
            
    pred_eps = model(x_batch, t_batch)
    loss = loss_fn(pred_eps, eps_batch)
    return loss

def get_samples(model, beta, alpha_hat, alpha, samples = 32):
    model.eval()
    global DEVICE

    with torch.no_grad():
        x_sample = torch.tensor(np.random.normal(size=(samples, 1, 28, 28)), 
                                        dtype=torch.float32, 
                                        device = DEVICE)
        for t in range(T-1, 0, -1):
            z = torch.tensor(np.random.normal(size=(samples, 1, 28, 28)), 
                                        dtype=torch.float32, 
                                        device = DEVICE)
            
            if t == 0 : z = 0 
            time_step = t
            t = torch.tensor(t, device = DEVICE).unsqueeze(0).expand(samples)
            alpha_batch = alpha_hat[t].reshape(samples, 1)
            alpha_batch = alpha_batch.unsqueeze(2).unsqueeze(3).expand(samples, 1, 28, 28)
            noise_pred = model(x_sample, t)
            x_sample = 1/torch.sqrt(alpha[time_step]) * (x_sample - ((1-alpha[time_step])/torch.sqrt(1-alpha_batch))*noise_pred)
            x_sample = x_sample + torch.sqrt(beta[time_step]) * z
        x_sample = x_sample.clamp(-1,1).cpu().numpy()
        x_sample = np.transpose(x_sample, (0, 2, 3, 1))
        return x_sample

def save_checkpoints(model, beta, alpha_hat, alpha, epoch, result_dir, model_name, samples = 4):
    samples = get_samples(model, beta, alpha_hat, alpha, samples)
    plot_samples(samples, epoch, result_dir)
    save_models(model, result_dir, model_name)
    plot_loss(history, result_dir)   
    return
    
if __name__ == "__main__":
    base_dir = 'data/mnist/'
    img_gzip = "train-images.idx3-ubyte"
    label_gzip = "train-labels.idx1-ubyte"

    hidden_channel = 256

    result_dir = "result_unconditional"
    model_name = f"model_{hidden_channel}.pt"

    build_dir(result_dir)

    batch_size = 32
    global DEVICE

    DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else DEVICE

    dataset = MnistDataset(img_gzip = img_gzip,
                        label_gzip = label_gzip,
                        base_dir = base_dir)

    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle=True)
    model = PredictNoise(in_channel = 1, out_channel = 1, 
                         hidden_channel = hidden_channel,
                         device = DEVICE).to(DEVICE)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-4)

    T = 1000
    beta_start, beta_end = (1e-4, 0.02)
    beta = torch.linspace(beta_start, beta_end, T, device = DEVICE)
    alpha = 1 - beta
    alpha_hat = torch.cumprod(alpha, dim = 0)

    epochs = 50
    history = {'loss':[]}

    for epoch in tqdm(range(epochs)):
        loss_ = []
        for idx, (x_batch, _) in enumerate(dataloader):
            loss = train_step(model, x_batch, alpha_hat, loss_fn)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_.append(loss.item())

        loss_ = np.mean(loss_)
        history['loss'].append(loss_)
        print(f"[{epoch+1}] Loss : {loss_:.4f}")

        if epoch % 5 == 0:
            save_checkpoints(model, beta, alpha_hat, alpha, epoch, result_dir, model_name, samples = 40)   

    save_checkpoints(model, beta, alpha_hat, alpha, epochs, result_dir, model_name, samples = 40)  
