import matplotlib.pyplot as plt

import os
import torch

def build_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def plot_loss(history, result_dir):
    fig = plt.figure(figsize=(12,4))
    fig = plt.plot(history['loss'])
    _ = plt.gca().set(xlabel='Epochs', ylabel='Loss', title='Training Loss')
    plt.savefig(f'{result_dir}/loss.png')
    plt.close()

def plot_samples(samples, epoch, result_dir):
    num_rows, num_cols = 10, 4
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(8,16))
    for row in range(num_rows):
        for col in range(num_cols):
            axs[row][col].imshow((samples[row*num_cols + col]+1)/2)
            axs[row][col].set(xticks=[], yticks=[]) 
    fig.suptitle(f"Epoch : {epoch}")
    plt.savefig(f"{result_dir}/samples.png")
    plt.close()

def save_models(model, result_dir, model_name):
    torch.save(model.state_dict(), f"{result_dir}/{model_name}.pt")
    return