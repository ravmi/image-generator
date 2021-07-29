import pickle
import os
import torch
import numpy as np
import PIL.Image

def generate_images(
        count,#number of images
        pretrained_path,#path to pretrained model
        unfrozen_dims,#how many dimentions of latentt spaces will have equal values for all images
        save_images=True,#save images as pngs?
        outdir='generated_images',
        device='cpu',
        seed=0,
        class_idx=None#cifar is cgan and requires additional number (class id)
        ):
    dev = torch.device(device)
    with open(pretrained_path, 'rb') as f:
        G = pickle.load(f)['G_ema'].to(dev)
    print(f'Latent space size of this model: {G.z_dim}')
    print(f'Numer of possible classes of this model: {G.c_dim}')
    if unfrozen_dims > G.z_dim:
        print(f'Too many unfrozen dims: {unfrozen_dims} > {G.z_dim}')

    np.random.seed(seed)
    os.makedirs(outdir, exist_ok=True)

    #print(dir(G))
    z_frozen = torch.from_numpy(np.random.randn(1, G.z_dim - unfrozen_dims)).to(dev)
    for i in range(count):
        print('Generating image (%d/%d) ...' % (i+1, count))

        z_unfrozen = torch.from_numpy(np.random.randn(1, unfrozen_dims)).to(dev)
        z = torch.cat((z_frozen, z_unfrozen), 1)

        label = torch.zeros([1, G.c_dim], device=device)
        if G.c_dim != 0:
            label[:, class_idx] = 1


        if device=='cpu':
            img = G(z, label, force_fp32=True)
        else:
            img = G(z, label)



        if save_images:
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/{i}.png')

generate_images(
    count=20,
    pretrained_path='./pretrained/ffhq.pkl',
    unfrozen_dims=50,
    seed=1000)
'''
generate_images(
    count=20,
    pretrained_path='./pretrained/metfaces.pkl',
    unfrozen_dims=5,
    seed=0)

generate_images(
    count=20,
    pretrained_path='./pretrained/afhqcat.pkl',
    unfrozen_dims=5,
    seed=0,
    c=None)
generate_images(
    count=20,
    pretrained_path='./pretrained/cifar10.pkl',
    unfrozen_dims=100,
    seed=2,
    class_idx=1)
'''
