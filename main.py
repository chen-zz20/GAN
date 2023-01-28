import argparse
import os
from time import time

from tensorboardX import SummaryWriter

import torch
import torch.optim as optim
from pytorch_fid import fid_score

from dataload import Choose_Dataset
from model import get_generator, get_discriminator
from trainer import Trainer

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--interpolation', action='store_true')
    parser.add_argument('--collapse', action='store_true')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for mini-batch training and evaluating. Default: 64')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of training epoch. Default: 20')
    parser.add_argument('--latent_dim', default=16, type=int)
    parser.add_argument('--generator_hidden_dim', default=16, type=int)
    parser.add_argument('--discriminator_hidden_dim', default=16, type=int)
    parser.add_argument('--saving_epochs', type=int, default=5)
    parser.add_argument('-lr', '--learning_rate', default=0.0002, type=float)
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--mode', type=str, default='MNIST',
	help='Training mode to choose from [MNIST, CIFAR10, CIFAR100]. Default: MNIST')
    parser.add_argument('--name', type=str, default='test',
	help='Give the model a name. Default: test')
    parser.add_argument('--notes', default='test', type=str, help='Something to note')
    parser.add_argument('--data_dir', default='./data', type=str, help='The path of the data directory')
    parser.add_argument('--train_dir', default='./train', type=str, help='The path of the train model directory')
    parser.add_argument('--log_dir', default='./log', type=str, help='The path of the log directory')
    parser.add_argument('--ipt_dir', default='./ipt', type=str, help='The path of the interpolation directory')
    parser.add_argument('--clp_dir', default='./clp', type=str, help='The path of the collapse directory')
    args = parser.parse_args()

    config = 'z-{}_batch-{}_num-train-epochs-{}'.format(args.latent_dim, args.batch_size, args.num_epochs)
    args.train_dir = os.path.join(args.train_dir, config)
    args.log_dir = os.path.join(args.log_dir, config)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')

    dataset = Choose_Dataset(args.batch_size, args.data_dir, args.mode)
    netG = get_generator(args.latent_dim, args.generator_hidden_dim, args.mode, device=device)
    netD = get_discriminator(args.discriminator_hidden_dim, args.mode, device=device)
    tb_writer = SummaryWriter(args.log_dir)

    if args.do_train:
        optimG = optim.Adam(netG.parameters(), lr=args.learning_rate, betas=(args.beta1, 0.999))
        optimD = optim.Adam(netD.parameters(), lr=args.learning_rate/10, betas=(args.beta1, 0.999))
        trainer = Trainer(netG, netD, optimG, optimD, dataset, args.train_dir, tb_writer, args.notes, device)
        trainer.train(args.num_epochs, args.saving_epochs)
    
    load_train_path = os.path.join(args.train_dir, args.notes)
    netG.load(args.train_dir, args.notes)

    if args.interpolation: # 线性插值
        from torchvision.utils import make_grid
        from torchvision.utils import save_image
        import numpy as np
        cnt = 0
        os.makedirs(args.ipt_dir, exist_ok=True)
        path = os.path.join(args.ipt_dir, 'number.png')
        z = [0, 0]
        while cnt < 2:
            # 随机产生一个隐变量输入，希望输出是我们预期的数字，这个目前只能人眼判断
            fixed_noise = torch.randn(1, args.latent_dim, 1, 1, device=device)
            imgs = make_grid(netG(fixed_noise)) * 0.5 + 0.5
            save_image(imgs, path)
            right = input("Is it the target? If 'no', juest print 'enter'; else print 'yes'!\n")
            if right:
                cnt += 1
                z[cnt-1] = fixed_noise.cpu().numpy()
        fixed_noise = torch.tensor(np.array([z[0]*i/9 + z[1]*(1-i/9) for i in range(10)]))
        fixed_noise = fixed_noise.reshape((10, args.latent_dim, 1, 1))
        imgs = make_grid(netG(fixed_noise)) * 0.5 + 0.5
        number = input("The generated number is: ")
        path = os.path.join(args.ipt_dir, f"{number}.png")
        save_image(imgs, path)
    
    if args.collapse: # 模式崩溃
        from torchvision.utils import make_grid
        from torchvision.utils import save_image

        os.makedirs(args.clp_dir, exist_ok=True)
        path = os.path.join(args.clp_dir, 'collapse.png')
        fixed_noise = torch.randn(50, args.latent_dim, 1, 1, device=device)
        imgs = make_grid(netG(fixed_noise)) * 0.5 + 0.5
        save_image(imgs, path)
    
    num_samples = 3000
    real_imgs = None
    real_dl = iter(dataset.training_loader)
    while real_imgs is None or real_imgs.size(0) < num_samples:
        imgs = next(real_dl)
        if real_imgs is None:
            real_imgs = imgs[0]
        else:
            real_imgs = torch.cat((real_imgs, imgs[0]), 0)
    real_imgs = real_imgs[:num_samples].expand(-1, 3, -1, -1) * 0.5 + 0.5

    with torch.no_grad():
        samples = None
        while samples is None or samples.size(0) < num_samples:
            imgs = netG.forward(torch.randn(args.batch_size, netG.latent_dim, 1, 1, device=device))
            if samples is None:
                samples = imgs
            else:
                samples = torch.cat((samples, imgs), 0)
    samples = samples[:num_samples].expand(-1, 3, -1, -1) * 0.5 + 0.5
    samples = samples.cpu()

    fid = fid_score.calculate_fid_given_images(real_imgs, samples, args.batch_size, device)
    tb_writer.add_scalar('fid', fid)
    print("FID score: {:.3f}".format(fid), flush=True)