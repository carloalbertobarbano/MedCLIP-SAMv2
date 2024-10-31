"""
Author: Carlo Alberto Barbano <carlo.barbano@unito.it>
Date: 31/10/24
"""
import argparse
import os
import uuid

import kornia.augmentation as kaugs
import torch
import torch.nn.functional as F
import torch.utils.tensorboard
import torchvision
import torchvision.datasets
import wandb
from transformers import AutoModel, AutoProcessor, AutoTokenizer
from torch.nn.utils import clip_grad_norm_
from transformers import AutoTokenizer

import sys
sys.path.append("../")
from saliency_maps.model.modeling_biomed_clip import BiomedCLIPModel
from helpers.transforms import Scale, Jitter, Normalize, ColorJitter, TotalVariation
from helpers.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description='CLIP inversion',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Misc
    parser.add_argument('--device', type=str, help='torch device', default='cuda')
    parser.add_argument('--save_dir', type=str, help='output dir', default='output')
    parser.add_argument('--save_every', default=100, type=int)
    parser.add_argument('--print_every', default=50, type=int)
    parser.add_argument('--trial', default=0, type=int, help='random seed')
    parser.add_argument('--uuid', action='store_true', help='use uuid for image file')

    # Model
    parser.add_argument('--model', type=str, help='model name', default='BiomedCLIP')

    # Input
    parser.add_argument('-p', '--prompt', nargs='+', type=str, default=[])
    parser.add_argument('--class_name', type=str, nargs="+",
                        help='(optional) folder name to use instead of prompt value', default=None)
    parser.add_argument('--caption_file', type=str, help='path to caption txt file', default=None)
    parser.add_argument('--caption_index', type=int, help='index of caption to use', default=None)
    parser.add_argument('--size', default=64, type=int)

    # Optim
    parser.add_argument('--optimizer', default='adam')
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--steps', default=3000, type=int)
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--l1', help='weight of l1 regularization', type=float, default=0.)
    parser.add_argument('--tv', help='weight of total variation regularization', default=0.005, type=float)
    parser.add_argument('--lr_annealing', action='store_true', help='enable lr annealing (cosine)')
    parser.add_argument('--upsize_schedule', type=int, nargs='+', default=[900, 1800])

    # Augmentation
    parser.add_argument('--jitter', action='store_true', help='use jitter')
    parser.add_argument('--jitter_lim', default=32, type=int)
    parser.add_argument('--cg_std', help='color jitter std', type=float, default=0.)
    parser.add_argument('--cg_mean', help='color jitter mean', type=float, default=0.)

    args = parser.parse_args()

    if args.caption_file:
        with open(args.caption_file, 'r') as f:
            captions = f.readlines()
        args.class_name, args.prompt = captions[args.caption_index].split('|')

    else:
        args.prompt = ' '.join(args.prompt)
        args.class_name = ' '.join(args.class_name).lower() if args.class_name else None

    print(f'class_name: <{args.class_name}>, prompt: <{args.prompt}>')

    if args.uuid:
        args.uuid = str(uuid.uuid4()) + "_"
    else:
        args.uuid = ''

    return args


def get_optimizer(params, opts):
    if opts.optimizer == 'adam':
        optimizer = torch.optim.Adam(params, lr=opts.lr)
    elif opts.optimizer == 'lbfgs':
        optimizer = torch.optim.LBFGS(params, lr=opts.lr)
    else:
        raise ValueError(f'Unknown optimizer: {opts.optimizer}')

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=(opts.steps - opts.upsize_schedule[-1]) // 3)

    return optimizer, scheduler


@torch.no_grad()
def get_text_embeddings(prompt, tokenizer, model: BiomedCLIPModel, opts):
    x_text = torch.tensor(tokenizer.encode(prompt, add_special_tokens=True)).to(opts.device)
    print("x_text:", x_text)
    text_features = model.get_text_features(x_text.unsqueeze(0))
    print("text features:", text_features.shape)
    return text_features


def main(opts):
    set_seed(opts.trial)

    run_name = (f"biomedclip_{opts.model.replace('/', '')}_lr{opts.lr}_l1{opts.l1}_tv{opts.tv}_"
                f"j{opts.jitter}_cg{opts.cg_mean}_{opts.cg_std}_"
                f"st{opts.steps}_{opts.trial}")
    output_folder = opts.class_name if opts.class_name else opts.prompt
    save_path = os.path.join(opts.save_dir, f"images/{run_name}/{output_folder}")
    logs_dir = os.path.join(opts.save_dir, f"logs/{run_name}/{output_folder}")
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    if opts.model == 'BiomedCLIP':
        model: BiomedCLIPModel = AutoModel.from_pretrained("../saliency_maps/model", trust_remote_code=True).to(opts.device)
        # processor = AutoProcessor.from_pretrained("chuhac/BiomedCLIP-vit-bert-hf", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained("chuhac/BiomedCLIP-vit-bert-hf", trust_remote_code=True)
    else:
        raise ValueError(f'Unknown model: {opts.model}')

    print("Model:", model)

    z_text = get_text_embeddings(opts.prompt, tokenizer, model, opts)
    print(f'z_text: {z_text.shape}')

    transform = []
    if opts.jitter:
        jitter = Jitter(opts.jitter_lim)
        transform.append(jitter)

    transform.extend([
        kaugs.RandomAffine(30, (0.1, 0.1),
                           (0.7, 1.2), p=0.5,
                           padding_mode='border',
                           same_on_batch=False),
        Scale(224, mode='bicubic'),
        ColorJitter(1, True,
                    mean=opts.cg_mean, std=opts.cg_std,
                    device=opts.device),
        Normalize([0.48145466, 0.4578275, 0.40821073],
                  [0.26862954, 0.26130258, 0.27577711]).to(opts.device)
    ])

    transform = kaugs.AugmentationSequential(*transform)
    # transform = NViewTransform(transform, opts.batch_size)

    wandb.init(project="multimodal-transfer", config=opts, name=opts.prompt,
               sync_tensorboard=True)
    writer = torch.utils.tensorboard.writer.SummaryWriter(logs_dir)

    torch.autograd.set_detect_anomaly(True)
    image = torch.rand((opts.batch_size, 3, opts.size, opts.size), device=opts.device)
    image.requires_grad_(True)
    optimizer, scheduler = get_optimizer([image], opts)

    max_grad_norm = 1.
    total_variation = TotalVariation(p=2)

    for step in range(1, opts.steps + 1):
        if step in opts.upsize_schedule:
            print("=> Upsizing")
            opts.size = opts.size * 2

            if opts.jitter:
                opts.jitter_lim = opts.jitter_lim * 2
            if opts.size >= 224:
                opts.size = 224

            image = F.interpolate(image.detach(), size=(opts.size, opts.size), mode='bicubic')
            image.requires_grad_(True)
            optimizer, scheduler = get_optimizer([image], opts)

        optimizer.zero_grad()

        def closure():
            input_image = transform(image)
            z_image = model.get_image_features(input_image)

            inversion_loss = torch.norm(z_image - z_text, p=2, dim=1).mean()
            l1_loss = torch.norm(image, p=1)
            tv_loss = total_variation(image)

            loss = inversion_loss + opts.l1 * l1_loss + opts.tv * tv_loss
            loss.backward()

            clip_grad_norm_([image], max_grad_norm)
            image.data = torch.clip(image.data, 0, 1)

            writer.add_scalar("lr", optimizer.param_groups[0]['lr'], step)
            writer.add_scalar('loss/inversion', inversion_loss.item(), step)
            writer.add_scalar('loss/l1', l1_loss.item(), step)
            writer.add_scalar('loss/tv', tv_loss.item(), step)

            if step % opts.print_every == 0:
                print(f'Step [{step}/{opts.steps}], inv loss: {loss.item():.4f}, '
                      f'l1: {l1_loss.item():.4f}, tv: {tv_loss.item():.4f}')

            return loss

        optimizer.step(closure)

        if opts.lr_annealing:
            scheduler.step()

        if step % opts.save_every == 0 or step == opts.steps or step == 1:
            writer.add_images('inverted', image, step)

            if step == opts.steps:
                for i in range(opts.batch_size):
                    path = os.path.join(save_path, f'{opts.uuid}_{i}.png')
                    torchvision.utils.save_image(image[i], path, normalize=True, scale_each=True)
                    print(f'Saved image at {path}')

                    with open(os.path.join(save_path, 'images.txt'), 'a') as f:
                        f.write(f"{path}|{opts.prompt}\n")


if __name__ == '__main__':
    main(parse_args())