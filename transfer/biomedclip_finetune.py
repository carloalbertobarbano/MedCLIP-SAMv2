"""
Author: Carlo Alberto Barbano <carlo.barbano@unito.it>
Date: 31/10/24
"""
import argparse
import torch
import torch.nn.functional as F
import os
import json
import copy
import torchvision.datasets

from torchvision.datasets import ImageFolder
from torch.utils.data import Subset
from transformers import AutoModel, AutoProcessor, AutoTokenizer

import sys
sys.path.append("../")
from saliency_maps.model.modeling_biomed_clip import BiomedCLIPModel
from helpers.transforms import Scale, Jitter, Normalize, ColorJitter, TotalVariation
from helpers.utils import set_seed
from biomedclip_invert import get_text_embeddings


def parse_args():
    parser = argparse.ArgumentParser(description='MedCLIP finetuning (single class)',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Misc
    parser.add_argument('--device', type=str, help='torch device', default='cuda')
    parser.add_argument('--save_dir', type=str, help='output dir', default='output')
    parser.add_argument('--print_every', default=1, type=int)
    parser.add_argument('--trial', default=0, type=int)

    # Data
    parser.add_argument('--data_dir', type=str, help='data directory', default='/scratch/data')
    parser.add_argument('--target_class', type=str, help='class name', default=None, required=True)
    parser.add_argument('--num_images', type=int, help='max number of images per class (None=unlimited)',
                        default=None)
    parser.add_argument('--caption_file', type=str, default="captions.json")

    # Model
    parser.add_argument('--model', type=str, help='model name', default='BiomedCLIP')

    # Training
    parser.add_argument('--definition_caption', action='store_true',
                        help='prepend each caption with "A <xxx> is <caption>"')
    parser.add_argument('--pl', action='store_true', help='plural definition')
    parser.add_argument('--batch_size', type=int, help='batch size', default=4)
    parser.add_argument('--lr', type=float, help='learning rate', default=1e-5)
    parser.add_argument('--weight_decay', type=float, help='weight decay', default=0.2)
    parser.add_argument('--temperature', type=float, help='InfoNCE temperature', default=0.07)
    parser.add_argument('--epochs', type=int, help='number of epochs', default=1)
    parser.add_argument('--noise', action='store_true', help='use noise image instead of inverted')
    parser.add_argument('--kd_weight', type=float, default=0, help='weight of knowledge distillation')
    args = parser.parse_args()
    return args


def infonce(image_features, text_features, args):
    # InfoNCE loss
    bsz = image_features.shape[0]
    mask = torch.eye(bsz, device=args.device).repeat(2, 2)
    inv_diagonal = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(bsz * 2, device=args.device).view(-1, 1),
        0
    )

    features = torch.cat([image_features, text_features], dim=0)
    anchor_dot_contrast = torch.div(
        torch.matmul(features, features.T),
        args.temperature
    )

    # Numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    alignment = logits
    uniformity = torch.exp(logits) * inv_diagonal
    uniformity = torch.log(uniformity.sum(1, keepdim=True))

    positive_mask = mask * inv_diagonal
    log_prob = alignment - uniformity
    log_prob = (positive_mask * log_prob).sum(1) / positive_mask.sum(1)
    loss = - (args.temperature / 0.07) * log_prob
    return loss.mean()


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parse_args()
    set_seed(args.trial)

    ft_method = "single"
    if args.noise:
        ft_method = "noise"

    save_dir = os.path.join(args.save_dir,
                            f"medclip_{args.model.replace('/', '_')}_finetune_{ft_method}_"
                            f"{args.target_class}"
                            f"{'_def' if args.definition_caption else ''}"
                            f"_bsz{args.batch_size}_lr{args.lr}_wd{args.weight_decay}_t{args.temperature}_"
                            f"kd{args.kd_weight}_"
                            f"{args.epochs}_s{args.trial}")
    os.makedirs(save_dir, exist_ok=True)

    if args.model == 'BiomedCLIP':
        model: BiomedCLIPModel = AutoModel.from_pretrained("../saliency_maps/model", trust_remote_code=True).to(
            args.device)
        processor = AutoProcessor.from_pretrained("chuhac/BiomedCLIP-vit-bert-hf", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained("chuhac/BiomedCLIP-vit-bert-hf", trust_remote_code=True)
    else:
        raise ValueError(f'Unknown model: {args.model}')

    teacher = copy.deepcopy(model)

    if not args.noise:
        dataset = ImageFolder(root=args.data_dir, transform=None)
        idx = [i for i in range(len(dataset)) if dataset.targets[i] == dataset.class_to_idx[args.target_class]]
        dataset = Subset(dataset, idx)
        print("Loaded", len(dataset), "images")

        if args.num_images is not None:
            sub_idx = list(range(args.num_images))
            dataset = Subset(dataset, sub_idx)
            print("Subset to", len(dataset), "images")
    else:
        # Create mock dataset with noise images
        dataset = torchvision.datasets.FakeData(size=10, transform=None, image_size=(3, 224, 224))
        print("Created noise dataset")

    def collate_fn(data):
        images, labels = zip(*data)
        labels = torch.tensor(labels)
        return images, labels

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8,
                                             collate_fn=collate_fn)

    with open(args.caption_file, "r") as f:
        captions = json.load(f)

    print(f"Loaded {len(captions)} captions")
    target_caption = captions[args.target_class]
    if args.definition_caption:
        print("Using definition captions")
        if args.pl:
            target_caption = f"{args.target_class} are {target_caption}"
        else:
            target_caption = f"A {args.target_class} is {target_caption}"
    print("Target caption:", target_caption)
    input_ids = torch.tensor(tokenizer.encode(target_caption, add_special_tokens=True)).unsqueeze(0).to(args.device)
    print("input_ids:", input_ids)

    optimizer = torch.optim.Adam(model.vision_model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                                 betas=(0.9, 0.98), eps=1e-6)  # https://github.com/openai/CLIP/issues/83
    model.train()
    teacher.eval()

    for epoch in range(1, args.epochs + 1):
        for idx, (img, _) in enumerate(dataloader):
            inputs = processor(images=img, return_tensors="pt")

            pixel_values = inputs['pixel_values'].to(args.device)

            optimizer.zero_grad()

            text_features = model.get_text_features(input_ids, attention_mask=None)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            image_features = model.get_image_features(pixel_values)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            loss = infonce(image_features, text_features.repeat(image_features.shape[0], 1), args)

            if args.kd_weight > 0:
                with torch.no_grad():
                    teacher_features = teacher.get_image_features(pixel_values)
                    teacher_features = teacher_features / teacher_features.norm(dim=-1, keepdim=True)
                loss = loss + args.kd_weight * F.mse_loss(image_features, teacher_features)

            loss.backward()
            optimizer.step()

            if (idx+1) % args.print_every == 0:
                print(f"Epoch {epoch} [batch {idx+1}/{len(dataloader)}], loss: {loss.item()}")

    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "args": args
    }, os.path.join(save_dir, f"weights.pth"))
    print("Saved weights to", f"{save_dir}/weights.pth")


if __name__ == '__main__':
    main()
