# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from pathlib import Path
import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch import nn, optim
import torch.distributed as dist
import torchvision.datasets as datasets

import augmentations as aug
from distributed import init_distributed_mode

import resnet

from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image


from resnet import resnet50
import cv2
import numpy as np
from torchvision.models import resnet50


def get_arguments():
    parser = argparse.ArgumentParser(description="Pretrain a resnet model with VICReg", add_help=False)

    # Data
    parser.add_argument("--data-dir", type=Path, default="./imagenet100", required=True,
                        help='Path to the image net dataset')

    # Checkpoints
    parser.add_argument("--exp-dir", type=Path, default="./exp",
                        help='Path to the experiment folder, where all logs/checkpoints will be stored')
    parser.add_argument("--log-freq-time", type=int, default=60,
                        help='Print logs to the stats.txt file every [log-freq-time] seconds')

    # Model
    parser.add_argument("--arch", type=str, default="resnet50",
                        help='Architecture of the backbone encoder network')
    parser.add_argument("--mlp", default="8192-8192-8192",
                        help='Size and number of layers of the MLP expander head')

    # Optim
    parser.add_argument("--epochs", type=int, default=100,
                        help='Number of epochs')
    parser.add_argument("--batch-size", type=int, default=2048,
                        help='Effective batch size (per worker batch size is [batch-size] / world-size)')
    parser.add_argument("--base-lr", type=float, default=0.2,
                        help='Base learning rate, effective learning after warmup is [base-lr] * [batch-size] / 256')
    parser.add_argument("--wd", type=float, default=1e-6,
                        help='Weight decay')

    # Loss
    parser.add_argument("--sim-coeff", type=float, default=25.0,
                        help='Invariance regularization loss coefficient')
    parser.add_argument("--std-coeff", type=float, default=25.0,
                        help='Variance regularization loss coefficient')
    parser.add_argument("--cov-coeff", type=float, default=1.0,
                        help='Covariance regularization loss coefficient')

    # Running
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')

    # Distributed
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist-url', default='env://',
                        help='url used to set up distributed training')


    ##cam
    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument(
        '--image-path',
        type=str,
        default='./examples/test.jpg',
        help='Input image path')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen_smooth',
        action='store_true',
        help='Reduce noise by taking the first principle componenet'
             'of cam_weights*activations')
    parser.add_argument('--method', type=str, default='gradcam',
                        choices=['gradcam', 'gradcam++',
                                 'scorecam', 'xgradcam',
                                 'ablationcam', 'eigencam',
                                 'eigengradcam', 'layercam', 'fullgrad'],
                        help='Can be gradcam/gradcam++/scorecam/xgradcam'
                             '/ablationcam/eigencam/eigengradcam/layercam')

    return parser


def main(args):
    torch.backends.cudnn.benchmark = True
    init_distributed_mode(args)
    print(args)
    gpu = torch.device(args.device)

    if args.rank == 0:
        args.exp_dir.mkdir(parents=True, exist_ok=True)
        stats_file = open(args.exp_dir / "stats.txt", "a", buffering=1)
        print(" ".join(sys.argv))
        print(" ".join(sys.argv), file=stats_file)

    transforms = aug.TrainTransform()

    dataset = datasets.ImageFolder(args.data_dir / "train/", transforms)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
    assert args.batch_size % args.world_size == 0
    per_device_batch_size = args.batch_size // args.world_size
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=per_device_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=sampler,
    )

    model = VICReg(args).cuda(gpu)
    model_backbone=model.backbone
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    optimizer = LARS(
        model.parameters(),
        lr=0,
        weight_decay=args.wd,
        weight_decay_filter=exclude_bias_and_norm,
        lars_adaptation_filter=exclude_bias_and_norm,
    )

    if (args.exp_dir / "model.pth").is_file():
        if args.rank == 0:
            print("resuming from checkpoint")
        ckpt = torch.load(args.exp_dir / "model.pth", map_location="cpu")
        start_epoch = ckpt["epoch"]
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
    else:
        start_epoch = 0

    start_time = last_logging = time.time()
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        for step, ((x, y), _) in enumerate(loader, start=epoch * len(loader)):
            x = x.cuda(gpu, non_blocking=True)
            y = y.cuda(gpu, non_blocking=True)

            lr = adjust_learning_rate(args, optimizer, loader, step)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss = model.forward(x, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            current_time = time.time()
            if args.rank == 0 and current_time - last_logging > args.log_freq_time:
                stats = dict(
                    epoch=epoch,
                    step=step,
                    loss=loss.item(),
                    time=int(current_time - start_time),
                    lr=lr,
                )
                print(json.dumps(stats))
                print(json.dumps(stats), file=stats_file)
                last_logging = current_time
        if args.rank == 0:
            state = dict(
                epoch=epoch + 1,
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
            )
            torch.save(state, args.exp_dir / "model.pth")
    if args.rank == 0:
        torch.save(model.module.backbone.state_dict(), args.exp_dir / "resnet50.pth")

    # cam
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad}

    model = model_backbone
    #model, _ = resnet50()
    #model = resnet50(pretrained=True)
    target_layers = [model.layer4[-1]]
    #--------------------------
    rgb_img = cv2.imread('./examples/test.JPEG', 1)[:, :, ::-1]
    rgb_img = np.float32(rgb_img) / 255
    rgb_img = cv2.resize(rgb_img, (224, 224))
    input_tensor = preprocess_image(rgb_img,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    rgb_img1 = cv2.imread('./examples/test1.JPEG', 1)[:, :, ::-1]
    rgb_img1 = np.float32(rgb_img1) / 255
    rgb_img1 = cv2.resize(rgb_img1, (224, 224))
    input_tensor1 = preprocess_image(rgb_img1,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    rgb_img2 = cv2.imread('./examples/test2.JPEG', 1)[:, :, ::-1]
    rgb_img2 = np.float32(rgb_img2) / 255
    rgb_img2 = cv2.resize(rgb_img2, (224, 224))
    input_tensor2 = preprocess_image(rgb_img2,
                                     mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    rgb_img3 = cv2.imread('./examples/test3.JPEG', 1)[:, :, ::-1]
    rgb_img3 = np.float32(rgb_img3) / 255
    rgb_img3 = cv2.resize(rgb_img3, (224, 224))
    input_tensor3 = preprocess_image(rgb_img3,
                                     mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    rgb_img4 = cv2.imread('./examples/test4.JPEG', 1)[:, :, ::-1]
    rgb_img4 = np.float32(rgb_img4) / 255
    rgb_img4 = cv2.resize(rgb_img4, (224, 224))
    input_tensor4 = preprocess_image(rgb_img4,
                                     mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    rgb_img5 = cv2.imread('./examples/test5.JPEG', 1)[:, :, ::-1]
    rgb_img5 = np.float32(rgb_img5) / 255
    rgb_img5 = cv2.resize(rgb_img5, (224, 224))
    input_tensor5 = preprocess_image(rgb_img5,
                                     mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    rgb_img6 = cv2.imread('./examples/test6.JPEG', 1)[:, :, ::-1]
    rgb_img6 = np.float32(rgb_img6) / 255
    rgb_img6 = cv2.resize(rgb_img6, (224, 224))
    input_tensor6 = preprocess_image(rgb_img6,
                                     mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    rgb_img7 = cv2.imread('./examples/test7.JPEG', 1)[:, :, ::-1]
    rgb_img7 = np.float32(rgb_img7) / 255
    rgb_img7 = cv2.resize(rgb_img7, (224, 224))
    input_tensor7 = preprocess_image(rgb_img7,
                                     mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    rgb_img8 = cv2.imread('./examples/test8.JPEG', 1)[:, :, ::-1]
    rgb_img8 = np.float32(rgb_img8) / 255
    rgb_img8 = cv2.resize(rgb_img8, (224, 224))
    input_tensor8 = preprocess_image(rgb_img8,
                                     mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    rgb_img9 = cv2.imread('./examples/test9.JPEG', 1)[:, :, ::-1]
    rgb_img9 = np.float32(rgb_img9) / 255
    rgb_img9 = cv2.resize(rgb_img9, (224, 224))
    input_tensor9 = preprocess_image(rgb_img9,
                                     mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    rgb_img10 = cv2.imread('./examples/test10.jpg', 1)[:, :, ::-1]
    rgb_img10 = np.float32(rgb_img10) / 255
    rgb_img10 = cv2.resize(rgb_img10, (224, 224))
    input_tensor10 = preprocess_image(rgb_img10,
                                     mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    rgb_img11 = cv2.imread('./examples/test11.jpg', 1)[:, :, ::-1]
    rgb_img11 = np.float32(rgb_img11) / 255
    rgb_img11 = cv2.resize(rgb_img11, (224, 224))
    input_tensor11 = preprocess_image(rgb_img11,
                                      mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])

    rgb_img12 = cv2.imread('./examples/test12.jpg', 1)[:, :, ::-1]
    rgb_img12 = np.float32(rgb_img12) / 255
    rgb_img12 = cv2.resize(rgb_img12, (224, 224))
    input_tensor12 = preprocess_image(rgb_img12,
                                      mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])

    rgb_img13 = cv2.imread('./examples/test13.jpg', 1)[:, :, ::-1]
    rgb_img13 = np.float32(rgb_img13) / 255
    rgb_img13 = cv2.resize(rgb_img13, (224, 224))
    input_tensor13 = preprocess_image(rgb_img13,
                                      mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])

    rgb_img14 = cv2.imread('./examples/test14.jpg', 1)[:, :, ::-1]
    rgb_img14 = np.float32(rgb_img14) / 255
    rgb_img14 = cv2.resize(rgb_img14, (224, 224))
    input_tensor14 = preprocess_image(rgb_img14,
                                      mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])

    rgb_img15 = cv2.imread('./examples/test15.jpg', 1)[:, :, ::-1]
    rgb_img15 = np.float32(rgb_img15) / 255
    rgb_img15 = cv2.resize(rgb_img15, (224, 224))
    input_tensor15 = preprocess_image(rgb_img15,
                                      mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])

    rgb_img16 = cv2.imread('./examples/test16.jpg', 1)[:, :, ::-1]
    rgb_img16 = np.float32(rgb_img16) / 255
    rgb_img16 = cv2.resize(rgb_img16, (224, 224))
    input_tensor16 = preprocess_image(rgb_img16,
                                      mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])

    rgb_img17 = cv2.imread('./examples/test17.jpg', 1)[:, :, ::-1]
    rgb_img17 = np.float32(rgb_img17) / 255
    rgb_img17 = cv2.resize(rgb_img17, (224, 224))
    input_tensor17 = preprocess_image(rgb_img17,
                                      mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])

    rgb_img18 = cv2.imread('./examples/test18.jpg', 1)[:, :, ::-1]
    rgb_img18 = np.float32(rgb_img18) / 255
    rgb_img18 = cv2.resize(rgb_img18, (224, 224))
    input_tensor18 = preprocess_image(rgb_img18,
                                      mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])

    rgb_img19 = cv2.imread('./examples/test19.jpg', 1)[:, :, ::-1]
    rgb_img19 = np.float32(rgb_img19) / 255
    rgb_img19 = cv2.resize(rgb_img19, (224, 224))
    input_tensor19 = preprocess_image(rgb_img19,
                                      mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])



    #----------------------------

    target_category = None#want to set None

    # Using the with statement ensures the context is freed, and you can
    # recreate different CAM objects in a loop.
    cam_algorithm = methods[args.method]

    with cam_algorithm(model=model,
                       target_layers=target_layers,
                       use_cuda=args.use_cuda) as cam:

        # AblationCAM and ScoreCAM have batched implementations.
        # You can override the internal batch size for faster computation.
        cam.batch_size = 32
        #-------------------------------------------------

        grayscale_cam, weights = cam(input_tensor=input_tensor,
                                     target_category=target_category,
                                     aug_smooth=args.aug_smooth,
                                     eigen_smooth=args.eigen_smooth)
        grayscale_cam = grayscale_cam[0, :]
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

        grayscale_cam1, weights1 = cam(input_tensor=input_tensor1,
                                     target_category=target_category,
                                     aug_smooth=args.aug_smooth,
                                     eigen_smooth=args.eigen_smooth)
        grayscale_cam1 = grayscale_cam1[0, :]
        cam_image1 = show_cam_on_image(rgb_img1, grayscale_cam1, use_rgb=True)
        cam_image1 = cv2.cvtColor(cam_image1, cv2.COLOR_RGB2BGR)

        grayscale_cam2, weights2 = cam(input_tensor=input_tensor2,
                                       target_category=target_category,
                                       aug_smooth=args.aug_smooth,
                                       eigen_smooth=args.eigen_smooth)
        grayscale_cam2 = grayscale_cam2[0, :]
        cam_image2 = show_cam_on_image(rgb_img2, grayscale_cam2, use_rgb=True)
        cam_image2 = cv2.cvtColor(cam_image2, cv2.COLOR_RGB2BGR)

        grayscale_cam3, weights3 = cam(input_tensor=input_tensor3,
                                       target_category=target_category,
                                       aug_smooth=args.aug_smooth,
                                       eigen_smooth=args.eigen_smooth)
        grayscale_cam3 = grayscale_cam3[0, :]
        cam_image3 = show_cam_on_image(rgb_img3, grayscale_cam3, use_rgb=True)
        cam_image3 = cv2.cvtColor(cam_image3, cv2.COLOR_RGB2BGR)

        grayscale_cam4, weights4 = cam(input_tensor=input_tensor4,
                                       target_category=target_category,
                                       aug_smooth=args.aug_smooth,
                                       eigen_smooth=args.eigen_smooth)
        grayscale_cam4 = grayscale_cam4[0, :]
        cam_image4 = show_cam_on_image(rgb_img4, grayscale_cam4, use_rgb=True)
        cam_image4 = cv2.cvtColor(cam_image4, cv2.COLOR_RGB2BGR)

        grayscale_cam5, weights5 = cam(input_tensor=input_tensor5,
                                       target_category=target_category,
                                       aug_smooth=args.aug_smooth,
                                       eigen_smooth=args.eigen_smooth)
        grayscale_cam5 = grayscale_cam5[0, :]
        cam_image5 = show_cam_on_image(rgb_img5, grayscale_cam5, use_rgb=True)
        cam_image5 = cv2.cvtColor(cam_image5, cv2.COLOR_RGB2BGR)

        grayscale_cam6, weights6 = cam(input_tensor=input_tensor6,
                                       target_category=target_category,
                                       aug_smooth=args.aug_smooth,
                                       eigen_smooth=args.eigen_smooth)
        grayscale_cam6 = grayscale_cam6[0, :]
        cam_image6 = show_cam_on_image(rgb_img6, grayscale_cam6, use_rgb=True)
        cam_image6 = cv2.cvtColor(cam_image6, cv2.COLOR_RGB2BGR)

        grayscale_cam7, weights7 = cam(input_tensor=input_tensor7,
                                       target_category=target_category,
                                       aug_smooth=args.aug_smooth,
                                       eigen_smooth=args.eigen_smooth)
        grayscale_cam7 = grayscale_cam7[0, :]
        cam_image7 = show_cam_on_image(rgb_img7, grayscale_cam7, use_rgb=True)
        cam_image7 = cv2.cvtColor(cam_image7, cv2.COLOR_RGB2BGR)

        grayscale_cam8, weights8 = cam(input_tensor=input_tensor8,
                                       target_category=target_category,
                                       aug_smooth=args.aug_smooth,
                                       eigen_smooth=args.eigen_smooth)
        grayscale_cam8 = grayscale_cam8[0, :]
        cam_image8 = show_cam_on_image(rgb_img8, grayscale_cam8, use_rgb=True)
        cam_image8 = cv2.cvtColor(cam_image8, cv2.COLOR_RGB2BGR)

        grayscale_cam9, weights9 = cam(input_tensor=input_tensor9,
                                       target_category=target_category,
                                       aug_smooth=args.aug_smooth,
                                       eigen_smooth=args.eigen_smooth)
        grayscale_cam9 = grayscale_cam9[0, :]
        cam_image9 = show_cam_on_image(rgb_img9, grayscale_cam9, use_rgb=True)
        cam_image9 = cv2.cvtColor(cam_image9, cv2.COLOR_RGB2BGR)

        grayscale_cam10, weights10 = cam(input_tensor=input_tensor10,
                                       target_category=target_category,
                                       aug_smooth=args.aug_smooth,
                                       eigen_smooth=args.eigen_smooth)
        grayscale_cam10 = grayscale_cam10[0, :]
        cam_image10 = show_cam_on_image(rgb_img10, grayscale_cam10, use_rgb=True)
        cam_image10 = cv2.cvtColor(cam_image10, cv2.COLOR_RGB2BGR)

        grayscale_cam11, weights11 = cam(input_tensor=input_tensor11,
                                         target_category=target_category,
                                         aug_smooth=args.aug_smooth,
                                         eigen_smooth=args.eigen_smooth)
        grayscale_cam11 = grayscale_cam11[0, :]
        cam_image11 = show_cam_on_image(rgb_img11, grayscale_cam11, use_rgb=True)
        cam_image11 = cv2.cvtColor(cam_image11, cv2.COLOR_RGB2BGR)

        grayscale_cam12, weights12 = cam(input_tensor=input_tensor12,
                                         target_category=target_category,
                                         aug_smooth=args.aug_smooth,
                                         eigen_smooth=args.eigen_smooth)
        grayscale_cam12 = grayscale_cam12[0, :]
        cam_image12 = show_cam_on_image(rgb_img12, grayscale_cam12, use_rgb=True)
        cam_image12 = cv2.cvtColor(cam_image12, cv2.COLOR_RGB2BGR)

        grayscale_cam13, weights13 = cam(input_tensor=input_tensor13,
                                         target_category=target_category,
                                         aug_smooth=args.aug_smooth,
                                         eigen_smooth=args.eigen_smooth)
        grayscale_cam13 = grayscale_cam13[0, :]
        cam_image13 = show_cam_on_image(rgb_img13, grayscale_cam13, use_rgb=True)
        cam_image13 = cv2.cvtColor(cam_image13, cv2.COLOR_RGB2BGR)

        grayscale_cam14, weights14 = cam(input_tensor=input_tensor14,
                                         target_category=target_category,
                                         aug_smooth=args.aug_smooth,
                                         eigen_smooth=args.eigen_smooth)
        grayscale_cam14 = grayscale_cam14[0, :]
        cam_image14 = show_cam_on_image(rgb_img14, grayscale_cam14, use_rgb=True)
        cam_image14 = cv2.cvtColor(cam_image14, cv2.COLOR_RGB2BGR)

        grayscale_cam15, weights15 = cam(input_tensor=input_tensor15,
                                         target_category=target_category,
                                         aug_smooth=args.aug_smooth,
                                         eigen_smooth=args.eigen_smooth)
        grayscale_cam15 = grayscale_cam15[0, :]
        cam_image15 = show_cam_on_image(rgb_img15, grayscale_cam15, use_rgb=True)
        cam_image15 = cv2.cvtColor(cam_image15, cv2.COLOR_RGB2BGR)

        grayscale_cam16, weights16 = cam(input_tensor=input_tensor16,
                                         target_category=target_category,
                                         aug_smooth=args.aug_smooth,
                                         eigen_smooth=args.eigen_smooth)
        grayscale_cam16 = grayscale_cam16[0, :]
        cam_image16 = show_cam_on_image(rgb_img16, grayscale_cam16, use_rgb=True)
        cam_image16 = cv2.cvtColor(cam_image16, cv2.COLOR_RGB2BGR)

        grayscale_cam17, weights17 = cam(input_tensor=input_tensor17,
                                         target_category=target_category,
                                         aug_smooth=args.aug_smooth,
                                         eigen_smooth=args.eigen_smooth)
        grayscale_cam17 = grayscale_cam17[0, :]
        cam_image17 = show_cam_on_image(rgb_img17, grayscale_cam17, use_rgb=True)
        cam_image17 = cv2.cvtColor(cam_image17, cv2.COLOR_RGB2BGR)

        grayscale_cam18, weights18 = cam(input_tensor=input_tensor18,
                                         target_category=target_category,
                                         aug_smooth=args.aug_smooth,
                                         eigen_smooth=args.eigen_smooth)
        grayscale_cam18 = grayscale_cam18[0, :]
        cam_image18 = show_cam_on_image(rgb_img18, grayscale_cam18, use_rgb=True)
        cam_image18 = cv2.cvtColor(cam_image18, cv2.COLOR_RGB2BGR)

        grayscale_cam19, weights19 = cam(input_tensor=input_tensor19,
                                         target_category=target_category,
                                         aug_smooth=args.aug_smooth,
                                         eigen_smooth=args.eigen_smooth)
        grayscale_cam19 = grayscale_cam19[0, :]
        cam_image19 = show_cam_on_image(rgb_img19, grayscale_cam19, use_rgb=True)
        cam_image19 = cv2.cvtColor(cam_image19, cv2.COLOR_RGB2BGR)




        #----------------------------------------------------------



    cv2.imwrite(f'{args.method}_cam.jpg', cam_image)
    cv2.imwrite(f'{args.method}_cam1.jpg', cam_image1)
    cv2.imwrite(f'{args.method}_cam2.jpg', cam_image2)
    cv2.imwrite(f'{args.method}_cam3.jpg', cam_image3)
    cv2.imwrite(f'{args.method}_cam4.jpg', cam_image4)
    cv2.imwrite(f'{args.method}_cam5.jpg', cam_image5)
    cv2.imwrite(f'{args.method}_cam6.jpg', cam_image6)
    cv2.imwrite(f'{args.method}_cam7.jpg', cam_image7)
    cv2.imwrite(f'{args.method}_cam8.jpg', cam_image8)
    cv2.imwrite(f'{args.method}_cam9.jpg', cam_image9)
    cv2.imwrite(f'{args.method}_cam10.jpg', cam_image10)
    cv2.imwrite(f'{args.method}_cam11.jpg', cam_image11)
    cv2.imwrite(f'{args.method}_cam12.jpg', cam_image12)
    cv2.imwrite(f'{args.method}_cam13.jpg', cam_image13)
    cv2.imwrite(f'{args.method}_cam14.jpg', cam_image14)
    cv2.imwrite(f'{args.method}_cam15.jpg', cam_image15)
    cv2.imwrite(f'{args.method}_cam16.jpg', cam_image16)
    cv2.imwrite(f'{args.method}_cam17.jpg', cam_image17)
    cv2.imwrite(f'{args.method}_cam18.jpg', cam_image18)
    cv2.imwrite(f'{args.method}_cam19.jpg', cam_image19)
    # cv2.imwrite(f'{args.method}_gb.jpg', gb)
    # cv2.imwrite(f'{args.method}_cam_gb.jpg', cam_gb)


def adjust_learning_rate(args, optimizer, loader, step):
    max_steps = args.epochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = args.base_lr * args.batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


class VICReg(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_features = int(args.mlp.split("-")[-1])
        self.backbone, self.embedding = resnet.__dict__[args.arch](
            zero_init_residual=True
        )
        self.projector = Projector(args, self.embedding)

    def forward(self, x, y):
        x = self.projector(self.backbone(x))
        y = self.projector(self.backbone(y))

        repr_loss = F.mse_loss(x, y)

        x = torch.cat(FullGatherLayer.apply(x), dim=0)
        y = torch.cat(FullGatherLayer.apply(y), dim=0)
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (self.args.batch_size - 1)
        cov_y = (y.T @ y) / (self.args.batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
            self.num_features
        ) + off_diagonal(cov_y).pow_(2).sum().div(self.num_features)

        loss = (
            self.args.sim_coeff * repr_loss
            + self.args.std_coeff * std_loss
            + self.args.cov_coeff * cov_loss
        )
        return loss


def Projector(args, embedding):
    mlp_spec = f"{embedding}-{args.mlp}"
    layers = []
    f = list(map(int, mlp_spec.split("-")))
    for i in range(len(f) - 2):
        layers.append(nn.Linear(f[i], f[i + 1]))
        layers.append(nn.BatchNorm1d(f[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(f[-2], f[-1], bias=False))
    return nn.Sequential(*layers)


def exclude_bias_and_norm(p):
    return p.ndim == 1


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class LARS(optim.Optimizer):
    def __init__(
        self,
        params,
        lr,
        weight_decay=0,
        momentum=0.9,
        eta=0.001,
        weight_decay_filter=None,
        lars_adaptation_filter=None,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            eta=eta,
            weight_decay_filter=weight_decay_filter,
            lars_adaptation_filter=lars_adaptation_filter,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                dp = p.grad

                if dp is None:
                    continue

                if g["weight_decay_filter"] is None or not g["weight_decay_filter"](p):
                    dp = dp.add(p, alpha=g["weight_decay"])

                if g["lars_adaptation_filter"] is None or not g[
                    "lars_adaptation_filter"
                ](p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(
                        param_norm > 0.0,
                        torch.where(
                            update_norm > 0, (g["eta"] * param_norm / update_norm), one
                        ),
                        one,
                    )
                    dp = dp.mul(q)

                param_state = self.state[p]
                if "mu" not in param_state:
                    param_state["mu"] = torch.zeros_like(p)
                mu = param_state["mu"]
                mu.mul_(g["momentum"]).add_(dp)

                p.add_(mu, alpha=-g["lr"])


def batch_all_gather(x):
    x_list = FullGatherLayer.apply(x)
    return torch.cat(x_list, dim=0)


class FullGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.environ["SLURM_JOB_ID"]}')
    exit()


def handle_sigterm(signum, frame):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser('VICReg training script', parents=[get_arguments()])
    args = parser.parse_args()
    main(args)
