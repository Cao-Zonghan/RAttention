import argparse
import os
import csv
from math import log10

import pandas as pd
import torch
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from torch.utils.tensorboard import SummaryWriter

from data_utils import TrainDatasetFromFolder_radar3D_adc, ValDatasetFromFolder_radar3D_adc, \
    TrainRadarADCChannelDataset, ValRadarADCChannelDataset
from loss import GeneratorLoss
from wavelet_LL2HH_attention import radar_2d_wavelet
# from generator_radar3d_adc import radar_2d_wavelet
# from generator_radar3d_adc import Generator_radar3D_adc, Generator_radar2D_adc_complex
from model_complex_fast_1 import Generator_radar2D_adc_complex_fast
from model_complex_wavelet import Wavelet_radar3D_adc
from model_unet3d_adc import Generator_radar2D_adc_UNet
# ---------------- argparse ---------------- #

parser = argparse.ArgumentParser(description='Train Super Resolution Models')

parser.add_argument('--crop_size', default=128, type=int)
parser.add_argument('--upscale_factor', default=1, type=int)
parser.add_argument('--num_epochs', default=50, type=int)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--model_name', default=None, type=str)
parser.add_argument('--datamode', default="real", type=str)
parser.add_argument('--lr', default=0.0002, type=float)
parser.add_argument('--low_Azimuth', type=int, default=4)
parser.add_argument('--high_Azimuth', type=int, default=12)
parser.add_argument('--lr_channel', type=int, default=[0, 2, 8, 10])
parser.add_argument('--output', default='./out')
parser.add_argument('--train_data', default='./train')
parser.add_argument('--val_data', default='./test')

# resume
parser.add_argument('--resume', default=None, type=str, help='checkpoint path')

mode = 'normal'

num_low_receiver = 4
num_angle_bins = 256
debug = False
abs_vis = True
tune_voc = False
beg_voc = 40


# ---------------- RAD ---------------- #

def RAD_map(range_plot):
    range_plot = np.fft.fft(range_plot, axis=0)
    range_doppler = np.fft.fft(range_plot, axis=2)
    range_doppler = np.fft.fftshift(range_doppler, axes=2)
    padding = ((0, 0), (0, num_angle_bins - range_doppler.shape[1]), (0, 0))
    range_azimuth = np.pad(range_doppler, padding, mode='constant')
    range_azimuth = np.fft.fft(range_azimuth, axis=1)
    range_azimuth = np.fft.fftshift(range_azimuth, axes=1)
    out_img = np.rot90(range_azimuth, 2, axes=(0, 1))
    return out_img


def to_complex_if_needed(tensor):
    if torch.is_complex(tensor):
        return tensor

    if tensor.dim() > 1 and tensor.size(1) == 2:
        return torch.complex(tensor[:, 0, ...], tensor[:, 1, ...])

    return tensor


def compute_signal_domain_metrics(pred_tensor, gt_tensor, eps=1e-12):
    pred_eval = to_complex_if_needed(pred_tensor.detach())
    gt_eval = to_complex_if_needed(gt_tensor.detach())

    pred_mag = torch.abs(pred_eval)
    gt_mag = torch.abs(gt_eval)

    if pred_mag.size(0) != gt_mag.size(0):
        raise ValueError(
            f"Batch size mismatch between prediction and GT: {pred_mag.size(0)} vs {gt_mag.size(0)}"
        )

    reduce_dims = tuple(range(1, gt_mag.ndim))
    norm = gt_mag.amax(dim=reduce_dims, keepdim=True).clamp_min(eps)

    pred_norm = pred_mag / norm
    gt_norm = gt_mag / norm

    sample_mse = ((pred_norm - gt_norm) ** 2).reshape(gt_mag.size(0), -1).mean(dim=1)
    sample_psnr = 10 * torch.log10(1.0 / (sample_mse + eps))

    batch_mse = float(sample_mse.mean().item())
    batch_psnr = float(sample_psnr.mean().item())

    return batch_mse, batch_psnr


def format_flops(flops):
    units = [(1e12, "TFLOPs"), (1e9, "GFLOPs"), (1e6, "MFLOPs"), (1e3, "KFLOPs")]
    for threshold, suffix in units:
        if flops >= threshold:
            return f"{flops / threshold:.3f} {suffix}"
    return f"{flops:.0f} FLOPs"


def estimate_model_flops(model, example_input):
    flops = {}
    hooks = []

    def register_module_hook(module):
        if len(list(module.children())) > 0:
            return

        def hook(_, inputs, output):
            x = inputs[0] if isinstance(inputs, tuple) and len(inputs) > 0 else inputs
            if not isinstance(x, torch.Tensor):
                return

            if isinstance(module, torch.nn.Conv2d):
                out = output[0] if isinstance(output, tuple) else output
                if not isinstance(out, torch.Tensor):
                    return
                batch_size = out.shape[0]
                out_channels = out.shape[1]
                out_height = out.shape[2]
                out_width = out.shape[3]
                kernel_h, kernel_w = module.kernel_size
                in_channels = module.in_channels // module.groups
                bias_ops = 1 if module.bias is not None else 0
                ops_per_element = kernel_h * kernel_w * in_channels + bias_ops
                flops[module] = 2 * batch_size * out_channels * out_height * out_width * ops_per_element
            elif isinstance(module, torch.nn.ConvTranspose2d):
                out = output[0] if isinstance(output, tuple) else output
                if not isinstance(out, torch.Tensor):
                    return
                batch_size = out.shape[0]
                out_channels = out.shape[1]
                out_height = out.shape[2]
                out_width = out.shape[3]
                kernel_h, kernel_w = module.kernel_size
                in_channels = module.in_channels // module.groups
                bias_ops = 1 if module.bias is not None else 0
                ops_per_element = kernel_h * kernel_w * in_channels + bias_ops
                flops[module] = 2 * batch_size * out_channels * out_height * out_width * ops_per_element
            elif isinstance(module, torch.nn.Linear):
                out = output[0] if isinstance(output, tuple) else output
                if not isinstance(out, torch.Tensor):
                    return
                batch_elements = out.numel() // out.shape[-1]
                bias_ops = 1 if module.bias is not None else 0
                flops[module] = 2 * batch_elements * module.out_features * (module.in_features + bias_ops)
            elif isinstance(module, torch.nn.BatchNorm2d):
                out = output[0] if isinstance(output, tuple) else output
                if not isinstance(out, torch.Tensor):
                    return
                flops[module] = 2 * out.numel()
            elif isinstance(module, (torch.nn.PReLU, torch.nn.GELU, torch.nn.ReLU, torch.nn.LeakyReLU)):
                out = output[0] if isinstance(output, tuple) else output
                if not isinstance(out, torch.Tensor):
                    return
                flops[module] = out.numel()

        hooks.append(module.register_forward_hook(hook))

    for module in model.modules():
        register_module_hook(module)

    was_training = model.training
    model.eval()
    with torch.no_grad():
        _ = model(example_input)
    if was_training:
        model.train()

    for hook in hooks:
        hook.remove()

    return sum(flops.values())


# ---------------- main ---------------- #

if __name__ == '__main__':

    opt = parser.parse_args()

    print(
        '[DATASET DEBUG] ValDatasetFromFolder_radar3D_adc signature requires: '
        'adc_data_dir, num_low_receiver, hr_data_dir, lr_data_dir, crop_size, upscale_factor, ...'
    )
    print(
        f"[DATASET DEBUG] current config -> datamode={opt.datamode}, crop_size={opt.crop_size}, "
        f"upscale_factor={opt.upscale_factor}, low_Azimuth={opt.low_Azimuth}, high_Azimuth={opt.high_Azimuth}"
    )
    print(
        '[DATASET DEBUG] current val_set call omits crop_size before upscale_factor; '
        'if datamode=real this will raise TypeError before DataLoader construction.'
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(opt.output, exist_ok=True)

    log_path = os.path.join(opt.output, "train_log.csv")
    best_record_path = os.path.join(opt.output, "best_model_record.csv")

    if not os.path.exists(log_path):
        with open(log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_mse", "batch_psnr", "best_batch_psnr", "best_epoch"])

    if not os.path.exists(best_record_path):
        with open(best_record_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["best_epoch", "best_batch_psnr", "train_loss", "val_mse"])

    writer_tb = SummaryWriter(opt.output)

    # ---------------- dataset ---------------- #

    if opt.datamode == 'real':
        train_set = TrainDatasetFromFolder_radar3D_adc(
            opt.train_data, opt.low_Azimuth, None, None,
            crop_size=opt.crop_size,
            upscale_factor=opt.upscale_factor,
            index_list=1,
            num_high_receiver=opt.high_Azimuth
        )

        val_set = ValDatasetFromFolder_radar3D_adc(
            opt.val_data, opt.low_Azimuth, None, None,
            crop_size=opt.crop_size,
            upscale_factor=opt.upscale_factor,
            index_list=1,
            num_high_receiver=opt.high_Azimuth
        )
    else:
        train_set = TrainRadarADCChannelDataset(
            opt.train_data, opt.high_Azimuth,
            lr_receiver_idx=opt.lr_channel,
            datamode=opt.datamode
        )

        val_set = ValRadarADCChannelDataset(
            opt.val_data, opt.high_Azimuth,
            lr_receiver_idx=opt.lr_channel,
            datamode=opt.datamode
        )

    train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False, num_workers=4)

    # ---------------- model ---------------- #

    UPSCALE_FACTOR = opt.high_Azimuth / opt.low_Azimuth

    if opt.datamode == "real":
        # netG = Generator_radar3D_adc(UPSCALE_FACTOR, input_dim=2)
        netG = radar_2d_wavelet(UPSCALE_FACTOR)
    else:
        # netG = Generator_radar2D_adc_complex(UPSCALE_FACTOR, input_dim=4)
        # netG = Generator_radar2D_adc_complex_fast(UPSCALE_FACTOR, input_dim=4)
        netG = Wavelet_radar3D_adc(UPSCALE_FACTOR, input_dim=2)
    netG = netG.to(device)

    criterion = GeneratorLoss(opt.datamode).to(device)
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr)

    start_epoch = 1

    # 仅保存在验证集上 batch_psnr 最好的模型
    best_psnr = float('-inf')
    best_epoch = 0

    # ---------------- resume ---------------- #

    if opt.resume is not None:
        ckpt = torch.load(opt.resume, map_location=device)
        netG.load_state_dict(ckpt["model"])
        optimizerG.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        if "best_psnr" in ckpt:
            best_psnr = ckpt["best_psnr"]
        if "best_epoch" in ckpt:
            best_epoch = ckpt["best_epoch"]
        print(f"Resumed from epoch {ckpt['epoch']}, Current Best batch_psnr: {best_psnr:.4f} (epoch {best_epoch})")

    example_input = torch.randn(1, 2, opt.low_Azimuth, opt.crop_size, opt.crop_size, device=device)
    generator_params = sum(p.numel() for p in netG.parameters())
    generator_flops = estimate_model_flops(netG, example_input)
    print(f"# Generator params: {generator_params}, FLOPs: {format_flops(generator_flops)}")

    # ---------------- training ---------------- #

    for epoch in range(start_epoch, opt.num_epochs + 1):

        netG.train()
        running_loss = 0
        count = 0

        train_bar = tqdm(train_loader)

        for adc_lr, adc_hr in train_bar:
            adc_lr = adc_lr.to(device)
            adc_hr = adc_hr.to(device)

            optimizerG.zero_grad()

            fake = netG(adc_lr)

            loss = criterion(None, fake, adc_hr)
            loss.backward()
            optimizerG.step()

            bs = adc_lr.size(0)
            running_loss += loss.item() * bs
            count += bs

            train_bar.set_description(f"[{epoch}] G_loss: {running_loss / count:.6f}")

        g_loss_epoch = running_loss / count

        # ---------------- validation ---------------- #

        netG.eval()
        mse_total = 0
        psnr_total = 0
        num = 0

        with torch.no_grad():
            for adc_lr, adc_hr in tqdm(val_loader):

                adc_lr = adc_lr.to(device)
                adc_hr = adc_hr.to(device)

                adc_sr = netG(adc_lr)
                mse, psnr = compute_signal_domain_metrics(adc_sr, adc_hr)

                bs = adc_lr.size(0)
                mse_total += mse * bs
                psnr_total += psnr * bs
                num += bs

        mse_eval = mse_total / num if num > 0 else 0
        batch_psnr = psnr_total / num if num > 0 else 0

        print(f"Epoch {epoch}: MSE {mse_eval:.6f} | batch_PSNR {batch_psnr:.4f}")

        # ---------------- save ---------------- #

        if batch_psnr > best_psnr:
            best_psnr = batch_psnr
            best_epoch = epoch
            best_model_path = os.path.join(opt.output, "netG_best.pth")
            state_dict = {
                "epoch": epoch,
                "best_epoch": best_epoch,
                "model": netG.state_dict(),
                "optimizer": optimizerG.state_dict(),
                "g_loss": g_loss_epoch,
                "mse": mse_eval,
                "batch_psnr": batch_psnr,
                "best_psnr": best_psnr
            }
            torch.save(state_dict, best_model_path)
            print(f"  [Info] New Best Model saved! batch_PSNR: {best_psnr:.4f} at epoch {best_epoch}")

            with open(best_record_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["best_epoch", "best_batch_psnr", "train_loss", "val_mse"])
                writer.writerow([best_epoch, best_psnr, g_loss_epoch, mse_eval])

        # CSV log
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, g_loss_epoch, mse_eval, batch_psnr, best_psnr, best_epoch])

        # TensorBoard
        writer_tb.add_scalar("Loss/G", g_loss_epoch, epoch)
        writer_tb.add_scalar("Val/batch_PSNR", batch_psnr, epoch)
        writer_tb.add_scalar("Val/MSE", mse_eval, epoch)

    writer_tb.close()
    print("Training finished.")