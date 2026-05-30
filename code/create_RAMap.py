import os
import argparse
import numpy as np
import torch
import cv2
from model import Generator_radar3D_adc, Generator_radar2D_adc_complex
from generator_radar3d_adc import radar_2d_wavelet
from glob import glob
from tqdm import tqdm

# -------------------------------------------------
# RAD helper
# -------------------------------------------------

num_angle_bins = 256


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


# -------------------------------------------------
# Metrics (LINEAR domain)
# -------------------------------------------------

def compute_mse_psnr(pred_rad, gt_rad, eps=1e-12):
    pred_mag = np.abs(pred_rad)
    gt_mag = np.abs(gt_rad)

    if pred_mag.shape != gt_mag.shape:
        raise ValueError(
            f"Prediction and GT must share the same shape, but got {pred_mag.shape} vs {gt_mag.shape}."
        )

    if pred_mag.ndim == 3:
        pred_mag = pred_mag[None, ...]
        gt_mag = gt_mag[None, ...]
        squeeze_output = True
    elif pred_mag.ndim >= 4:
        squeeze_output = False
    else:
        raise ValueError(
            f"Expected [R, A, D] or [B, R, A, D] inputs, but got ndim={pred_mag.ndim}."
        )

    reduce_axes = tuple(range(1, gt_mag.ndim))
    norm = np.maximum(gt_mag.max(axis=reduce_axes, keepdims=True), eps)

    pred_norm = pred_mag / norm
    gt_norm = gt_mag / norm

    sample_mse = ((pred_norm - gt_norm) ** 2).reshape(gt_mag.shape[0], -1).mean(axis=1)
    sample_psnr = 10 * np.log10(1.0 / (sample_mse + eps))

    if squeeze_output:
        return float(sample_mse[0]), float(sample_psnr[0])

    return sample_mse, sample_psnr


# -------------------------------------------------
# load adc npy
# -------------------------------------------------

def load_adc_npy_for_infer(
        npy_path,
        num_high_receiver,
        num_low_receiver,
        lr_receiver_idx=None,
        datamode='real',
        scale_factor=100.0):

    adc = np.load(npy_path)
    adc = torch.from_numpy(adc)

    if datamode == 'real':
        adc = adc.permute(2, 0, 1)
        adc = torch.view_as_real(adc)
        adc = adc.permute(3, 2, 0, 1).float()
        adc = adc / scale_factor

        A = adc.shape[1]
        start = (A - num_high_receiver) // 2
        adc_hr = adc[:, start:start + num_high_receiver]

        if lr_receiver_idx is None:
            if num_low_receiver is None:
                raise ValueError("num_low_receiver must be provided when lr_receiver_idx is None.")
            start = (num_high_receiver - num_low_receiver) // 2
            adc_lr = adc_hr[:, start:start + num_low_receiver]
        else:
            idx = torch.as_tensor(lr_receiver_idx, dtype=torch.long)
            adc_lr = adc_hr.index_select(dim=1, index=idx)

    else:
        adc = adc.permute(1, 2, 0).to(torch.complex64) / scale_factor

        A = adc.shape[0]
        start = (A - num_high_receiver) // 2
        adc_hr = adc[start:start + num_high_receiver]

        if lr_receiver_idx is None:
            if num_low_receiver is None:
                raise ValueError("num_low_receiver must be provided when lr_receiver_idx is None.")
            start = (num_high_receiver - num_low_receiver) // 2
            adc_lr = adc_hr[start:start + num_low_receiver]
        else:
            idx = torch.as_tensor(lr_receiver_idx, dtype=torch.long)
            adc_lr = adc_hr.index_select(dim=0, index=idx)

    return adc_lr.unsqueeze(0), adc_hr.unsqueeze(0)


# -------------------------------------------------
# main
# -------------------------------------------------

def main():
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser()

    parser.add_argument('--npy_path', required=True)
    parser.add_argument('--real_model', required=True)
    parser.add_argument('--complex_model', required=True)
    parser.add_argument('--save_dir', default='./infer_out')

    parser.add_argument('--low_Azimuth', type=int, default=4)
    parser.add_argument('--high_Azimuth', type=int, default=12)
    parser.add_argument('--lr_channel', type=int, nargs='*', default=None)
    parser.add_argument('--fp16', action="store_true")
    parser.add_argument('--batch_size', type=int, default=32)

    opt = parser.parse_args()

    os.makedirs(opt.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    UPSCALE_FACTOR = opt.high_Azimuth / opt.low_Azimuth

    # ---------------- models ----------------

    print("Loading REAL model...")
    netG_real = Generator_radar3D_adc(UPSCALE_FACTOR, input_dim=2)
    ckpt = torch.load(opt.real_model, map_location="cpu")
    netG_real.load_state_dict(ckpt['model'] if isinstance(ckpt, dict) else ckpt)
    netG_real = netG_real.to(device).eval()

    # print("Loading COMPLEX model...")
    # netG_cplx = Generator_radar2D_adc_complex(UPSCALE_FACTOR, input_dim=4)
    # ckpt = torch.load(opt.complex_model, map_location="cpu")
    # netG_cplx.load_state_dict(ckpt['model'] if isinstance(ckpt, dict) else ckpt)
    # netG_cplx = netG_cplx.to(device).eval()

    print("Loading wavelat model...")
    netG_cplx = radar_2d_wavelet(UPSCALE_FACTOR)
    ckpt = torch.load(opt.complex_model, map_location="cpu")
    netG_cplx.load_state_dict(ckpt['model'] if isinstance(ckpt, dict) else ckpt)
    netG_cplx = netG_cplx.to(device).eval()

    if opt.fp16:
        netG_real = netG_real.half()
        netG_cplx = netG_cplx.half()

    # ---------------- collect npy ----------------

    if os.path.isdir(opt.npy_path):
        npy_list = sorted(glob(os.path.join(opt.npy_path, "*.npy")))
    else:
        npy_list = [opt.npy_path]

    print("Found", len(npy_list), "files")

    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    # ----------- metrics accumulators -----------
    sum_mse_real = 0.0
    sum_psnr_real = 0.0
    sum_mse_cplx = 0.0
    sum_psnr_cplx = 0.0
    count = 0

    pbar = tqdm(list(chunks(npy_list, opt.batch_size)), desc="Batch infer")

    with torch.inference_mode():

        for batch_files in pbar:

            lr_real_list = []
            lr_cplx_list = []
            hr_list = []
            names = []

            for f in batch_files:
                name = os.path.splitext(os.path.basename(f))[0]
                names.append(name)

                adc_lr_real, adc_hr_real = load_adc_npy_for_infer(
                    f,
                    num_high_receiver=opt.high_Azimuth,
                    num_low_receiver=opt.low_Azimuth,
                    lr_receiver_idx=opt.lr_channel,
                    datamode='real'
                )

                adc_lr_cplx, _ = load_adc_npy_for_infer(
                    f,
                    num_high_receiver=opt.high_Azimuth,
                    num_low_receiver=opt.low_Azimuth,
                    lr_receiver_idx=opt.lr_channel,
                    datamode='complex'
                )

                lr_real_list.append(adc_lr_real)
                lr_cplx_list.append(adc_lr_cplx)
                hr_list.append(adc_hr_real)

            adc_lr_real = torch.cat(lr_real_list, 0).to(device)
            adc_lr_cplx = torch.cat(lr_cplx_list, 0).to(device)
            adc_hr = torch.cat(hr_list, 0).to(device)

            if opt.fp16:
                adc_lr_real = adc_lr_real.half()
                adc_lr_cplx = adc_lr_cplx.half()

            # -------- forward --------

            adc_sr_real = netG_real(adc_lr_real)
            adc_sr_cplx = netG_cplx(adc_lr_real)

            adc_lr_real = adc_lr_real.cpu()
            adc_sr_real = adc_sr_real.cpu()
            adc_sr_cplx = adc_sr_cplx.cpu()
            adc_hr = adc_hr.cpu()

            lr_batch = (adc_lr_real[:, 0] + 1j * adc_lr_real[:, 1]).numpy().transpose(0, 2, 1, 3)
            sr_real_batch = (adc_sr_real[:, 0] + 1j * adc_sr_real[:, 1]).numpy().transpose(0, 2, 1, 3)
            sr_cplx_batch = (adc_sr_cplx[:, 0] + 1j * adc_sr_cplx[:, 1]).numpy().transpose(0, 2, 1, 3)
            gt_batch = (adc_hr[:, 0] + 1j * adc_hr[:, 1]).numpy().transpose(0, 2, 1, 3)

            mse_real_batch, psnr_real_batch = compute_mse_psnr(sr_real_batch, gt_batch)
            mse_cplx_batch, psnr_cplx_batch = compute_mse_psnr(sr_cplx_batch, gt_batch)

            for i in range(sr_real_batch.shape[0]):
                lr = lr_batch[i]
                sr_real = sr_real_batch[i]
                sr_cplx = sr_cplx_batch[i]
                gt = gt_batch[i]

                sum_mse_real += float(mse_real_batch[i])
                sum_psnr_real += float(psnr_real_batch[i])
                sum_mse_cplx += float(mse_cplx_batch[i])
                sum_psnr_cplx += float(psnr_cplx_batch[i])
                count += 1

                # ---------- update tqdm ----------
                if count % 5 == 0:
                    pbar.set_postfix({
                        "REAL_PSNR": f"{sum_psnr_real / count:.2f}",
                        "CPLX_PSNR": f"{sum_psnr_cplx / count:.2f}",
                        "REAL_MSE": f"{sum_mse_real / count:.2e}",
                        "CPLX_MSE": f"{sum_mse_cplx / count:.2e}",
                    })

                # ---------- RAD (for visualization) ----------
                rad_lr = RAD_map(lr)
                rad_real = RAD_map(sr_real)
                rad_cplx = RAD_map(sr_cplx)
                rad_gt = RAD_map(gt)

                imgs = [
                    np.abs(rad_lr).max(axis=2),
                    np.abs(rad_real).max(axis=2),
                    np.abs(rad_cplx).max(axis=2),
                    np.abs(rad_gt).max(axis=2)
                ]

                imgs = [20 * np.log10(x + 1e-6) for x in imgs]
                mx = max(x.max() for x in imgs)
                imgs = [(x / mx * 255).astype(np.uint8) for x in imgs]
                imgs = [cv2.applyColorMap(x, cv2.COLORMAP_OCEAN) for x in imgs]

                gap = 20
                h = imgs[0].shape[0]
                space = np.ones((h, gap, 3), dtype=np.uint8) * 255

                compare = np.concatenate(
                    [imgs[0], space, imgs[1], space, imgs[2], space, imgs[3]],
                    axis=1
                )

                save_path = os.path.join(opt.save_dir, f"{names[i]}.png")
                cv2.imwrite(save_path, compare)

    print("\nFinal Results:")
    print(f"REAL  -> PSNR: {sum_psnr_real / count:.2f}, MSE: {sum_mse_real / count:.4e}")
    print(f"CPLX  -> PSNR: {sum_psnr_cplx / count:.2f}, MSE: {sum_mse_cplx / count:.4e}")

    print("Inference finished.")


if __name__ == '__main__':
    main()