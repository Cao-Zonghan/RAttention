import argparse
import os
import cv2
import numpy as np
import pandas as pd
import pytorch_ssim
import torch
import util.helper as helper
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_utils import TrainDatasetFromFolder_radar3D_adc, ValDatasetFromFolder_radar3D_adc, display_transform, TrainRadarADCChannelDataset, ValRadarADCChannelDataset
from loss import GeneratorLoss, GeneratorLoss_L1
from model import Generator_radar3D, UNet_3D, Generator_radar3D_adc, Generator_radar2D_adc_complex
from math import log10
from inference import adc_to_image
from model_complex_fast import Generator_radar2D_adc_complex_fast

parser = argparse.ArgumentParser(description='Train Super Resolution Models')
parser.add_argument('--crop_size', default=128, type=int, help='training images crop size')
parser.add_argument('--upscale_factor', default=1, type=int,
                    help='super resolution upscale factor')
parser.add_argument('--num_epochs', default=50, type=int, help='train epoch number')
parser.add_argument('--batch_size', type=int, default=16, help='id to focus on')
parser.add_argument('--model_name', default=None, type=str, help='generator model epoch name')
parser.add_argument('--datamode', default="real", type=str, help='choose to use real or complex')
parser.add_argument('--lr', default=0.0002, type=float, help='learning rate')
parser.add_argument('--low_Azimuth', type=int, default=4, help='lr number of Azimuth')
parser.add_argument('--high_Azimuth', type=int, default=12, help='hr number of Azimuth')
parser.add_argument('--lr_channel', type=int, default=[0, 2, 8, 10], help='choose Datacube downsample channel')
parser.add_argument('--run_mode', default='train', choices=['train', 'infer', 'test'], help='train or inference')
parser.add_argument('--npy_path', type=str, default=None, help='raw adc npy file for inference')
parser.add_argument('--save_dir', type=str, default='./infer_out', help='directory to save inference images')
parser.add_argument('--output', metavar='DIR', default='./out',
                    help='path to output folder. If not set, will be created in data folder')
parser.add_argument('--train_data', metavar='DIR', default='./train',
                    help='path to output folder. If not set, will be created in data folder') 
parser.add_argument('--val_data', metavar='DIR', default='./test',
                    help='path to output folder. If not set, will be created in data folder')  


# mode = 'extend'
# mode = 'lap-extend'
# mode = 'eval-ssr'
# mode = 'else'

# mode = 'extend'
# mode = 'extend2'
mode = 'normal'
# mode = 'extend3'

num_low_receiver = 4

num_angle_bins = 256
# num_angle_bins = 128

# debug = True
debug = False
abs_vis = True
tune_voc = False
beg_voc = 40

def RAD_map(range_plot, use_gpu=True):
    if torch.is_tensor(range_plot):
        xp = torch
        tensor = range_plot if torch.is_complex(range_plot) else torch.as_tensor(range_plot)
        device = tensor.device

        if use_gpu and torch.cuda.is_available() and device.type != 'cuda':
            tensor = tensor.to('cuda', non_blocking=True)
            device = tensor.device

        range_plot_fft = xp.fft.fft(tensor, dim=0)
        range_doppler = xp.fft.fft(range_plot_fft, dim=2)
        range_doppler = xp.fft.fftshift(range_doppler, dim=2)

        angle_pad = max(0, num_angle_bins - range_doppler.shape[1])
        if angle_pad > 0:
            pad_shape = list(range_doppler.shape)
            pad_shape[1] = angle_pad
            pad_tensor = xp.zeros(*pad_shape, dtype=range_doppler.dtype, device=device)
            range_azimuth = xp.cat((range_doppler, pad_tensor), dim=1)
        else:
            range_azimuth = range_doppler[:, :num_angle_bins, :]

        range_azimuth = xp.fft.fft(range_azimuth, dim=1)
        range_azimuth = xp.fft.fftshift(range_azimuth, dim=1)
        out_img = xp.rot90(range_azimuth, 2, dims=(0, 1))
        return out_img

    if use_gpu and torch.cuda.is_available():
        tensor = torch.as_tensor(range_plot)
        if not torch.is_complex(tensor) and np.iscomplexobj(range_plot):
            tensor = tensor.to(torch.complex64 if range_plot.dtype == np.complex64 else torch.complex128)
        return RAD_map(tensor, use_gpu=True)

    range_plot = np.fft.fft(range_plot, axis=0)
    range_doppler = np.fft.fft(range_plot, axis=2)
    range_doppler = np.fft.fftshift(range_doppler, axes=2)
    angle_pad = max(0, num_angle_bins - range_doppler.shape[1])
    if angle_pad > 0:
        padding = ((0, 0), (0, angle_pad), (0, 0))
        range_azimuth = np.pad(range_doppler, padding, mode='constant')
    else:
        range_azimuth = range_doppler[:, :num_angle_bins, :]

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

    norm = gt_mag.max().item() + eps
    pred_norm = pred_mag / norm
    gt_norm = gt_mag / norm

    batch_mse = torch.mean((pred_norm - gt_norm) ** 2).item()
    batch_mse = float(np.real(batch_mse))
    batch_psnr = float(np.real(10 * np.log10(1.0 / (batch_mse + eps))))

    return batch_mse, batch_psnr

if __name__ == '__main__':


    opt = parser.parse_args()
    
    CROP_SIZE = opt.crop_size
    UPSCALE_FACTOR = opt.upscale_factor
    NUM_EPOCHS = opt.num_epochs
    out_path = opt.output
    batch_size = opt.batch_size
    datamode = opt.datamode

    train_data_adc_dir = opt.train_data
    val_data_adc_dir = opt.val_data

    num_low_receiver = opt.low_Azimuth
    num_high_receiver = opt.high_Azimuth

    lr_channel = opt.lr_channel

    print("Inupt train dir",train_data_adc_dir)
    print("Inupt test dir",val_data_adc_dir)

    if datamode == 'real':
        train_set = TrainDatasetFromFolder_radar3D_adc(train_data_adc_dir, num_low_receiver, None, None, crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR,\
            index_list=1, num_high_receiver=num_high_receiver)
        val_set = ValDatasetFromFolder_radar3D_adc(val_data_adc_dir, num_low_receiver, None, None, upscale_factor=UPSCALE_FACTOR, index_list=1,\
            num_high_receiver=num_high_receiver)
    elif datamode == 'complex':
        train_set = TrainRadarADCChannelDataset(train_data_adc_dir, num_high_receiver, lr_receiver_idx = lr_channel, datamode=datamode)
        val_set = ValRadarADCChannelDataset(val_data_adc_dir, num_high_receiver, lr_receiver_idx = lr_channel, datamode=datamode)

    train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=16, shuffle=False)
    
    # netG = Generator_radar3D(UPSCALE_FACTOR)
    UPSCALE_FACTOR = opt.high_Azimuth/num_low_receiver
    if datamode == "real":
        netG =Generator_radar3D_adc(UPSCALE_FACTOR,input_dim=2)
    elif datamode == "complex":
        # netG = Generator_radar2D_adc_complex(UPSCALE_FACTOR,input_dim=4)
        netG = Generator_radar2D_adc_complex_fast(UPSCALE_FACTOR,input_dim=4)
    # netG = UNet_3D()

    if opt.model_name is not None:
        netG.load_state_dict(torch.load(opt.model_name))
        print("Pre-trained weights loaded")

    print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
    # netD = Discriminator_radar()
    # print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))
    
    generator_criterion = GeneratorLoss(datamode)
    # generator_criterion = GeneratorLoss_L1()
    
    if torch.cuda.is_available():
        netG.cuda()
        # netD.cuda()
        generator_criterion.cuda()

    optimizerG = optim.Adam(netG.parameters())
    # optimizerD = optim.Adam(netD.parameters())
    
    results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}
    best_psnr = float('-inf')
    best_epoch = None
    best_model_path = None

    # ---------------- Inference mode ----------------
    if opt.run_mode == 'infer':
        assert opt.npy_path is not None, "Please provide --npy_path"

        if not os.path.exists(opt.save_dir):
            os.makedirs(opt.save_dir)

        # ---------- build model (same as training) ----------
        UPSCALE_FACTOR = opt.high_Azimuth / opt.low_Azimuth
        datamode = opt.datamode

        if datamode == "real":
            netG = Generator_radar3D_adc(UPSCALE_FACTOR, input_dim=2)
        elif datamode == "complex":
            # netG = Generator_radar2D_adc_complex(UPSCALE_FACTOR, input_dim=4)
            netG = Generator_radar2D_adc_complex_fast(UPSCALE_FACTOR, input_dim=4)
        assert opt.model_name is not None, "Inference requires --model_name"
        netG.load_state_dict(torch.load(opt.model_name, map_location='cpu'))
        print("Loaded model:", opt.model_name)

        if torch.cuda.is_available():
            netG.cuda()

        netG.eval()

        # ---------- load npy ----------
        # assume shape: [2, A, R, D] or compatible
        adc_lr = np.load(opt.npy_path)
        adc_lr = torch.from_numpy(adc_lr).float().unsqueeze(0)

        if torch.cuda.is_available():
            adc_lr = adc_lr.cuda()

        # ---------- forward ----------
        with torch.no_grad():
            adc_sr = netG(adc_lr)

        # ---------- visualization ----------
        # reuse inference.py logic
        img, rad, ra = adc_to_image(adc_sr[0], abs_vis=True)

        save_path = os.path.join(opt.save_dir, "sr.png")
        cv2.imwrite(save_path, (img * 255).astype(np.uint8)[:, :, ::-1])

        print("Saved SR image to:", save_path)
        exit(0)

    for epoch in range(1, NUM_EPOCHS + 1):
        train_bar = tqdm(train_loader)
        running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}
    
        netG.train()
        # netD.train()
        for adc_data_low, adc in train_bar:
            if debug:
                break

            # import pdb
            # pdb.set_trace()
            g_update_first = True
            batch_size = batch_size
            running_results['batch_sizes'] += batch_size

            ############################
            # (1) Update D network: maximize D(x)-1-D(G(z))
            ###########################



            # real_img = Variable(target)
            # adc_data_low = Variable()
            # if torch.cuda.is_available():
            #     real_img = real_img.cuda()

            z = Variable(adc_data_low)
            y = Variable(adc)

            if torch.cuda.is_available():
                z = z.cuda()
                y = y.cuda()
            # fake_img = netG(z)
            ############################
            # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
            ###########################
            netG.zero_grad()
            ## The two lines below are added to prevent runetime error in Google Colab ##
            # fake_img = netG(z)
            fake_adc = netG(z)
            # fake_out = netD(fake_img).mean()
            fake_out=None
            ##
            # import pdb
            # pdb.set_trace()
            g_loss = generator_criterion(fake_out, fake_adc, y)
            g_loss.backward()

            # fake_img = netG(z)
            # fake_out = netD(fake_img).mean()


            optimizerG.step()

            # loss for current batch before optimization
            running_results['g_loss'] += g_loss.item() * batch_size
            # running_results['g2_loss'] += g2_loss.item() * batch_size
            # running_results['d_loss'] += d_loss.item() * batch_size
            # running_results['d_score'] += real_out.item() * batch_size
            # running_results['g_score'] += fake_out.item() * batch_size

            desp = '[%d/%d] Loss_G1: %.4f' % (epoch, NUM_EPOCHS,
                running_results['g_loss'] / running_results['batch_sizes'])
            train_bar.set_description((desp))

        netG.eval()

        if not os.path.exists(out_path):
            os.makedirs(out_path)

        with torch.no_grad():
            val_bar = tqdm(
                val_loader,
                leave=True,
                dynamic_ncols=False,
                position=0
            )
            # 删除了 val_images = []，因为不再需要保存过程图片
            valing_results = {'mse': 0, 'psnr': 0, 'batch_sizes': 0}

            # 移除了计数器 i，如果逻辑中不需要它的话
            for adc_lr, adc_hr in val_bar:
                # 建议：如果 val_loader 的 batch_size 并非固定为 16，这里最好用 adc_lr.size(0) 动态获取
                batch_size = adc_lr.size(0)
                valing_results['batch_sizes'] += batch_size

                if torch.cuda.is_available():
                    adc_lr = adc_lr.cuda()
                    adc_hr = adc_hr.cuda()

                # --- 模型推理 ---
                adc_sr = netG(adc_lr)
                # --- 在信号域上直接计算指标（参考 create_RAMap.py 的归一化方式） ---
                eps = 1e-12
                batch_mse, batch_psnr = compute_signal_domain_metrics(adc_sr, adc_hr, eps=eps)

                valing_results['mse'] += batch_mse * batch_size
                valing_results['mse_eval'] = float(np.real(valing_results['mse'] / valing_results['batch_sizes']))
                valing_results['psnr'] = float(np.real(10 * np.log10(1.0 / (valing_results['mse_eval'] + eps))))
                valing_results['batch_psnr'] = batch_psnr

                # adc_sr = to_complex_if_needed(adc_sr)
                # rad_sr = RAD_map(adc_sr)
                #
                # if tune_voc:
                #     rad_sr = rad_sr[:, :, beg_voc:]

                # 注意：此处删除了 helper.getLog 和 helper.norm2Image 等可视化代码

                val_bar.set_postfix_str(
                    f"mse={valing_results['mse_eval']:.6e}, "
                    f"psnr={valing_results['psnr']:.4f} dB, "
                    f"batch_psnr={valing_results['batch_psnr']:.4f} dB"
                )
                # if i>=5:
                #     break
               

            # import pdb
            # pdb.set_trace()
            # len_val = (len(val_images)//15)*15
            # val_images = val_images[:len_val]
            # val_images = torch.stack(val_images)
            # val_images = torch.chunk(val_images, val_images.size(0) // 15)
            # val_save_bar = tqdm(val_images, desc='[saving training results]')
            # index = 1
            # for image in val_save_bar:
            #     # import pdb
            #     # pdb.set_trace()
            #     image = utils.make_grid(image, nrow=3, padding=5)
            #     utils.save_image(image, out_path + 'epoch_%d_index_%d_%.4f.png' % (epoch, index, valing_results['mse_eval']), padding=5)
            #     index += 1
            #
        # save model parameters (only keep the best checkpoint judged by batch PSNR)
        current_psnr = valing_results['batch_psnr']
        if current_psnr > best_psnr:
            if best_model_path is not None and os.path.exists(best_model_path):
                os.remove(best_model_path)
            best_psnr = current_psnr
            best_epoch = epoch
            best_model_path = os.path.join(out_path, 'netG_best_epoch_%d_batch_psnr_%.4f.pth' % (epoch, current_psnr))
            torch.save(netG.state_dict(), best_model_path)
            print('Saved best model at epoch %d with batch PSNR %.4f dB -> %s' % (best_epoch, best_psnr, best_model_path))

        # save loss\scores\psnr\ssim
        # results['d_loss'].append(running_results['d_loss'] / running_results['batch_sizes'])
        results['g_loss'].append(running_results['g_loss'] / running_results['batch_sizes'])
        # results['d_score'].append(running_results['d_score'] / running_results['batch_sizes'])
        results['g_score'].append(running_results['g_score'] / running_results['batch_sizes'])
        results['psnr'].append(valing_results['psnr'])
        # results['ssim'].append(valing_results['ssim'])

        stats_path = os.path.join(out_path, 'train_results.txt')
        with open(stats_path, 'w', encoding='utf-8') as f:
            for idx in range(len(results['g_loss'])):
                f.write(
                    'Epoch %d: Loss_G=%.6f, PSNR=%.4f dB\n' % (
                        idx + 1,
                        results['g_loss'][idx],
                        results['psnr'][idx]
                    )
                )
            if best_epoch is not None:
                f.write('\nBest epoch: %d\n' % best_epoch)
                f.write('Best batch PSNR: %.4f dB\n' % best_psnr)
                f.write('Best model: %s\n' % best_model_path)
    
        # if (epoch+1) % 10 == 0 and epoch != 0:
        #     out_path = 'statistics/'
        #     data_frame = pd.DataFrame(
        #         data={'Loss_D': results['d_loss'], 'Loss_G': results['g_loss'], 'Score_D': results['d_score'],
        #               'Score_G': results['g_score'], 'PSNR': results['psnr'], 'SSIM': results['ssim']},
        #         index=range(1, epoch + 1))
        #     data_frame.to_csv(out_path + 'srf_' + str(UPSCALE_FACTOR) + '_train_results.csv', index_label='Epoch')
