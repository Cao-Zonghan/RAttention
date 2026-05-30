from os import listdir
from os.path import join
import tempfile

from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize
import numpy as np
import torch
from glob import glob
import util.helper as helper


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def is_numpy_file(filename):
    return any(filename.endswith(extension) for extension in ['.npy'])


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def train_hr_transform(crop_size):
    return Compose([
        # RandomCrop(crop_size),
        ToTensor(),
    ])


def train_lr_transform(crop_size, upscale_factor):
    return Compose([
        ToPILImage(),
        Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
        ToTensor()
    ])


def display_transform():
    return Compose([
        ToPILImage(),
        Resize(400),
        CenterCrop(400),
        ToTensor()
    ])


def RAD_map(range_plot):
    range_plot = np.fft.fft(range_plot, axis=0)
    range_doppler = np.fft.fft(range_plot, axis=2)
    range_doppler = np.fft.fftshift(range_doppler, axes=2)
    padding = ((0, 0), (0, num_angle_bins - range_doppler.shape[1]), (0, 0))
    range_azimuth = np.pad(range_doppler, padding, mode='constant')

    # import pdb
    # pdb.set_trace()
    range_azimuth = np.fft.fft(range_azimuth, axis=1)
    range_azimuth = np.fft.fftshift(range_azimuth, axes=1)
    out_img = np.rot90(range_azimuth, 2, axes=(0, 1))
    # out_img = range_azimuth
    return out_img


class TrainDatasetFromFolder(Dataset):
    def __init__(self, hr_data_dir, lr_data_dir, crop_size, upscale_factor):
        super(TrainDatasetFromFolder, self).__init__()
        self.hr_filenames = [join(hr_data_dir, x) for x in sorted(listdir(hr_data_dir)) if is_image_file(x)]
        self.lr_filenames = [join(lr_data_dir, x) for x in sorted(listdir(lr_data_dir)) if is_image_file(x)]
        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = train_hr_transform(crop_size)
        # self.lr_transform = train_lr_transform(crop_size, upscale_factor)

    def __getitem__(self, index):
        hr_image = self.hr_transform(Image.open(self.hr_filenames[index]).convert('RGB'))
        lr_image = self.hr_transform(Image.open(self.lr_filenames[index]).convert('RGB'))
        # lr_image = self.lr_transform(hr_image)
        return lr_image, hr_image

    def __len__(self):
        return len(self.hr_filenames)


def complexTo2Channels(target_array):
    """ transfer complex a + bi to [a, b]"""
    # assert target_array.dtype == np.complex64
    ### NOTE: transfer complex to (magnitude) ###
    output_array = getMagnitude(target_array)
    output_array = getLog(output_array)
    return output_array


def getMagnitude(target_array, power_order=1):
    """ get magnitude out of complex number """
    target_array = np.abs(target_array)
    target_array = pow(target_array, power_order)
    return target_array


def getLog(target_array, scalar=1., log_10=True):
    """ get Log values """
    if log_10:
        return scalar * np.log10(target_array + 1.)
    else:
        return target_array


class TrainDatasetFromFolder_radar3D(Dataset):
    def __init__(self, hr_data_dir, lr_data_dir, crop_size, upscale_factor, index_list=None):
        super(TrainDatasetFromFolder_radar3D, self).__init__()
        if index_list != None:
            self.hr_filenames = [x for x in sorted(glob(hr_data_dir)) if is_numpy_file(x)]
            self.lr_filenames = [x for x in sorted(glob(lr_data_dir)) if is_numpy_file(x)]
            print(len(self.hr_filenames))
            print(len(self.lr_filenames))
            # import pdb
            # pdb.set_trace()
        else:
            self.hr_filenames = [join(hr_data_dir, x) for x in sorted(listdir(hr_data_dir)) if is_numpy_file(x)]
            self.lr_filenames = [join(lr_data_dir, x) for x in sorted(listdir(lr_data_dir)) if is_numpy_file(x)]

        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = train_hr_transform(crop_size)
        # self.lr_transform = train_lr_transform(crop_size, upscale_factor)

    def __getitem__(self, index):

        # import pdb
        # pdb.set_trace()

        # hr_image = torch.from_numpy(np.log2(np.load(self.hr_filenames[index]).transpose(2,0,1))).type(torch.FloatTensor)
        # lr_image = torch.from_numpy(np.log2(np.load(self.lr_filenames[index]).transpose(2,0,1))).type(torch.FloatTensor)

        # hr_image = hr_image.unsqueeze(0)/13-1
        # lr_image = lr_image.unsqueeze(0)/13-1

        hr_data = complexTo2Channels(np.load(self.hr_filenames[index]))
        hr_image = torch.from_numpy(hr_data.transpose(2, 0, 1)).type(torch.FloatTensor) / 10

        lr_data = complexTo2Channels(np.load(self.lr_filenames[index]))
        lr_image = torch.from_numpy(lr_data.transpose(2, 0, 1)).type(torch.FloatTensor) / 10

        return lr_image[None], hr_image[None]

    def __len__(self):
        return len(self.hr_filenames)


class TrainDatasetFromFolder_radar2D(Dataset):
    def __init__(self, hr_data_dir, lr_data_dir, crop_size, upscale_factor, index_list=None):
        super().__init__()
        if index_list != None:
            self.hr_filenames = [x for x in sorted(glob(hr_data_dir)) if is_numpy_file(x)]
            self.lr_filenames = [x for x in sorted(glob(lr_data_dir)) if is_numpy_file(x)]
            print(len(self.hr_filenames))
            print(len(self.lr_filenames))
            # import pdb
            # pdb.set_trace()
        else:
            self.hr_filenames = [join(hr_data_dir, x) for x in sorted(listdir(hr_data_dir)) if is_numpy_file(x)]
            self.lr_filenames = [join(lr_data_dir, x) for x in sorted(listdir(lr_data_dir)) if is_numpy_file(x)]

        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = train_hr_transform(crop_size)
        # self.lr_transform = train_lr_transform(crop_size, upscale_factor)

    def __getitem__(self, index):
        # hr_data = complexTo2Channels(np.load(self.hr_filenames[index]))
        # hr_image = torch.from_numpy(hr_data.transpose(2,0,1)).type(torch.FloatTensor)/10

        # import pdb
        # pdb.set_trace()
        hr_data = helper.getLog(helper.getSumDim(helper.getMagnitude(np.load(self.hr_filenames[index]) \
                                                                     , power_order=1), target_axis=-1), scalar=10,
                                log_10=True)
        # print(hr_data.shape)
        hr_image = torch.from_numpy(hr_data).type(torch.FloatTensor) / 10

        lr_data = helper.getLog(helper.getSumDim(helper.getMagnitude(np.load(self.lr_filenames[index]) \
                                                                     , power_order=1), target_axis=-1), scalar=10,
                                log_10=True)
        lr_image = torch.from_numpy(lr_data).type(torch.FloatTensor) / 10

        return lr_image[None], hr_image[None]

    def __len__(self):
        return len(self.hr_filenames)


class ValDatasetFromFolder_radar2D(Dataset):
    def __init__(self, hr_data_dir, lr_data_dir, upscale_factor, index_list=None):
        super().__init__()
        self.upscale_factor = upscale_factor
        # self.image_filenames = [join(hr_data_dir, x) for x in listdir(hr_data_dir) if is_image_file(x)]
        if index_list != None:
            self.hr_filenames = [x for x in sorted(glob(hr_data_dir)) if is_numpy_file(x)]
            self.lr_filenames = [x for x in sorted(glob(lr_data_dir)) if is_numpy_file(x)]

        else:
            self.hr_filenames = [join(hr_data_dir, x) for x in sorted(listdir(hr_data_dir)) if is_numpy_file(x)]
            self.lr_filenames = [join(lr_data_dir, x) for x in sorted(listdir(lr_data_dir)) if is_numpy_file(x)]

        # import pdb
        # pdb.set_trace()
        # self.hr_filenames = [join(hr_data_dir, x) for x in sorted(listdir(hr_data_dir)) if is_numpy_file(x)]
        # self.lr_filenames = [join(lr_data_dir, x) for x in sorted(listdir(lr_data_dir)) if is_numpy_file(x)]

    def __getitem__(self, index):

        # hr_data = complexTo2Channels(np.load(self.hr_filenames[index]))
        # hr_image = torch.from_numpy(hr_data.transpose(2,0,1)).type(torch.FloatTensor)/10

        # lr_data = complexTo2Channels(np.load(self.lr_filenames[index]))
        # lr_image = torch.from_numpy(lr_data.transpose(2,0,1)).type(torch.FloatTensor)/10

        hr_data = helper.getLog(helper.getSumDim(helper.getMagnitude(np.load(self.hr_filenames[index]) \
                                                                     , power_order=1), target_axis=-1), scalar=10,
                                log_10=True)
        # print(hr_data.shape)
        hr_image = torch.from_numpy(hr_data).type(torch.FloatTensor) / 10

        lr_data = helper.getLog(helper.getSumDim(helper.getMagnitude(np.load(self.lr_filenames[index]) \
                                                                     , power_order=1), target_axis=-1), scalar=10,
                                log_10=True)
        lr_image = torch.from_numpy(lr_data).type(torch.FloatTensor) / 10

        return (lr_image[None]).type(torch.FloatTensor), (lr_image[None]).type(torch.FloatTensor), (
        hr_image[None]).type(torch.FloatTensor)

    def __len__(self):
        return len(self.hr_filenames)


class TrainDatasetFromFolder_radar3D_adc(Dataset):
    def __init__(self, adc_data_dir, num_low_receiver, hr_data_dir, lr_data_dir, crop_size, upscale_factor,
                 index_list=None, num_high_receiver=12, lr_receiver_idx=None):
        super().__init__()
        if index_list != None:  # Use this function for now
            self.adc_filenames = [x for x in sorted(glob(adc_data_dir + '*')) if is_numpy_file(x)]
            # self.hr_filenames = [x for x in sorted(glob(hr_data_dir)) if is_numpy_file(x)]
            # self.lr_filenames = [x for x in sorted(glob(lr_data_dir)) if is_numpy_file(x)]
            print(len(self.adc_filenames))
            # print(len(self.lr_filenames))
            # import pdb
            # pdb.set_trace()
        else:
            raise NotImplementedError
            self.hr_filenames = [join(hr_data_dir, x) for x in sorted(listdir(hr_data_dir)) if is_numpy_file(x)]
            self.lr_filenames = [join(lr_data_dir, x) for x in sorted(listdir(lr_data_dir)) if is_numpy_file(x)]

        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = train_hr_transform(crop_size)
        self.num_low_receiver = num_low_receiver
        self.num_high_receiver = num_high_receiver
        self.lr_receiver_idx = lr_receiver_idx
        # self.lr_transform = train_lr_transform(crop_size, upscale_factor)

    def _select_lr_channels(self, adc_data_high):
        if self.lr_receiver_idx is None:
            cut_l = (adc_data_high.shape[1] - self.num_low_receiver) // 2
            return adc_data_high[:, cut_l:cut_l + self.num_low_receiver, :, :]

        if isinstance(self.lr_receiver_idx, int):
            return adc_data_high[:, :self.lr_receiver_idx, :, :]

        idx = torch.as_tensor(self.lr_receiver_idx, dtype=torch.long)
        return adc_data_high.index_select(dim=1, index=idx)

    def __getitem__(self, index):

        numpy_adc = np.load(self.adc_filenames[index])
        # print(numpy_adc[0,0])
        adc_data = torch.from_numpy(numpy_adc)  # D A R
        adc_data = adc_data.permute(2, 0, 1)  # R D A
        # R D A 2
        adc_data = torch.view_as_real(adc_data).permute(3, 2, 0, 1).type(torch.FloatTensor) / 100  # devide by 100
        # 2 A R D

        cut_l = (adc_data.shape[1] - self.num_high_receiver) // 2
        # print()
        adc_data_high = adc_data[:, cut_l:cut_l + self.num_high_receiver, :, :]
        adc_data_low = self._select_lr_channels(adc_data_high)
        # print(adc_data_low.shape)

        # hr_data = complexTo2Channels(np.load(self.hr_filenames[index]))
        # hr_image = torch.from_numpy(hr_data.transpose(2,0,1)).type(torch.FloatTensor)/10

        # lr_data = complexTo2Channels(np.load(self.lr_filenames[index]))
        # lr_image = torch.from_numpy(lr_data.transpose(2,0,1)).type(torch.FloatTensor)/10

        return adc_data_low, adc_data_high
        # return adc_data_high, adc_data_low, lr_image[None], hr_image[None]
        # return adc_data, adc_data_low, lr_image[None], hr_image[None]

    def __len__(self):
        return len(self.adc_filenames)


class TrainRadarADCChannelDataset(Dataset):

    def __init__(
            self,
            adc_data_dir,
            num_high_receiver,
            lr_receiver_idx=None,
            scale_factor=100,
            datamode='real'  # [新增]
    ):
        """
        Args:
            adc_data_dir (str): ADC .npy 数据目录
            num_high_receiver (int): HR 接收通道数
            lr_receiver_idx (None | int | list | np.ndarray):
            scale_factor (float): 数据归一化因子
            datamode (str): 'real' (默认) 或 'complex'
        """
        super().__init__()

        self.adc_filenames = sorted([
            x for x in glob(adc_data_dir + '*')
            if is_numpy_file(x)
        ])

        self.num_high_receiver = num_high_receiver
        self.lr_receiver_idx = lr_receiver_idx
        self.scale_factor = scale_factor
        self.datamode = datamode  # [新增]

    def _select_lr_channels(self, adc_data):
        """
        根据 lr_receiver_idx 选择 LR 通道
        real:    [2, A, R, D] -> dim 1
        complex: [A, R, D]    -> dim 0
        """
        # [新增] 确定通道维度：real在第1维，complex在第0维
        dim = 1 if self.datamode == 'real' else 0
        A = adc_data.shape[dim]

        # 情况 1：默认居中裁剪
        if self.lr_receiver_idx is None:
            num_lr = A // 2
            start = (A - num_lr) // 2
            # 根据维度切片
            if self.datamode == 'real':
                return adc_data[:, start:start + num_lr]
            else:
                return adc_data[start:start + num_lr]

        # 情况 2：整数 → 前 N 个通道
        if isinstance(self.lr_receiver_idx, int):
            if self.datamode == 'real':
                return adc_data[:, :self.lr_receiver_idx]
            else:
                return adc_data[:self.lr_receiver_idx]

        # 情况 3：显式索引
        idx = torch.as_tensor(self.lr_receiver_idx, dtype=torch.long)
        return adc_data.index_select(dim=dim, index=idx)

    def __getitem__(self, index):
        adc = np.load(self.adc_filenames[index])
        adc = torch.from_numpy(adc)  # [D, A, R]

        if self.datamode == 'real':
            # === Real Mode ===
            # [D, A, R] → [2, A, R, D]
            adc = adc.permute(2, 0, 1)
            adc = torch.view_as_real(adc)
            adc = adc.permute(3, 2, 0, 1).float() / self.scale_factor

            # 裁剪 HR (A在第1维)
            A = adc.shape[1]
            start = (A - self.num_high_receiver) // 2
            adc_hr = adc[:, start:start + self.num_high_receiver]

        else:
            # === Complex Mode ===
            # [D, A, R] → [A, R, D]
            adc = adc.permute(1, 2, 0) / self.scale_factor
            adc = adc.to(torch.complex64)

            # 裁剪 HR (A在第0维)
            A = adc.shape[0]
            start = (A - self.num_high_receiver) // 2
            adc_hr = adc[start:start + self.num_high_receiver]

        adc_lr = self._select_lr_channels(adc_hr)

        return adc_lr, adc_hr

    def __len__(self):
        return len(self.adc_filenames)


class ValDatasetFromFolder_radar3D(Dataset):
    def __init__(self, hr_data_dir, lr_data_dir, upscale_factor, index_list=None):
        super().__init__()
        self.upscale_factor = upscale_factor
        # self.image_filenames = [join(hr_data_dir, x) for x in listdir(hr_data_dir) if is_image_file(x)]
        if index_list != None:
            self.hr_filenames = [x for x in sorted(glob(hr_data_dir)) if is_numpy_file(x)]
            self.lr_filenames = [x for x in sorted(glob(lr_data_dir)) if is_numpy_file(x)]

        else:
            self.hr_filenames = [join(hr_data_dir, x) for x in sorted(listdir(hr_data_dir)) if is_numpy_file(x)]
            self.lr_filenames = [join(lr_data_dir, x) for x in sorted(listdir(lr_data_dir)) if is_numpy_file(x)]

        # import pdb
        # pdb.set_trace()
        # self.hr_filenames = [join(hr_data_dir, x) for x in sorted(listdir(hr_data_dir)) if is_numpy_file(x)]
        # self.lr_filenames = [join(lr_data_dir, x) for x in sorted(listdir(lr_data_dir)) if is_numpy_file(x)]

    def __getitem__(self, index):

        hr_data = (np.load(self.hr_filenames[index]))
        lr_data = (np.load(self.lr_filenames[index]))

        hr_data = complexTo2Channels(hr_data)
        hr_image = torch.from_numpy(hr_data.transpose(2, 0, 1)).type(torch.FloatTensor) / 10

        lr_data = complexTo2Channels(lr_data)
        lr_image = torch.from_numpy(lr_data.transpose(2, 0, 1)).type(torch.FloatTensor) / 10

        return (lr_image[None]).type(torch.FloatTensor), (lr_image[None]).type(torch.FloatTensor), (
        hr_image[None]).type(torch.FloatTensor)

    def __len__(self):
        return len(self.hr_filenames)


class ValRadarADCChannelDataset(Dataset):

    def __init__(
            self,
            adc_data_dir,
            num_high_receiver,
            lr_receiver_idx=None,
            scale_factor=100,
            datamode='real'  # [新增]
    ):
        """
        Args:
            adc_data_dir (str): ADC .npy 数据目录
            num_high_receiver (int): HR 接收通道数
            lr_receiver_idx (None | int | list | np.ndarray):
            scale_factor (float): 数据归一化因子
            datamode (str): 'real' (默认) 或 'complex'
        """
        super().__init__()

        self.adc_filenames = sorted([
            x for x in glob(adc_data_dir + '*')
            if is_numpy_file(x)
        ])

        self.num_high_receiver = num_high_receiver
        self.lr_receiver_idx = lr_receiver_idx
        self.scale_factor = scale_factor
        self.datamode = datamode  # [新增]

    def _select_lr_channels(self, adc_data):
        """
        根据 lr_receiver_idx 选择 LR 通道
        real:    [2, A, R, D] -> dim 1
        complex: [A, R, D]    -> dim 0
        """
        # [新增] 确定通道维度：real在第1维，complex在第0维
        dim = 1 if self.datamode == 'real' else 0
        A = adc_data.shape[dim]

        # 情况 1：默认居中裁剪
        if self.lr_receiver_idx is None:
            num_lr = A // 2
            start = (A - num_lr) // 2
            # 根据维度切片
            if self.datamode == 'real':
                return adc_data[:, start:start + num_lr]
            else:
                return adc_data[start:start + num_lr]

        # 情况 2：整数 → 前 N 个通道
        if isinstance(self.lr_receiver_idx, int):
            if self.datamode == 'real':
                return adc_data[:, :self.lr_receiver_idx]
            else:
                return adc_data[:self.lr_receiver_idx]

        # 情况 3：显式索引
        idx = torch.as_tensor(self.lr_receiver_idx, dtype=torch.long)
        return adc_data.index_select(dim=dim, index=idx)

    def __getitem__(self, index):
        adc = np.load(self.adc_filenames[index])
        adc = torch.from_numpy(adc)  # [D, A, R]

        if self.datamode == 'real':
            # === Real Mode ===
            # [D, A, R] → [2, A, R, D]
            adc = adc.permute(2, 0, 1)
            adc = torch.view_as_real(adc)
            adc = adc.permute(3, 2, 0, 1).float() / self.scale_factor

            # 裁剪 HR (A在第1维)
            A = adc.shape[1]
            start = (A - self.num_high_receiver) // 2
            adc_hr = adc[:, start:start + self.num_high_receiver]

        else:
            # === Complex Mode ===
            # [D, A, R] → [A, R, D]
            adc = adc.permute(1, 2, 0) / self.scale_factor
            adc = adc.to(torch.complex64)

            # 裁剪 HR (A在第0维)
            A = adc.shape[0]
            start = (A - self.num_high_receiver) // 2
            adc_hr = adc[start:start + self.num_high_receiver]

        adc_lr = self._select_lr_channels(adc_hr)

        return adc_lr, adc_hr

    def __len__(self):
        return len(self.adc_filenames)


class ValDatasetFromFolder_radar3D_adc(Dataset):
    def __init__(self, adc_data_dir, num_low_receiver, hr_data_dir, lr_data_dir, crop_size, upscale_factor,
                 index_list=None, num_high_receiver=12, lr_receiver_idx=None):
        super().__init__()
        if index_list != None:  # Use this function for now
            self.adc_filenames = [x for x in sorted(glob(adc_data_dir + '*')) if is_numpy_file(x)]
            # self.hr_filenames = [x for x in sorted(glob(hr_data_dir)) if is_numpy_file(x)]
            # self.lr_filenames = [x for x in sorted(glob(lr_data_dir)) if is_numpy_file(x)]
            print(len(self.adc_filenames))
            # print(len(self.lr_filenames))
            # import pdb
            # pdb.set_trace()
        else:
            raise NotImplementedError
            self.hr_filenames = [join(hr_data_dir, x) for x in sorted(listdir(hr_data_dir)) if is_numpy_file(x)]
            self.lr_filenames = [join(lr_data_dir, x) for x in sorted(listdir(lr_data_dir)) if is_numpy_file(x)]

        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = train_hr_transform(crop_size)
        self.num_low_receiver = num_low_receiver
        self.num_high_receiver = num_high_receiver
        self.lr_receiver_idx = lr_receiver_idx
        # self.lr_transform = train_lr_transform(crop_size, upscale_factor)

    def _select_lr_channels(self, adc_data_high):
        if self.lr_receiver_idx is None:
            cut_l = (adc_data_high.shape[1] - self.num_low_receiver) // 2
            return adc_data_high[:, cut_l:cut_l + self.num_low_receiver, :, :]

        if isinstance(self.lr_receiver_idx, int):
            return adc_data_high[:, :self.lr_receiver_idx, :, :]

        idx = torch.as_tensor(self.lr_receiver_idx, dtype=torch.long)
        return adc_data_high.index_select(dim=1, index=idx)

    def __getitem__(self, index):

        numpy_adc = np.load(self.adc_filenames[index])
        # print(numpy_adc[0,0])
        adc_data = torch.from_numpy(numpy_adc)  # D A R
        adc_data = adc_data.permute(2, 0, 1)  # R D A
        # R D A 2
        adc_data = torch.view_as_real(adc_data).permute(3, 2, 0, 1).type(torch.FloatTensor) / 100  # devide by 100
        # 2 A R D

        cut_l = (adc_data.shape[1] - self.num_high_receiver) // 2
        # print()
        adc_data_high = adc_data[:, cut_l:cut_l + self.num_high_receiver, :, :]
        adc_data_low = self._select_lr_channels(adc_data_high)
        # print(adc_data_low.shape)

        # hr_data = complexTo2Channels(np.load(self.hr_filenames[index]))
        # hr_image = torch.from_numpy(hr_data.transpose(2,0,1)).type(torch.FloatTensor)/10

        # lr_data = complexTo2Channels(np.load(self.lr_filenames[index]))
        # lr_image = torch.from_numpy(lr_data.transpose(2,0,1)).type(torch.FloatTensor)/10

        return adc_data_low, adc_data_high
        # return adc_data_high, adc_data_low, lr_image[None], hr_image[None]
        # return adc_data, adc_data_low, lr_image[None], hr_image[None]

    def __len__(self):
        return len(self.adc_filenames)


def test_radar3d_adc_lr_channel_selection():
    """
    测试 TrainDatasetFromFolder_radar3D_adc 和 ValDatasetFromFolder_radar3D_adc
    是否能够根据 lr_channel（对应 lr_receiver_idx）抽取指定接收通道。

    测试样本构造方式：
    - 原始 ADC 的每个接收通道都填充为同一个常数；
    - 第 a 个接收通道的实部和虚部都填充值 a * 100；
    - 经过数据集内部除以 100 后，网络张量中该接收通道的代表值就是 a。

    这样只要修改 test_lr_channels，直接运行当前文件，
    就能从打印结果看出最终抽取到了哪些接收通道。
    """

    def _build_expected_tensors(numpy_adc, num_high_receiver, lr_channel):
        adc_data = torch.from_numpy(numpy_adc)
        adc_data = adc_data.permute(2, 0, 1)
        adc_data = torch.view_as_real(adc_data).permute(3, 2, 0, 1).type(torch.FloatTensor) / 100
        cut_l = (adc_data.shape[1] - num_high_receiver) // 2
        adc_data_high = adc_data[:, cut_l:cut_l + num_high_receiver, :, :]
        idx = torch.as_tensor(lr_channel, dtype=torch.long)
        adc_data_low = adc_data_high.index_select(dim=1, index=idx)
        return adc_data_low, adc_data_high

    def _extract_receiver_values(adc_tensor):
        return [int(round(value)) for value in adc_tensor[0, :, 0, 0].tolist()]

    with tempfile.TemporaryDirectory() as temp_dir:
        adc_path = join(temp_dir, 'sample.npy')
        num_doppler, num_receiver, num_range = 2, 16, 3
        numpy_adc = np.zeros((num_doppler, num_receiver, num_range), dtype=np.complex64)

        for a in range(num_receiver):
            channel_value = float(a * 100)
            numpy_adc[:, a, :] = np.complex64(channel_value + 1j * channel_value)

        np.save(adc_path, numpy_adc)

        adc_data_dir = join(temp_dir, '')
        num_high_receiver = 12

        # 直接修改这里，即可测试不同 lr_channel 的抽取结果。
        test_lr_channels = [
            [0, 2, 8, 10],
            [1, 3, 5, 11],
        ]

        for dataset_cls in [TrainDatasetFromFolder_radar3D_adc, ValDatasetFromFolder_radar3D_adc]:
            print('\n' + '=' * 80)
            print(f'测试数据集: {dataset_cls.__name__}')
            sampled_low_tensors = []

            for lr_channel in test_lr_channels:
                dataset = dataset_cls(
                    adc_data_dir=adc_data_dir,
                    num_low_receiver=len(lr_channel),
                    hr_data_dir=None,
                    lr_data_dir=None,
                    crop_size=1,
                    upscale_factor=1,
                    index_list=1,
                    num_high_receiver=num_high_receiver,
                    lr_receiver_idx=lr_channel,
                )

                adc_data_low, adc_data_high = dataset[0]
                expected_low, expected_high = _build_expected_tensors(
                    numpy_adc=numpy_adc,
                    num_high_receiver=num_high_receiver,
                    lr_channel=lr_channel,
                )

                assert adc_data_high.shape == expected_high.shape, \
                    f'{dataset_cls.__name__} high shape mismatch: {adc_data_high.shape} vs {expected_high.shape}'
                assert adc_data_low.shape == expected_low.shape, \
                    f'{dataset_cls.__name__} low shape mismatch: {adc_data_low.shape} vs {expected_low.shape}'
                assert torch.allclose(adc_data_high, expected_high), \
                    f'{dataset_cls.__name__} high receiver crop mismatch when lr_channel={lr_channel}'
                assert torch.allclose(adc_data_low, expected_low), \
                    f'{dataset_cls.__name__} low receiver selection mismatch when lr_channel={lr_channel}'

                high_values = _extract_receiver_values(adc_data_high)
                low_values = _extract_receiver_values(adc_data_low)

                print(f'num_high_receiver = {num_high_receiver}')
                print(f'HR 居中裁剪后的接收通道代表值: {high_values}')
                print(f'设置 lr_channel = {lr_channel}')
                print(f'抽取得到的 LR 接收通道代表值: {low_values}')

                sampled_low_tensors.append(adc_data_low.clone())

            assert not torch.allclose(sampled_low_tensors[0], sampled_low_tensors[1]), \
                f'{dataset_cls.__name__} returns the same tensor for different lr_channel settings'

        print('\n所有断言校验通过。')

    return 'TrainDatasetFromFolder_radar3D_adc and ValDatasetFromFolder_radar3D_adc lr_channel test passed.'


class TrainDatasetFromFolder_radar(Dataset):
    def __init__(self, hr_data_dir, lr_data_dir, crop_size, upscale_factor):
        super(TrainDatasetFromFolder_radar, self).__init__()
        self.hr_filenames = [join(hr_data_dir, x) for x in sorted(listdir(hr_data_dir)) if is_numpy_file(x)]
        self.lr_filenames = [join(lr_data_dir, x) for x in sorted(listdir(lr_data_dir)) if is_numpy_file(x)]
        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = train_hr_transform(crop_size)
        # self.lr_transform = train_lr_transform(crop_size, upscale_factor)

    def __getitem__(self, index):
        # import pdb
        # pdb.set_trace()
        # hr_image = torch.from_numpy(np.load(self.hr_filenames[index]).transpose(2,0,1)).type(torch.FloatTensor)
        # lr_image = torch.from_numpy(np.load(self.lr_filenames[index]).transpose(2,0,1)).type(torch.FloatTensor)
        hr_image = torch.from_numpy(np.log2(np.load(self.hr_filenames[index]).transpose(2, 0, 1))).type(
            torch.FloatTensor)
        lr_image = torch.from_numpy(np.log2(np.load(self.lr_filenames[index]).transpose(2, 0, 1))).type(
            torch.FloatTensor)
        # hr_image = (hr_image*10-140)*9/255
        # lr_image = (lr_image*10-140)*9/255
        # hr_image = (hr_image*10-100)*7/255
        # lr_image = (lr_image*10-100)*7/255
        # hr_image = hr_image*10/255
        # lr_image = lr_image*10/255
        # hr_image = np.clip(hr_image*13,0,255)/255
        # lr_image = np.clip(lr_image*13,0,255)/255
        # hr_image = np.clip((hr_image-14)*10,-255,255)/255
        # lr_image = np.clip((lr_image-14)*10,-255,255)/255
        # out_img = np.clip(out_img,0,255)
        # lr_image = self.lr_transform(hr_image)
        hr_image = hr_image / 13 - 1
        lr_image = lr_image / 13 - 1
        return lr_image, hr_image

    def __len__(self):
        return len(self.hr_filenames)


class ValDatasetFromFolder_radar(Dataset):
    def __init__(self, hr_data_dir, lr_data_dir, upscale_factor):
        super(ValDatasetFromFolder_radar, self).__init__()
        self.upscale_factor = upscale_factor
        # self.image_filenames = [join(hr_data_dir, x) for x in listdir(hr_data_dir) if is_image_file(x)]
        self.hr_filenames = [join(hr_data_dir, x) for x in sorted(listdir(hr_data_dir)) if is_numpy_file(x)]
        self.lr_filenames = [join(lr_data_dir, x) for x in sorted(listdir(lr_data_dir)) if is_numpy_file(x)]

    def __getitem__(self, index):
        # hr_image = Image.open(self.image_filenames[index])
        # hr_image = torch.from_numpy(np.load(self.hr_filenames[index]).transpose(2,0,1))
        # lr_image = torch.from_numpy(np.load(self.lr_filenames[index]).transpose(2,0,1))

        hr_image_ = torch.from_numpy(np.log2(np.load(self.hr_filenames[index]).transpose(2, 0, 1)))
        lr_image_ = torch.from_numpy(np.log2(np.load(self.lr_filenames[index]).transpose(2, 0, 1)))
        # hr_image = (hr_image*10)*9/255
        # lr_image = (lr_image*10)*9/255
        # hr_image = hr_image*10/255
        # lr_image = lr_image*10/255
        # hr_image = np.clip(hr_image*13,0,255)/255
        # lr_image = np.clip(lr_image*13,0,255)/255
        # hr_image = np.clip((hr_image-14)*10,-255,255)/255
        # lr_image = np.clip((lr_image-14)*10,-255,255)/255
        hr_image = hr_image_ / 13 - 1
        lr_image = lr_image_ / 13 - 1

        # w, h = hr_image.size
        # crop_size = calculate_valid_crop_size(min(w, h), self.upscale_factor)
        # lr_scale = Resize(crop_size // self.upscale_factor, interpolation=Image.BICUBIC)
        # hr_scale = Resize(crop_size, interpolation=Image.BICUBIC)

        # hr_image = CenterCrop(crop_size)(hr_image)
        # lr_image = CenterCrop(crop_size)(lr_image)
        # lr_image = lr_scale(hr_image)

        # hr_restore_img = hr_scale(lr_image)
        return (lr_image).type(torch.FloatTensor), (lr_image_).type(torch.FloatTensor), (hr_image_).type(
            torch.FloatTensor)

    def __len__(self):
        return len(self.hr_filenames)


class ValDatasetFromFolder(Dataset):
    def __init__(self, hr_data_dir, lr_data_dir, upscale_factor):
        super(ValDatasetFromFolder, self).__init__()
        self.upscale_factor = upscale_factor
        # self.image_filenames = [join(hr_data_dir, x) for x in listdir(hr_data_dir) if is_image_file(x)]
        self.hr_filenames = [join(hr_data_dir, x) for x in sorted(listdir(hr_data_dir)) if is_image_file(x)]
        self.lr_filenames = [join(lr_data_dir, x) for x in sorted(listdir(lr_data_dir)) if is_image_file(x)]

    def __getitem__(self, index):
        # hr_image = Image.open(self.image_filenames[index])
        hr_image = Image.open(self.hr_filenames[index]).convert('RGB')
        lr_image = Image.open(self.lr_filenames[index]).convert('RGB')
        w, h = hr_image.size
        crop_size = calculate_valid_crop_size(min(w, h), self.upscale_factor)
        lr_scale = Resize(crop_size // self.upscale_factor, interpolation=Image.BICUBIC)
        hr_scale = Resize(crop_size, interpolation=Image.BICUBIC)

        hr_image = CenterCrop(crop_size)(hr_image)
        lr_image = CenterCrop(crop_size)(lr_image)
        lr_image = lr_scale(hr_image)

        hr_restore_img = hr_scale(lr_image)
        return ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

    def __len__(self):
        return len(self.hr_filenames)


# class ValDatasetFromFolder(Dataset):
#     def __init__(self, dataset_dir, upscale_factor):
#         super(ValDatasetFromFolder, self).__init__()
#         self.upscale_factor = upscale_factor
#         self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]

#     def __getitem__(self, index):
#         hr_image = Image.open(self.image_filenames[index])
#         w, h = hr_image.size
#         crop_size = calculate_valid_crop_size(min(w, h), self.upscale_factor)
#         lr_scale = Resize(crop_size // self.upscale_factor, interpolation=Image.BICUBIC)
#         hr_scale = Resize(crop_size, interpolation=Image.BICUBIC)
#         hr_image = CenterCrop(crop_size)(hr_image)
#         lr_image = lr_scale(hr_image)
#         hr_restore_img = hr_scale(lr_image)
#         return ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

#     def __len__(self):
#         return len(self.image_filenames)

class TestDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(TestDatasetFromFolder, self).__init__()
        self.lr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/data/'
        self.hr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/target/'
        self.upscale_factor = upscale_factor
        self.lr_filenames = [join(self.lr_path, x) for x in listdir(self.lr_path) if is_image_file(x)]
        self.hr_filenames = [join(self.hr_path, x) for x in listdir(self.hr_path) if is_image_file(x)]

    def __getitem__(self, index):
        image_name = self.lr_filenames[index].split('/')[-1]
        lr_image = Image.open(self.lr_filenames[index])
        w, h = lr_image.size
        hr_image = Image.open(self.hr_filenames[index])
        hr_scale = Resize((self.upscale_factor * h, self.upscale_factor * w), interpolation=Image.BICUBIC)
        hr_restore_img = hr_scale(lr_image)
        return image_name, ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

    def __len__(self):
        return len(self.lr_filenames)


if __name__ == '__main__':
    print(test_radar3d_adc_lr_channel_selection())
