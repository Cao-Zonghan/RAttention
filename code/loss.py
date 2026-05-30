import torch
from torch import nn
from torchvision.models.vgg import vgg16

class ComplexMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, target):

        loss_real = self.mse(pred.real, target.real)
        loss_imag = self.mse(pred.imag, target.imag)

        loss = loss_real + loss_imag

        return loss.real



class GeneratorLoss(nn.Module):
    def __init__(self, datamode):
        super(GeneratorLoss, self).__init__()
        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
        self.complex_mse_loss = ComplexMSELoss()
        self.tv_loss = TVLoss()
        self.datamode = datamode

    def forward(self, out_labels, out_images, target_images):
        # Adversarial Loss
        # adversarial_loss = torch.mean(1 - out_labels)
        # Perception Loss
        # perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))
        # Image Loss
        if self.datamode == "real":
            image_loss = self.mse_loss(out_images, target_images)
        elif self.datamode == "complex":
            image_loss = self.complex_mse_loss(out_images, target_images)
        # TV Loss
        tv_loss = self.tv_loss(out_images)
        # return image_loss + 0.001 * adversarial_loss + 0.006 * perception_loss + 2e-8 * tv_loss
        # return image_loss + 0.001 * adversarial_loss + 2e-8 * tv_loss
        return image_loss + 2e-8 * tv_loss

class GeneratorLoss_L1(nn.Module):
    def __init__(self):
        super(GeneratorLoss_L1, self).__init__()
        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.L1Loss()
        self.tv_loss = TVLoss()

    def forward(self, out_labels, out_images, target_images):
        # Adversarial Loss
        # adversarial_loss = torch.mean(1 - out_labels)
        # Perception Loss
        # perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))
        # Image Loss
        image_loss = self.mse_loss(out_images, target_images)
        # TV Loss
        tv_loss = self.tv_loss(out_images)
        # return image_loss + 0.001 * adversarial_loss + 0.006 * perception_loss + 2e-8 * tv_loss
        # return image_loss + 0.001 * adversarial_loss + 2e-8 * tv_loss
        return image_loss + 2e-8 * tv_loss

class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super().__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):

        batch_size = x.size(0)
        h_x = x.size(2)
        w_x = x.size(3)

        dh = x[:, :, 1:, :] - x[:, :, :h_x - 1, :]
        dw = x[:, :, :, 1:] - x[:, :, :, :w_x - 1]

        # complex magnitude squared
        h_tv = torch.abs(dh).pow(2).sum()
        w_tv = torch.abs(dw).pow(2).sum()

        count_h = dh.numel()
        count_w = dw.numel()

        loss = self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

        return loss.real



if __name__ == "__main__":
    g_loss = GeneratorLoss()
    print(g_loss)
