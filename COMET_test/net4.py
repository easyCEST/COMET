import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from Nolocal_network import *



class pfrb_sig_block(nn.Module):
    def __init__(self, in_ch=1, ou_ch=1,num_fre=31, num_represent=4, num_base_represent=1):
        super(pfrb_sig_block, self).__init__()
        self.in_ch = in_ch
        self.ou_ch = ou_ch
        self.num_fre = num_fre
        self.num_represent = num_represent
        self.num_base_represent = num_base_represent
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.in_ch, self.num_represent*self.in_ch, kernel_size=3, padding=1, stride=1),   # 每一个fre单独输入
            nn.InstanceNorm2d(self.num_represent*self.in_ch),
            nn.LeakyReLU(True)
        )
        self.con_base = nn.Sequential(
            nn.Conv2d(self.num_fre*self.num_represent*self.in_ch, self.num_represent*self.in_ch*self.num_base_represent, kernel_size=1, padding=0, stride=1),
            nn.InstanceNorm2d(self.num_represent*self.in_ch*self.num_base_represent),
            nn.LeakyReLU(True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.num_represent*self.in_ch*(self.num_base_represent + 1), self.in_ch*self.num_represent, kernel_size=3, padding=1, stride=1),   # 每一个fre单独输入
            nn.InstanceNorm2d(self.in_ch),
            nn.LeakyReLU(True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(self.in_ch*(self.num_represent+1),self.ou_ch, kernel_size=3,
                     padding=1, stride=1),  # 每一个fre单独输入
            nn.InstanceNorm2d(self.ou_ch),
            nn.LeakyReLU(True)
        )
    def forward(self,x):
        B,CH,H,W = x.size()


        x = x.reshape([-1, H,W])
        x = x.unsqueeze(1)   # B*CH / 1 / H / W
        inpu1 = self.conv1(x)  # B*CH / represent / H / w
        base = inpu1.reshape([B, CH, self.num_represent, H ,W])
        base = base.reshape([B,CH*self.num_represent, H, W])

        base = self.con_base(base)  # B / self.num_represent / H / W
        base = base.unsqueeze(1)
        base = torch.repeat_interleave(base, repeats=CH, dim=1)
        base = base.reshape([-1 , self.num_represent, H, W]) # B*CH / self.num_represent / H / W
        base = torch.cat([base, inpu1],dim=1)

        base = self.conv2(base)
        base = torch.cat([base, x],1)            # B*CH / self.num_represent / H / W
        base = self.conv3(base)     # B*OU / 1 / H / W
        base = base.reshape([B, -1, H ,W])
        return base        # N, ou_ch * num_fre, H, W
class pfrb_sig_block_ce(nn.Module):
    def __init__(self, in_ch=1, ou_ch=1,num_fre=31, num_represent=4, num_base_represent=1):
        super(pfrb_sig_block_ce, self).__init__()
        self.in_ch = in_ch

        self.ou_ch = ou_ch
        self.num_fre = num_fre
        self.num_represent = num_represent
        self.num_base_represent = num_base_represent
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.in_ch, self.num_represent*self.in_ch, kernel_size=3, padding=1, stride=1),   # 每一个fre单独输入
            nn.InstanceNorm2d(self.num_represent*self.in_ch),
            nn.LeakyReLU(True)
        )
        self.con_base = nn.Sequential(
            nn.Conv3d(self.num_represent, self.num_represent*2, kernel_size=1, padding=0, stride=1),
            nn.InstanceNorm3d(self.num_represent*2),
            nn.Conv3d(self.num_represent*2, self.num_represent, kernel_size=3, padding=1, stride=1),
            nn.InstanceNorm3d(self.num_represent),
            nn.LeakyReLU(True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(self.num_represent*2, self.in_ch*self.num_represent, kernel_size=3, padding=1, stride=1),   # 每一个fre单独输入
            nn.InstanceNorm3d(self.in_ch*self.num_represent),
            nn.LeakyReLU(True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(self.in_ch*(self.num_represent+1),self.ou_ch, kernel_size=3,
                     padding=1, stride=1),  # 每一个fre单独输入
            nn.InstanceNorm2d(self.ou_ch),
            nn.LeakyReLU(True)
        )
    def forward(self,x):
        B,CH,H,W = x.size()


        x = x.reshape([-1, H,W])
        x = x.unsqueeze(1)   # B*CH / 1 / H / W
        inpu1 = self.conv1(x)  # B*CH / represent / H / w
        base = inpu1.reshape([B, CH, self.num_represent, H ,W])
        base1 = base.permute(0,2,1,3,4)   # B/ represent / CH / H / w
        # base = base.reshape([B,CH*self.num_represent, H, W])

        base = self.con_base(base1)  # B / self.num_represent/ CH / H / W
        # base = base.permute(0, 2, 1, 3, 4)   # B / CH / self.num_represent / H / W

        # base = base.reshape([-1 , self.num_represent, H, W]) # B*CH / self.num_represent / H / W
        base = torch.cat([base, base1],dim=1)   # B / self.num_represent * 2 / CH / H / W

        base = self.conv2(base)  #  B / self.num_represent / CH / H / W
        base = base.permute(0, 2, 1, 3, 4)
        base = base.reshape([-1 , self.num_represent, H, W])
        base = torch.cat([base, x],1)            # B*CH / self.num_represent / H / W
        base = self.conv3(base)     # B*OU / 1 / H / W
        base = base.reshape([B, -1, H ,W])
        return base        # N, ou_ch * num_fre, H, W

class C_CT(nn.Module):
    def __init__(self, num_fre=31, num_base_repre=2, num_represent=2, inter_ch=3):
        super(C_CT, self).__init__()
        self.inter_ch = inter_ch
        self.num_fre = num_fre

        self.layer1_UP =  nn.Sequential(
            pfrb_sig_block(1, self.inter_ch, num_fre=self.num_fre, num_represent=num_represent, num_base_represent=num_base_repre),
            nn.ConvTranspose2d(self.inter_ch*self.num_fre, self.inter_ch*self.num_fre, kernel_size=4, padding=1, stride=2),
            nn.InstanceNorm2d(self.inter_ch*self.num_fre),
            nn.LeakyReLU(True)
        )           ## 上采样
        # self.layer2_UP = nn.Sequential(
        #     pfrb_sig_block(self.inter_ch, self.inter_ch)
        #
        # )
        self.layer2_UP =  nn.Sequential(
            pfrb_sig_block(self.inter_ch*2, self.inter_ch, num_fre=self.num_fre, num_represent=num_represent, num_base_represent=num_base_repre),
            nn.ConvTranspose2d(self.inter_ch * self.num_fre, self.inter_ch * self.num_fre, kernel_size=4, padding=1, stride=2),
            nn.InstanceNorm2d(self.inter_ch * self.num_fre),
            nn.LeakyReLU(True)
        )
        self.layer3_UP = nn.Sequential(
            pfrb_sig_block((self.inter_ch+1), 1, num_fre=self.num_fre, num_represent=num_represent, num_base_represent=num_base_repre)
            # nn.Conv2d(self.inter_ch * self.num_fre, self.inter_ch * self.num_fre, kernel_size=4, padding=1, stride=2),
            # nn.BatchNorm2d(self.inter_ch * self.num_fre),
            # nn.LeakyReLU(True)
        )
        self.layer2_DOWN = nn.Sequential(
            pfrb_sig_block(1, self.inter_ch, num_fre=self.num_fre, num_represent=num_represent,
                           num_base_represent=num_base_repre),
            nn.Conv2d(self.inter_ch * self.num_fre, self.inter_ch * self.num_fre, kernel_size=4, padding=1, stride=2),
            nn.InstanceNorm2d(self.inter_ch * self.num_fre),
            nn.LeakyReLU(True)
        )
        self.layer1_DOWN = nn.Sequential(
            pfrb_sig_block(self.inter_ch, 1, num_fre=self.num_fre, num_represent=num_represent,
                           num_base_represent=num_base_repre),
            nn.Conv2d(1 * self.num_fre, 1 * self.num_fre, kernel_size=4, padding=1, stride=2),
            nn.InstanceNorm2d(1 * self.num_fre),
            nn.LeakyReLU(True)
        )
    def forward(self, Zk, lr_hsi):
        down1 = self.layer2_DOWN(Zk)   #  inter
        e = self.layer1_DOWN(down1)  # 1   lr_hsi

        e = e - lr_hsi
        up1 = self.layer1_UP(e)
        up1 = torch.cat([up1, down1], dim=1) # inter * 2
        up1 = self.layer2_UP(up1)
        up1 = torch.cat([up1, Zk], dim=1)  # inter + 1
        up1 = self.layer3_UP(up1)
        return up1, e

class R(nn.Module):   # 降采过程
    def __init__(self, in_ch, ou_ch):
        super(R, self).__init__()
        self.ly_nonlocal = NLBlock_new(in_channels=in_ch, num_block=4)   # 256/4 256/4
        self.layer =  nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 3, kernel_size=3, padding=1, stride=1),

            nn.Conv2d(in_ch // 3, ou_ch, kernel_size=3, padding=1, stride=1),
            nn.InstanceNorm2d(ou_ch),
            nn.LeakyReLU(True)
        )
    def forward(self,x):
        x = self.ly_nonlocal(x)
        x = self.layer(x)

        return x

class Rt_sig(nn.Module):
    def __init__(self, in_ch, ou_ch, num_block=4):
        super(Rt_sig, self).__init__()
        self.in_ch = in_ch
        self.ou_ch = ou_ch
        self.num_block = num_block
        self.ly_nonlocal = NLBlock_new(in_channels=in_ch, num_block=8)
        self.ly = nn.Sequential(
            pfrb_sig_block(num_fre=256 // 4),
            pfrb_sig_block(num_fre=256 // 4)
        )
        self.ly_out = nn.Sequential(
            nn.Conv2d(in_ch, in_ch*4, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(in_ch*4),
            nn.LeakyReLU(True),
            nn.Conv2d(in_ch*4, ou_ch, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(ou_ch),
            nn.LeakyReLU(True)
        )
    def forward(self,x, H_true):
        B, C, H, W = x.size()
        x = self.ly_nonlocal(x)
        x = space_to_depth(x,self.num_block)
        if H_true:
            x = x.permute(0, 2, 1, 3)
            x = self.ly(x)
            x = x.permute(0, 2, 1, 3)
            x = depth_to_space(x, 4)
            x = self.ly_out(x)

        else:
            x = x.permute(0, 3, 1, 2)       # B / W / SLC/ H
            x = self.ly(x)
            x = x.permute(0, 2, 3, 1)       # B / slc / H / W
            x = depth_to_space(x, self.num_block)
            x = self.ly_out(x)

        return x

class Prox(nn.Module):
    def __init__(self,in_ch, ou_ch):
        super(Prox, self).__init__()
        self.ly1 = pfrb_sig_block(in_ch=1, ou_ch=1)
        self.ly_d1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(in_ch, in_ch*2, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(in_ch*2),
            nn.LeakyReLU(True)
        )
        self.ly_d2 = nn.Sequential(
            nn.Conv2d(in_ch*2, in_ch * 2, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(in_ch * 2, in_ch*4, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(in_ch*4),
            nn.LeakyReLU(True)
        )
        self.ly_u2 = nn.Sequential(
            nn.ConvTranspose2d(in_ch*4, in_ch * 2, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(in_ch * 2, in_ch*2, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(in_ch * 2),
            nn.LeakyReLU(True)
        )
        self.ly_u1 = nn.Sequential(
            nn.ConvTranspose2d(in_ch*4, in_ch * 2, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(in_ch * 2, in_ch*2, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(in_ch * 2),
            nn.LeakyReLU(True)
        )
        self.ly_out = nn.Sequential(
            nn.Conv2d(in_ch*3, ou_ch, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(ou_ch, ou_ch, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(ou_ch ),
            nn.LeakyReLU(True)
        )
    def forward(self,x):
        d1 = self.ly_d1(x)
        d2 = self.ly_d2(d1)
        d2 = self.ly_u2(d2)
        d2 = torch.cat([d2, d1],dim=1)
        d2 = self.ly_u1(d2)
        d2 = torch.cat([d2,x],dim=1)

        d2 = self.ly_out(d2)
        d2 = self.ly1(d2)

        return d2




class OUSC(nn.Module):
    def __init__(self,config):
        super(OUSC, self).__init__()
        self.num_spectral = config.num_spectral
        self.num_slc = config.num_slc
        self.start_Rt = Rt_sig(self.num_slc, self.num_spectral)
        self.Rt = Rt_sig(self.num_slc, self.num_spectral)
        self.R = R(self.num_spectral, self.num_slc)
        self.C_CT = C_CT()
        self.pro = Prox(self.num_spectral, self.num_spectral)
        self.pro_bre = Prox(self.num_spectral, self.num_spectral)
    def forward(self,lr_hsi,hr_msi):
        out = self.start_Rt(hr_msi, True)


        e_hsi1 = self.R(out) - hr_msi
        pro_left = self.Rt(e_hsi1, True)
        pro_right = self.Rt(e_hsi1, False)
        pro_fre, e_msi1 = self.C_CT(out, lr_hsi)
        ###
        pro_left_out = self.pro_bre(pro_left + out)
        pro_right_out = self.pro_bre(pro_right + out)
        pro_fre = self.pro_bre(pro_fre + out)
        ###
        out1 = self.pro(pro_fre + pro_left + pro_right)
        return out1, e_hsi1, e_msi1, pro_left_out, pro_right_out, pro_fre

class OUSC_1new(nn.Module):
    def __init__(self,config):
        super(OUSC_1new, self).__init__()
        self.num_spectral = config.num_spectral
        self.num_slc = config.num_slc
        self.start_Rt = Rt_sig(self.num_slc, self.num_spectral)
        self.Rt = Rt_sig(self.num_slc, self.num_spectral)
        self.R = R(self.num_spectral, self.num_slc)
        self.C_CT = C_CT()
        self.pro = Prox(self.num_spectral, self.num_spectral)
        # self.pro_bre = Prox(self.num_spectral, self.num_spectral)
    def forward(self,lr_hsi,hr_msi):
        out = self.start_Rt(hr_msi, True)


        e_hsi1 = self.R(out) - hr_msi
        pro_left = self.Rt(e_hsi1, True)
        pro_right = self.Rt(e_hsi1, False)
        pro_fre, e_msi1 = self.C_CT(out, lr_hsi)
        ###
        pro_left_out = self.pro(pro_left + out)
        pro_right_out = self.pro(pro_right + out)
        pro_fre = self.pro(pro_fre + out)
        ###
        out1 = self.pro((pro_fre + pro_left)/2 + pro_right)
        return out1, e_hsi1, e_msi1, pro_left_out, pro_right_out, pro_fre











