import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from net4 import *
from Nolocal_network import *


class Ca_attention(nn.Module):
    def __init__(self, in_ch, num_spectral):
        super(Ca_attention, self).__init__()
        self.layer1 = nn.Conv2d(in_channels=in_ch, out_channels=1, kernel_size=1, stride=1)
        self.acti1 = nn.LeakyReLU()
        self.layer2 = nn.Conv2d(in_channels=1, out_channels=num_spectral, kernel_size=1, stride=1)
        self.acti2 = nn.Sigmoid()

    def forward(self, x):
        x = torch.mean(x, dim=[2, 3], keepdim=True)  # batch_size/bands/img_w/img_h
        ca = self.layer1(x)
        ca = self.acti1(ca)
        ca = self.layer2(ca)
        ca = self.acti2(ca)
        return ca


class Sa_attention(nn.Module):
    def __init__(self, in_ch, kernal):
        super(Sa_attention, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=1, kernel_size=kernal, stride=1, padding=int(kernal / 2)),
            nn.Sigmoid(),
            # nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernal, stride=1),
            # nn.Sigmoid()
        )

    def forward(self, x):
        x = torch.mean(x, dim=[1], keepdim=True)  # 在1bands维平均
        sa = self.layer(x)
        return sa

class Block_Resunet_2d(nn.Module):
    def __init__(self, in_ch, ou_ch):
        super(Block_Resunet_2d, self).__init__()
        self.in_ch = in_ch
        self.ou_ch = ou_ch
        self.encoder1_a = nn.Sequential(nn.Conv2d(in_channels=self.in_ch,out_channels=self.in_ch,kernel_size=5, padding=2, stride=1),    # 256*256*3
                                        nn.BatchNorm2d(self.in_ch)
                                      )
        self.encoder1_b = nn.Sequential(nn.Conv2d(self.in_ch, self.in_ch*2,kernel_size=4,padding=1,stride=2),      # 128*128*6
                                      nn.BatchNorm2d(self.in_ch*2),
                                      nn.LeakyReLU(True))
        self.encoder2_a = nn.Sequential(
            nn.Conv2d(self.in_ch*2,self.in_ch*2, kernel_size=3, padding=1, stride=1),                              # 128*128*6
            nn.BatchNorm2d(self.in_ch * 2)
        )
        self.encoder2_b = nn.Sequential(
            nn.Conv2d(self.in_ch * 2, self.in_ch * 4, kernel_size=4, padding=1, stride=2),                         # 64*64*12
            nn.BatchNorm2d(self.in_ch * 4),
            nn.LeakyReLU(True)
        )
        # self.encoder3_a = nn.Sequential(nn.Conv2d(self.in_ch*4,self.in_ch*4,kernel_size=3,padding=1,stride=1),
        #                                 nn.BatchNorm2d(self.in_ch*4)
        #                                 )
        # self.encoder3_b = nn.Sequential(
        #     nn.Conv2d(self.in_ch * 4, self.in_ch * 8, kernel_size=4, padding=1, stride=2),
        #     nn.BatchNorm2d(self.in_ch * 8),
        #     nn.LeakyReLU(True)
        # )

        self.decoder1_a = nn.Sequential(nn.Conv2d(self.in_ch*8,self.in_ch*4,kernel_size=3,padding=1,stride=1),
                     nn.BatchNorm2d(self.in_ch*4)
                     )
        self.decoder1_b = nn.Sequential(nn.ConvTranspose2d(self.in_ch*4,self.in_ch*4,kernel_size=4,padding=1, stride=2),
                                        nn.BatchNorm2d(self.in_ch*4),
                                        nn.LeakyReLU(True)
                                        )
        self.decoder2_a = nn.Sequential(
            nn.Conv2d(self.in_ch * 6, self.in_ch * 4, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(self.in_ch * 4)
        )
        self.decoder2_b = nn.Sequential(
            nn.ConvTranspose2d(self.in_ch*4, self.in_ch*2,kernel_size=4, padding=1,stride=2),
            nn.BatchNorm2d(self.in_ch * 2),
            nn.LeakyReLU(True)
        )
        self.decoder3_a = nn.Sequential(
            nn.Conv2d(self.in_ch * 3, self.in_ch * 2, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(self.in_ch * 2)
        )
        self.decoder3_b = nn.Sequential(
            nn.ConvTranspose2d(self.in_ch*2, self.ou_ch,kernel_size=3, padding=1,stride=1),
            nn.BatchNorm2d(self.ou_ch),
            nn.LeakyReLU(True)
        )


        self.Bottneck = nn.Sequential(
            nn.Conv2d(self.in_ch*4,self.in_ch*8,kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(self.in_ch*8),
            nn.LeakyReLU(True),
            nn.Conv2d(self.in_ch * 8, self.in_ch * 4, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(self.in_ch * 4),
            nn.LeakyReLU(True)
        )
    def forward(self, x):                            # 256 * 256 * 3
        en1_a = self.encoder1_a(x)                   # 256 * 256 * 3
        en1_b = self.encoder1_b(en1_a + x)           # 128 * 128 * 6

        en2_a = self.encoder2_a(en1_b)               # 128 * 128 * 6
        en2_b = self.encoder2_b(en2_a+ en1_b)        # 64  * 64  * 12
        # en3_a = self.encoder3_a(en2_b)
        # en3_b = self.encoder3_b(en3_a+en2_b)
        en4 = self.Bottneck(en2_b)                   # 64  * 64  * 12
        de1 = torch.concat([en4, en2_b], dim=1)      # 64  * 64  * 24
        de1_a = self.decoder1_a(de1)                 # 64  * 64  * 12
        de1_b = self.decoder1_b(en4 + de1_a)         # 128 * 128 * 12
        de2_a = self.decoder2_a(torch.concat([en1_b, de1_b],dim=1))  #  128 * 128 * 18  --- 128 * 128 * 12
        de2_b = self.decoder2_b(de1_b + de2_a)                       #  256 * 256 * 6
        de3_a = self.decoder3_a(torch.concat([x, de2_b],dim=1))      #  256 * 256 * 9   --- 256 * 256 * 6
        de3_b = self.decoder3_b(de2_b + de3_a)                       #  256 * 256 * out
        # de4_a = self.decoder4_a(torch.concat([x, de3_b], dim=1))
        # de4_b = self.decoder4_b(de3_b + de4_a)
        # de5 = self.decoder5(de4_b)
        return de3_b  # de5

class Decoder_Resnet_3d(nn.Module):   ####
    def __init__(self,kernel):     #BATCH/ CH / S / H / W
        super(Decoder_Resnet_3d, self).__init__()
        self.ker = kernel  # (3 / 1 / 1)
        self.d1= nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=1,kernel_size=self.ker, padding=(1,0,0), stride=1),
            nn.BatchNorm3d(1),
            nn.LeakyReLU(True)
        )
        self.d2= nn.Sequential(
            nn.ConvTranspose3d(in_channels=1, out_channels=2, kernel_size=(1, 4, 4), padding=(0,1,1),stride=(1,2,2)),
            nn.BatchNorm3d(2),
            nn.LeakyReLU(True)
        )
        self.d3 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=2, out_channels=4,kernel_size=(1, 4, 4), padding=(0,1,1), stride=(1,2,2)),
            # nn.ConvTranspose3d(in_channels=2, out_channels=4, kernel_size=self.ker, padding=(1, 0, 0), stride=1),
            nn.BatchNorm3d(4),
            nn.LeakyReLU(True)
        )


        self.d7 = nn.Sequential(

            nn.Conv3d(in_channels=4, out_channels=2, kernel_size=self.ker, padding=(1, 0, 0), stride=1),
            nn.BatchNorm3d(2),
            nn.LeakyReLU(True),
            nn.Conv3d(in_channels=2, out_channels=1, kernel_size=self.ker, padding=(1, 0, 0), stride=1),
            nn.BatchNorm3d(1),
            nn.LeakyReLU(True)
        )

    def forward(self, x):

        d1 = self.d1(x)
        d2 = self.d2(x + d1)
        d2 = self.d3(d2)
        d2 = self.d7(d2)
        return d2

class Decoder_Resnet_3d_down(nn.Module):   ####
    def __init__(self,kernel):     #BATCH/ CH / S / H / W
        super(Decoder_Resnet_3d_down, self).__init__()
        self.ker = kernel  # (3 / 1 / 1)
        self.d1= nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=1,kernel_size=self.ker, padding=(1,0,0), stride=1),
            nn.BatchNorm3d(1),
            nn.LeakyReLU(True)
        )
        self.d2= nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=2, kernel_size=(1, 4, 4), padding=(0,1,1),stride=(1,2,2)),
            nn.BatchNorm3d(2),
            nn.LeakyReLU(True)
        )
        self.d3 = nn.Sequential(
            nn.Conv3d(in_channels=2, out_channels=4,kernel_size=(1, 4, 4), padding=(0,1,1), stride=(1,2,2)),
            # nn.Conv3d(in_channels=2, out_channels=4, kernel_size=self.ker, padding=(1, 0, 0), stride=1),
            nn.BatchNorm3d(4),
            nn.LeakyReLU(True)
        )
        #### 这里只适用于8倍
        self.d7 = nn.Sequential(
            nn.Conv3d(in_channels=4, out_channels=2, kernel_size=self.ker, padding=(1, 0, 0), stride=1),
            nn.BatchNorm3d(2),
            nn.LeakyReLU(True),
            nn.Conv3d(in_channels=2, out_channels=1, kernel_size=self.ker, padding=(1, 0, 0), stride=1),
            nn.BatchNorm3d(1),
            nn.LeakyReLU(True)
        )
    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(x + d1)
        d2 = self.d3(d2)
        d2 = self.d7(d2)
        return d2





class OUSC_net_2_full_CNN(nn.Module):
    def __init__(self, config):
        super(OUSC_net_2_full_CNN, self).__init__()
        self.num_spectral = config.num_spectral
        self.num_slc = config.num_slc
        self.Res2d_start = nn.Sequential(
            NLBlock_new(in_channels=self.num_slc),
            Block_Resunet_2d(in_ch=self.num_slc, ou_ch=self.num_spectral)
        )
        self.Res2d_s_up = nn.Sequential(
            NLBlock_new(in_channels=self.num_slc),
            Block_Resunet_2d(in_ch=self.num_slc, ou_ch=self.num_spectral)
        )
        self.ce_xiang = pfrb_sig_block(in_ch=1, ou_ch=1, num_fre=256)
        self.C_1 = pfrb_sig_block()
        # self.Ct = pfrb_sig_block()
        self.C_2 = Decoder_Resnet_3d_down((3, 1, 1))
        self.Ct = pfrb_sig_block(in_ch=1, ou_ch=1, num_fre=256)
        self.CNNlayer = nn.Conv2d(in_channels=self.num_spectral, out_channels=self.num_spectral, stride=1, kernel_size=3, padding=1)
        self.Res3d_up = nn.Sequential(
            Decoder_Resnet_3d((3, 1, 1))
        )
        self.pro = Block_Resunet_2d(in_ch=self.num_spectral, ou_ch=self.num_spectral)
        self.Down1 = nn.Sequential(
            NLBlock_new(in_channels=self.num_spectral, num_block=2),
            nn.Conv2d(in_channels=self.num_spectral, out_channels=self.num_slc, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.num_slc),
            nn.LeakyReLU(True)
        )
        self.out = Block_Resunet_2d(in_ch=self.num_spectral + self.num_slc, ou_ch=self.num_spectral)

    def forward(self, lr_hsi, hr_msi):
        out = self.Res2d_start(hr_msi)

        e_hsi1 = self.Down1(out) - hr_msi
        pro_R_1 = self.Res2d_s_up(e_hsi1)

        pro_left_1 = pro_R_1.permute(0, 2, 1, 3)  # B / H / slc / W
        pro_left_1 = self.ce_xiang(pro_left_1)
        pro_left_1 = pro_left_1.permute(0, 2, 1, 3)

        pro_right_1 = pro_R_1.permute(0, 3, 1, 2)  # B / W / slc / H
        pro_right_1 = self.ce_xiang(pro_right_1)
        pro_right_1 = pro_right_1.permute(0, 2, 3, 1)

        e_msi1 = self.C_1(out)
        e_msi1_1 = e_msi1

        e_msi1 = e_msi1.permute(0, 2, 1, 3)
        e_msi1 = self.Ct(e_msi1)
        e_msi1 = e_msi1.permute(0, 2, 1, 3)
        e_msi1 = self.CNNlayer(e_msi1)

        e_msi1_2 = ((e_msi1 * 0.1 + e_msi1_1)).permute(0, 3, 1, 2)
        e_msi1 = self.Ct(e_msi1_2)
        e_msi1 = ((e_msi1 * 0.1 + e_msi1_2)).permute(0, 2, 3, 1)
        e_msi1 = self.CNNlayer(e_msi1)


        e_msi1 = torch.squeeze(self.C_2(0.1 * e_msi1.unsqueeze(1) + e_msi1_1.unsqueeze(1)), 1)
        e_msi1 = e_msi1 - lr_hsi
        pre_pro_1 = torch.squeeze(self.Res3d_up(e_msi1.unsqueeze(1)), 1)
        out1 = self.pro((pre_pro_1 + pro_right_1 + pro_left_1) / 3 + out)

        #####
        e_hsi2 = self.Down1(out1) - hr_msi
        pro_R_2 = self.Res2d_s_up(e_hsi2)

        pro_left_2 = pro_R_2.permute(0, 2, 1, 3)  # B / H / slc / W
        pro_left_r = self.ce_xiang(pro_left_2)
        pro_left_r = pro_left_r.permute(0, 2, 1, 3)
        pro_left_r = self.pro(pro_left_r + out1)
        pro_left_2 = pro_left_2.permute(0, 2, 1, 3)

        pro_right_2 = pro_R_2.permute(0, 3, 1, 2)  # B / W / slc / H
        pro_right_r = self.ce_xiang(pro_right_2)
        pro_right_r = pro_right_r.permute(0, 2, 3, 1)  #
        pro_right_r = self.pro(pro_right_r + out1)
        pro_right_2 = pro_right_2.permute(0, 2, 3, 1)

        e_msi2 = self.C_1(out1)

        e_msi2_1 = e_msi2

        e_msi2 = e_msi2.permute(0, 2, 1, 3)
        e_msi2 = self.Ct(e_msi2)
        e_msi2 = e_msi2.permute(0, 2, 1, 3)
        e_msi2 = self.CNNlayer(e_msi2)

        e_msi2_2 = ((e_msi2 * 0.1 + e_msi2_1)).permute(0, 3, 1, 2)
        e_msi2 = self.Ct(e_msi2_2)
        e_msi2 = ((e_msi2 * 0.1 + e_msi2_2)).permute(0, 2, 3, 1)
        e_msi2 = self.CNNlayer(e_msi2)




        e_msi2 = torch.squeeze(self.C_2(0.1 * e_msi2.unsqueeze(1) + e_msi2_1.unsqueeze(1)), 1)
        e_msi2 = e_msi2 - lr_hsi
        pre_pro_2 = torch.squeeze(self.Res3d_up(e_msi2.unsqueeze(1)), 1)
        pre_pro_r = self.pro(pre_pro_2 + out1)
        out2 = self.pro((pre_pro_2 + pro_right_2 + pro_left_2) / 3 + out1)

        out = self.out(torch.cat([out2, hr_msi], dim=1))

        return out, out1, out2, e_hsi1, e_msi1, e_hsi2, e_msi2, pro_left_r, pro_right_r, pre_pro_r










class OUSC_net_2_full(nn.Module):
    def __init__(self, config):
        super(OUSC_net_2_full, self).__init__()
        self.num_spectral = config.num_spectral
        self.num_slc = config.num_slc
        self.Res2d_start = nn.Sequential(
            NLBlock_new(in_channels=self.num_slc),
            Block_Resunet_2d(in_ch=self.num_slc, ou_ch=self.num_spectral)
        )
        self.Res2d_s_up = nn.Sequential(
                NLBlock_new(in_channels=self.num_slc),
                Block_Resunet_2d(in_ch=self.num_slc, ou_ch=self.num_spectral)
            )
        self.ce_xiang = pfrb_sig_block(in_ch=1,ou_ch=1,num_fre=256)
        self.C_1 = pfrb_sig_block()
        # self.Ct = pfrb_sig_block()
        self.C_2 = Decoder_Resnet_3d_down((3, 1, 1))
        self.Ct = pfrb_sig_block(in_ch=1,ou_ch=1,num_fre=256)
        self.Res3d_up = nn.Sequential(
            Decoder_Resnet_3d((3, 1, 1))
        )
        self.pro = Block_Resunet_2d(in_ch=self.num_spectral, ou_ch=self.num_spectral)
        self.Down1 = nn.Sequential(
            NLBlock_new(in_channels=self.num_spectral, num_block=2),
            nn.Conv2d(in_channels=self.num_spectral, out_channels=self.num_slc, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.num_slc),
            nn.LeakyReLU(True)
        )
        self.out = Block_Resunet_2d(in_ch=self.num_spectral+self.num_slc, ou_ch=self.num_spectral)
    def forward(self, lr_hsi, hr_msi):
        out = self.Res2d_start(hr_msi)

        e_hsi1 = self.Down1(out) - hr_msi
        pro_R_1 = self.Res2d_s_up(e_hsi1)


        pro_left_1 = pro_R_1.permute(0, 2, 1, 3)  # B / H / slc / W
        pro_left_1 = self.ce_xiang(pro_left_1)
        pro_left_1 = pro_left_1.permute(0, 2, 1, 3)

        pro_right_1 = pro_R_1.permute(0, 3, 1, 2)  # B / W / slc / H
        pro_right_1 = self.ce_xiang(pro_right_1)
        pro_right_1 = pro_right_1.permute(0, 2, 3, 1)

        e_msi1 = self.C_1(out)
        e_msi1_1 = e_msi1


        e_msi1 = e_msi1.permute(0,2,1,3)
        e_msi1 = self.Ct(e_msi1)
        e_msi1 = e_msi1.permute(0,2,1,3)

        e_msi1_2 = ((e_msi1*0.1 + e_msi1_1)).permute(0,3,1,2)
        e_msi1 = self.Ct(e_msi1_2)
        e_msi1 = ((e_msi1*0.1 + e_msi1_2)).permute(0,2,3,1)
        # e_msi1 = (e_msi1 + e_msi1_2)/2

        e_msi1 = torch.squeeze(self.C_2(0.1 * e_msi1.unsqueeze(1) + e_msi1_1.unsqueeze(1)), 1)
        e_msi1 = e_msi1 - lr_hsi
        pre_pro_1 = torch.squeeze(self.Res3d_up(e_msi1.unsqueeze(1)), 1)
        out1 = self.pro((pre_pro_1 + pro_right_1 + pro_left_1) / 3 + out)

        #####
        e_hsi2 = self.Down1(out1) - hr_msi
        pro_R_2 = self.Res2d_s_up(e_hsi2)

        pro_left_2 = pro_R_2.permute(0,2,1,3)      # B / H / slc / W
        pro_left_r = self.ce_xiang(pro_left_2)
        pro_left_r = pro_left_r.permute(0,2,1,3)
        pro_left_r = self.pro(pro_left_r + out1)
        pro_left_2 = pro_left_2.permute(0,2,1,3)


        pro_right_2 = pro_R_2.permute(0,3,1,2)     # B / W / slc / H
        pro_right_r = self.ce_xiang(pro_right_2)
        pro_right_r = pro_right_r.permute(0,2,3,1)   #
        pro_right_r = self.pro(pro_right_r + out1)
        pro_right_2 = pro_right_2.permute(0,2,3,1)

        e_msi2 = self.C_1(out1)

        e_msi2_1 = e_msi2


        e_msi2 = e_msi2.permute(0, 2, 1, 3)
        e_msi2 = self.Ct(e_msi2)
        e_msi2 = e_msi2.permute(0, 2, 1, 3)

        e_msi2_2 = ((e_msi2*0.1 + e_msi2_1) ).permute(0, 3, 1, 2)
        e_msi2 = self.Ct(e_msi2_2)
        e_msi2 = ((e_msi2*0.1 + e_msi2_2) ).permute(0, 2, 3, 1)
        # e_msi2 = (e_msi2 + e_msi2_2) / 2

        # e_msi2 = e_msi2.permute(0, 3, 1, 2)
        # e_msi2 = self.Ct(e_msi2)
        # e_msi2 = e_msi2.permute(0, 2, 3, 1)

        e_msi2 = torch.squeeze(self.C_2(0.1 * e_msi2.unsqueeze(1) + e_msi2_1.unsqueeze(1)),1)
        e_msi2 = e_msi2 - lr_hsi
        pre_pro_2 = torch.squeeze(self.Res3d_up(e_msi2.unsqueeze(1)),1)
        pre_pro_r = self.pro(pre_pro_2 + out1)
        out2 = self.pro((pre_pro_2 + pro_right_2 + pro_left_2)/3 + out1)
        
        out = self.out(torch.cat([out2, hr_msi],dim=1))

        return out, out1, out2, e_hsi1, e_msi1, e_hsi2,e_msi2,pro_left_r, pro_right_r, pre_pro_r



if __name__ == '__main__':
    # a = torch.arange(1, 5)
    # b = (a < 3)
    # print(b.shape)
    # print(b.dtype)
    # c = b * a
    # print(b)
    # print('________')
    # print(c)
    # print(c.dtype)
    a = '0 2 3 4'
    b = list(map(int, a.split(' ')))
    print(b)