import torch
import torch.nn as nn
import torch.nn.functional as F



class Ca_attention(nn.Module):
    def __init__(self, in_ch ,num_spectral):
        super(Ca_attention, self).__init__()
        self.layer1 = nn.Conv2d(in_channels=in_ch, out_channels=1, kernel_size=1, stride=1)
        self.acti1 = nn.LeakyReLU()
        self.layer2 = nn.Conv2d(in_channels=1, out_channels=num_spectral, kernel_size=1, stride=1)
        self.acti2 = nn.Sigmoid()
    def forward(self, x):
        x = torch.mean(x, dim=[2,3], keepdim=True)    # batch_size/bands/img_w/img_h
        ca = self.layer1(x)
        ca = self.acti1(ca)
        ca = self.layer2(ca)
        ca = self.acti2(ca)
        return ca



class Sa_attention(nn.Module):
    def __init__(self, in_ch, kernal):
        super(Sa_attention, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=1, kernel_size=kernal, padding=int(kernal/2), stride=1),
            nn.Sigmoid(),
            # nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernal, stride=1),
            # nn.Sigmoid()
        )
    def forward(self,x):
        # print(x.shape)
        # assert 0
        sa = self.layer(x)
        return sa

class Block_Resunet_2d(nn.Module):
    def __init__(self, in_ch, ou_ch):
        super(Block_Resunet_2d, self).__init__()
        self.in_ch = in_ch
        self.ou_ch = ou_ch
        self.encoder1_a = nn.Sequential(nn.Conv2d(in_channels=self.in_ch,out_channels=self.in_ch,kernel_size=5, padding=2, stride=1),
                                        nn.BatchNorm2d(self.in_ch)
                                      )
        self.encoder1_b = nn.Sequential(nn.Conv2d(self.in_ch, self.in_ch*2,kernel_size=4,padding=1,stride=2),
                                      nn.BatchNorm2d(self.in_ch*2),
                                      nn.LeakyReLU(True))
        self.encoder2_a = nn.Sequential(
            nn.Conv2d(self.in_ch*2,self.in_ch*2, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(self.in_ch * 2)
        )
        self.encoder2_b = nn.Sequential(
            nn.Conv2d(self.in_ch * 2, self.in_ch * 4, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm2d(self.in_ch * 4),
            nn.LeakyReLU(True)
        )
        self.encoder3_a = nn.Sequential(nn.Conv2d(self.in_ch*4,self.in_ch*4,kernel_size=3,padding=1,stride=1),
                                        nn.BatchNorm2d(self.in_ch*4)
                                        )
        self.encoder3_b = nn.Sequential(
            nn.Conv2d(self.in_ch * 4, self.in_ch * 8, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm2d(self.in_ch * 8),
            nn.LeakyReLU(True)
        )

        self.decoder1_a = nn.Sequential(nn.Conv2d(self.in_ch*16,self.in_ch*8,kernel_size=3,padding=1,stride=1),
                     nn.BatchNorm2d(self.in_ch*8)
                     )
        self.decoder1_b = nn.Sequential(nn.ConvTranspose2d(self.in_ch*8,self.in_ch*8,kernel_size=4,padding=1, stride=2),
                                        nn.BatchNorm2d(self.in_ch*8),
                                        nn.LeakyReLU(True)
                                        )
        self.decoder2_a = nn.Sequential(
            nn.Conv2d(self.in_ch * 12, self.in_ch * 8, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(self.in_ch * 8)
        )
        self.decoder2_b = nn.Sequential(
            nn.ConvTranspose2d(self.in_ch*8, self.in_ch*4,kernel_size=4, padding=1,stride=2),
            nn.BatchNorm2d(self.in_ch * 4),
            nn.LeakyReLU(True)
        )
        self.decoder3_a = nn.Sequential(
            nn.Conv2d(self.in_ch * 6, self.in_ch * 4, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(self.in_ch * 4)
        )
        self.decoder3_b = nn.Sequential(
            nn.ConvTranspose2d(self.in_ch*4, self.in_ch*2,kernel_size=4, padding=1,stride=2),
            nn.BatchNorm2d(self.in_ch * 2),
            nn.LeakyReLU(True)
        )
        self.decoder4_a = nn.Sequential(
            nn.Conv2d(self.in_ch*3,self.in_ch*2, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(self.in_ch*2),
            nn.LeakyReLU(True)
        )
        self.decoder4_b = nn.Sequential(
            nn.Conv2d(self.in_ch*2,self.in_ch,kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(self.in_ch),
            nn.LeakyReLU(True)
        )
        self.decoder5 = nn.Sequential(
            nn.Conv2d(self.in_ch,self.ou_ch,kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(self.ou_ch),
            nn.LeakyReLU(True)
        )

        self.Bottneck = nn.Sequential(
            nn.Conv2d(self.in_ch*8,self.in_ch*16,kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(self.in_ch*16),
            nn.LeakyReLU(True),
            nn.Conv2d(self.in_ch * 16, self.in_ch * 8, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(self.in_ch * 8),
            nn.LeakyReLU(True)
        )
    def forward(self, x):
        en1_a = self.encoder1_a(x)
        en1_b = self.encoder1_b(en1_a + x)

        en2_a = self.encoder2_a(en1_b)
        en2_b = self.encoder2_b(en2_a+ en1_b)
        en3_a = self.encoder3_a(en2_b)
        en3_b = self.encoder3_b(en3_a+en2_b)
        en4 = self.Bottneck(en3_b)
        de1 = torch.concat([en4, en3_b],dim=1)
        de1_a = self.decoder1_a(de1)
        de1_b = self.decoder1_b(en4 + de1_a)
        de2_a = self.decoder2_a(torch.concat([en2_b, de1_b],dim=1))
        de2_b = self.decoder2_b(de1_b + de2_a)
        de3_a = self.decoder3_a(torch.concat([en1_b, de2_b],dim=1))
        de3_b = self.decoder3_b(de2_b + de3_a)
        de4_a = self.decoder4_a(torch.concat([x, de3_b], dim=1))
        de4_b = self.decoder4_b(de3_b + de4_a)
        de5 = self.decoder5(de4_b)
        return de5


class Decoder_Resnet_3d(nn.Module):   #### 只适用与8倍
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
            nn.Conv3d(in_channels=2, out_channels=2, kernel_size=self.ker, padding=(1, 0, 0), stride=1),
            nn.BatchNorm3d(2),
            nn.LeakyReLU(True)
        )
        self.d4 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=2, out_channels=4, kernel_size=(1, 4, 4), padding=(0, 1, 1), stride=(1,2,2)),
            nn.BatchNorm3d(4),
            nn.LeakyReLU(True)
        )
        self.d5 = nn.Sequential(
            nn.Conv3d(in_channels=4, out_channels=4, kernel_size=self.ker, padding=(1, 0, 0), stride=1),
            nn.BatchNorm3d(4),
            nn.LeakyReLU(True)
        )
        #self.d6 = nn.Sequential(
        #    nn.ConvTranspose3d(in_channels=4, out_channels=8, kernel_size=(1, 4, 4), padding=(0, 1, 1), stride=(1,2,2)),
        #    nn.BatchNorm3d(8),
        #    nn.LeakyReLU(True)
        #)
        #### 这里只适用于8倍
        self.d7 = nn.Sequential(
            #nn.Conv3d(in_channels=8, out_channels=8, kernel_size=self.ker, padding=(1, 0, 0), stride=1),
            #nn.BatchNorm3d(8),
            #nn.LeakyReLU(True),
            nn.Conv3d(in_channels=4, out_channels=4, kernel_size=self.ker, padding=(1, 0, 0), stride=1),
            nn.BatchNorm3d(4),
            nn.LeakyReLU(True),
            nn.Conv3d(in_channels=4, out_channels=2, kernel_size=self.ker, padding=(1, 0, 0), stride=1),
            nn.BatchNorm3d(2),
            nn.LeakyReLU(True),
            nn.Conv3d(in_channels=2, out_channels=1, kernel_size=self.ker, padding=(1, 0, 0), stride=1),
            nn.BatchNorm3d(1),
            nn.LeakyReLU(True),
        )

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(x + d1)
        d3 = self.d3(d2)
        d4 = self.d4(d2 + d3)
        d5 = self.d5(d4)
        # d6 = self.d6(d5 + d4)
        d7 = self.d7(d5)
        return d7



class Net(nn.Module):
    def __init__(self, config=None):
        super(Net, self).__init__()
        self.config = config
        self.num_spec = config.num_spectral
        self.num_slc = config.num_slc
        self.slc = self.config.slc  # list 形式  [1 到 n]

        # self.num_spec = 31
        # self.num_slc = 3
        # self.slc = [10,16,23]  # list 形式  [1 到 n]
        # self.ker = (3,1,1)
        self.sa = Sa_attention(self.num_slc, 5)
        self.ca = Ca_attention(self.num_spec, self.num_spec)
        self.Res2 = Block_Resunet_2d(self.num_spec, self.num_spec)
        self.Res1 = Block_Resunet_2d(self.num_slc, self.num_slc)
        self.Res3 = Decoder_Resnet_3d(self.config.ker)  # （3 1 1）
        self.down1 = nn.Sequential(
            nn.Conv2d((self.num_spec+self.num_slc), self.num_spec, kernel_size=3, padding=1,stride=1),
            nn.BatchNorm2d(self.num_spec),
            nn.LeakyReLU(True)
        )
        # self.down2 = nn.Sequential(
        #     nn.Conv2d(self.num_spec, self.num_spec, kernel_size=5, padding=2,stride=1),
        #     nn.BatchNorm2d(self.num_spec),
        #     nn.LeakyReLU(True)
        # )
        # self.down3 = nn.Sequential(
        #     nn.Conv2d(self.num_spec, self.num_spec, kernel_size=5, padding=2,stride=1),
        #     nn.BatchNorm2d(self.num_spec),
        #     nn.LeakyReLU(True)
        # )
        # self.down4 = nn.Sequential(
        #     nn.Conv2d(self.num_spec, self.num_spec, kernel_size=5, padding=2,stride=1),
        #     nn.BatchNorm2d(self.num_spec),
        #     nn.LeakyReLU(True)
        # )
    def split_slc(self, lr_hsi_pred, hr_hsi):


        out = lr_hsi_pred[:,0:self.slc[0],:,:]

        for i in range(self.num_slc):
            out = torch.concat([out,self._unsqueeze( hr_hsi[:, i, :, :])], dim=1)
            if i == (self.num_slc-1):
                out = torch.concat([out, lr_hsi_pred[:, self.slc[i]:self.num_spec, :, :]], dim=1)
            else:
                out = torch.concat([out, lr_hsi_pred[:,self.slc[i]:self.slc[i+1],:,:]],dim=1)
        return out
    def _unsqueeze(self,x):
        x = torch.unsqueeze(x,dim=1)
        return x
    def _squeeze(self,x):
        x = torch.squeeze(x,dim=1)
        return x
    def _fft(self,x):
        pass
    def _ifft(self,x):
        pass
    def forward(self, lr_hsi, hr_msi):
        sa = self.sa(hr_msi)
        ca = self.ca(lr_hsi)
        lr_hsi = self._unsqueeze(lr_hsi)
        lr_pre1 = self.Res3(lr_hsi)
        lr_pre1 = self._squeeze(lr_pre1)
        hr_pre1 = self.Res1(hr_msi)
        mix = self.split_slc(lr_pre1,hr_pre1)
        mix = self.down1(mix)
        mix = mix * sa
        mix = mix * ca
        out = self.Res2(mix)
        # out1 = self.down2(mix)
        # out2 = self.down3(mix + out1)
        # out3 = self.down4(out2 + out2)
        return out



if __name__ == '__main__':
    lr_hsi = torch.ones([4,31,14,14])
    hr_msi = torch.ones([4,3,112,112])

    model = Net()
    pred = model(lr_hsi, hr_msi)