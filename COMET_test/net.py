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
            nn.Conv2d(in_channels=in_ch, out_channels=1, kernel_size=kernal, stride=1, padding=int(kernal/2)),
            nn.Sigmoid(),
            # nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernal, stride=1),
            # nn.Sigmoid()
        )
    def forward(self,x):
        x = torch.mean(x, dim=[1], keepdim=True) # 在1bands维平均
        sa = self.layer(x)
        return sa


class Block(nn.Module):
    def __init__(self, in_ch,ou_ch):
        super(Block, self).__init__()
        self.ly1 = nn.Conv2d(in_ch, ou_ch, kernel_size=3, stride=1,padding=1)
        self.acti1 = nn.LeakyReLU()
        self.ly2 = nn.Conv2d(ou_ch, ou_ch, kernel_size=3, stride=1,padding=1)
        self.acti2 = nn.LeakyReLU()
        self.BN1 = nn.BatchNorm2d(num_features=ou_ch)
        self.BN2 = nn.BatchNorm2d(num_features=ou_ch)
    def forward(self, x):
        out1 = self.ly1(x)
        out1 = self.BN1(out1)
        out1 = self.acti1(out1)
        out2 = self.ly2(out1)
        out2 = self.BN2(out2)
        out2 = self.acti2(out2)
        out = out2 + x
        return out


class Resnet(nn.Module):
    def __init__(self, in_ch,ou_ch,num_res):
        super(Resnet, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(in_ch, ou_ch, kernel_size=3, padding=1, stride=1),
                                    nn.BatchNorm2d(ou_ch),
                                    nn.LeakyReLU())
        self.layer2 = self._make_block(ou_ch, ou_ch, num_res)
    def _make_block(self, in_ch, ou_ch,num_res):
        layer = []
        for i in range(num_res):
            if i == 1:
                layer.append(Block(in_ch, ou_ch))
            else:
                layer.append(Block(ou_ch, ou_ch))
        layer = nn.Sequential(*layer)
        return layer
    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        return out


class H_net(nn.Module):
    def __init__(self, config, dev):
        super(H_net, self).__init__()
        self.config = config
        self.down_rate = config.down_rate
        self.shape = list(map(int,self.config.shape.split(' ')))  # batch_num, spec_num, H, W
        self.num_spectral = config.num_spectral
        self.num_slc = config.num_slc
        self.rank = config.rank
        self.num_res = config.num_res
        self.dev = dev
        self.ly1 = nn.Sequential(
            nn.Conv2d((self.rank-self.num_slc), out_channels=self.num_spectral, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.num_spectral),
            nn.LeakyReLU()
        )
        self.ly2 = nn.Sequential(
            nn.Conv2d(self.num_slc, out_channels=self.num_spectral, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.num_spectral),
            nn.LeakyReLU()
        )
        self.Down = nn.Sequential(
            nn.Conv2d(self.num_spectral, out_channels=self.num_spectral, kernel_size=(2+self.down_rate), stride=self.down_rate, padding=1),
            nn.BatchNorm2d(self.num_spectral),
            nn.LeakyReLU()
        )
        self.Up = nn.Sequential(
            nn.ConvTranspose2d(self.num_spectral, (self.rank-self.num_slc), kernel_size=(2+self.down_rate), padding=1, stride=self.down_rate),
            nn.BatchNorm2d(self.rank-self.num_slc),
            nn.LeakyReLU(),
            nn.Conv2d((self.rank-self.num_slc), out_channels=(self.rank-self.num_slc), kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d((self.rank-self.num_slc)),
            nn.LeakyReLU()
        )
        self.map = Map(self.config, self.dev)

        self.map_H = nn.Sequential(
            nn.Conv2d(self.shape[2], out_channels=int(self.shape[2] / 4), kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(int(self.shape[2] / 4)),
            nn.LeakyReLU(),
            nn.Conv2d(int(self.shape[2] / 4), self.shape[2], kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(self.shape[2]),
            nn.LeakyReLU()
        )
        self.map_W = nn.Sequential(
            nn.Conv2d(self.shape[3], int(self.shape[3] / 4), kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(int(self.shape[3] / 4)),
            nn.LeakyReLU(),
            nn.Conv2d(int(self.shape[3] / 4), self.shape[3], kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(self.shape[3]),
            nn.LeakyReLU()
        )
        self.map_S = nn.Sequential(
            nn.Conv2d(self.num_spectral, self.num_spectral, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(self.num_spectral),
            nn.LeakyReLU()
        )
        self.resnet = Resnet(in_ch=(self.rank-self.num_slc), ou_ch=(self.rank-self.num_slc), num_res=self.num_res)
    def forward(self, hr_rhsi, hr_msi, lr_hsi):
        out1 = self.ly1(hr_rhsi)
        out2 = self.ly2(hr_msi)
        pred_hsi = out2 + out1
        pred_hsi1 = torch.permute(pred_hsi, [0, 2, 3, 1])
        pred_hsi1 = self.map_H(pred_hsi1)
        pred_hsi1 = torch.permute(pred_hsi1, [0, 3, 1, 2])
        pred_hsi2 = torch.permute(pred_hsi, [0, 3, 1, 2])
        pred_hsi2 = self.map_W(pred_hsi2)
        pred_hsi2 = torch.permute(pred_hsi2, [0, 2, 3, 1])
        pred_hsi3 = self.map_S(pred_hsi)
        pred_hsi = (pred_hsi1 + pred_hsi2 + pred_hsi3) / 3
        out3 = self.Down(pred_hsi)
        e_lr = out3 - lr_hsi
        out4 = -self.Up(e_lr)
        out = hr_rhsi + out4
        out = self.map(out)
        out = self.resnet(out)
        return out, pred_hsi, e_lr


class HSInet(nn.Module):
    def __init__(self, config, dev):
        super(HSInet, self).__init__()
        self.dev = dev
        self.config = config
        self.down_rate = config.down_rate
        self.num_spectral = config.num_spectral
        self.num_slc = config.num_slc
        self.rank = config.rank
        self.num_res = config.num_res
        self.k = self.config.k
        self.ca = Ca_attention(self.num_spectral, self.num_spectral)
        self.sa = Sa_attention(1, kernal=5)
        self.layer1 = nn.Sequential(
            nn.Conv2d(self.num_slc, self.num_spectral, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(self.num_spectral),
            nn.LeakyReLU()
        )
        self.Down1 = nn.Sequential(
            nn.Conv2d(self.num_spectral, out_channels=self.num_spectral, kernel_size=(2 + self.down_rate),
                      stride=self.down_rate, padding=1),
            nn.BatchNorm2d(self.num_spectral),
            nn.LeakyReLU()
        )
        self.Up = nn.Sequential(
            nn.ConvTranspose2d(self.num_spectral, (self.rank-self.num_slc), kernel_size=(2+self.down_rate), padding=1, stride=self.down_rate),
            nn.BatchNorm2d(self.rank-self.num_slc),
            nn.LeakyReLU(),
            nn.Conv2d((self.rank-self.num_slc), out_channels=(self.rank - self.num_slc), kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d((self.rank - self.num_slc)),
            nn.LeakyReLU()
        )
        self.map_1 = Map(config,dev)
        self.resnet1 = Resnet(in_ch=(self.rank - self.num_slc), ou_ch=(self.rank - self.num_slc), num_res=self.num_res)
        self.midlayer = []
        for i in range(self.k):
            self.midlayer.append(H_net(config,self.dev).to(self.dev))

        self.layer_fl1 = nn.Sequential(
            nn.Conv2d((self.rank - self.num_slc), self.num_spectral, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(self.num_spectral),
            nn.LeakyReLU()
        )
        self.layer_fl2 = nn.Sequential(
            nn.Conv2d(self.num_slc, out_channels=self.num_spectral, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.num_spectral),
            nn.LeakyReLU()
        )
        self.Down_fl = nn.Sequential(
            nn.Conv2d(self.num_spectral, out_channels=self.num_spectral, kernel_size=(2 + self.down_rate),
                      stride=self.down_rate, padding=1),
            nn.BatchNorm2d(self.num_spectral),
            nn.LeakyReLU()
        )

        self.resnet_fl = Resnet(in_ch=self.num_spectral, ou_ch=self.num_spectral, num_res=self.num_res)
    def forward(self, lr_hsi, hr_msi):
        ####
        
        hr_hsi_list = []
        e_list = []
        out1_hr = self.layer1(hr_msi)
        out1 = self.Down1(out1_hr)

        e1 = out1 - lr_hsi
        
        # print(e1.shape)
        out1 = self.Up(e1)
        ######
        out1 = self.map_1(out1)
        ######
        hr_rhsi = self.resnet1(out1)
        hr_hsi_list.append(out1_hr)
        e_list.append(e1)

        for k in range(self.k):
           hr_rhsi, pred_hsi, e_lr = self.midlayer[k](hr_rhsi, hr_msi, lr_hsi)
           hr_hsi_list.append(pred_hsi)
           e_list.append(e_lr)

        out_fl = self.layer_fl1(hr_rhsi) + self.layer_fl2(hr_msi)
        hr_hsi_list.append(out_fl)
        e_fl = self.Down_fl(out_fl) - lr_hsi
        e_list.append(e_fl)


        if self.config.ca_sa:
            ca = self.ca(lr_hsi)
            sa = self.sa(hr_msi)
            out_fl = out_fl * sa

            pred_hr_hsi = self.resnet_fl(out_fl)
            pred_hr_hsi = pred_hr_hsi * ca
        else:

            pred_hr_hsi = self.resnet_fl(out_fl)
        return pred_hr_hsi, hr_hsi_list, e_list


class H_loss(nn.Module):
    def __init__(self, config):
        super(H_loss, self).__init__()
        self.rate_z = config.rate_z
        self.rate_e = config.rate_e
    def forward(self,pred_hsi, pred_list,e_loss_list, hr_hsi):
        hsi_pred =torch.mean(torch.pow((pred_hsi-hr_hsi),2))
        a = hsi_pred
        for k in range(len(pred_list)):
            a = a + self.rate_z * (torch.mean(torch.pow((pred_list[k]-hr_hsi),2) + self.rate_e *torch.mean(torch.pow(e_loss_list[k], 2))))

        return a

class Map(nn.Module):
    def __init__(self,config, dev):
        super(Map, self).__init__()
        self.config = config
        self.shape = list(map(int,config.shape.split(' ')))
        self.dev = dev
        self.clas = config.clas
        self.num_spectral = config.num_spectral
        self.num_slc = config.num_slc
        self.rank = config.rank
        
        self.ly_map = nn.Sequential( nn.Conv2d((self.rank - self.num_slc), 1, stride=1,kernel_size=3, padding=1),
                                         nn.BatchNorm2d(num_features=1),
                                         nn.Sigmoid()
                                         )
        if self.config.spectral_map:
            self.ly_spe = nn.Sequential(nn.Conv1d(in_channels=(self.shape[2]*self.shape[3]),out_channels=(self.shape[2]*self.shape[3]),stride=1,padding=3,kernel_size=7),
                                        nn.BatchNorm1d(num_features=(self.shape[2]*self.shape[3])),
                                        nn.LeakyReLU()
                                        )
        self.fl = nn.Flatten()
        self.ful_clas = nn.Sequential(nn.Linear(in_features=(self.shape[2]*self.shape[3]),out_features=(self.clas-1)),

                                 nn.Sigmoid()
                                 )
        self.maplayer = []

        for i in range(self.clas):
            self.maplayer.append(nn.Conv2d((self.rank - self.num_slc), (self.rank - self.num_slc), stride=1, kernel_size=3, padding=1).to(self.dev))

        self.B = nn.Sequential(
            nn.BatchNorm2d(num_features=(self.rank - self.num_slc)),
            nn.LeakyReLU()
        )

    def forward(self, hr_rhsi):
        shape = hr_rhsi.shape
        if self.config.spectral_map:
            x = torch.permute(hr_rhsi,[0, 2, 3, 1])
            x = torch.reshape(x, [shape[0],shape[2]*shape[3], shape[1]])
            x = self.ly_spe(x)
            x = torch.reshape(x,[shape[0], shape[2], shape[3], shape[1]])
            hr_rhsi = torch.permute(x, [0, 3, 1, 2])


        map = self.ly_map(hr_rhsi)
        num_class = self.fl(map)

        
        num_class = self.ful_clas(num_class)
        num_c = torch.mean(num_class, dim=0)

        maskfl = []
        mask = torch.clone(map).repeat(1,shape[1],1,1)
        for i in range(self.clas):
            if i == self.clas-1:
                a = (mask > 0.0000001)
            else:
                a = (mask > num_c[i])
            b = a * map
            maskfl.append(self.maplayer[i](b))
            mask = mask - a*mask
        ou = maskfl[0]
        for i in range(1,self.clas):
            ou = ou + maskfl[i]
        ou = self.B(ou)
        return ou

def angle_loss(pred, ref):
    eps = 1e+06
    nom_pred = torch.sum(torch.pow(pred,2))
    nom_true = torch.sum(torch.pow(ref,2))
    nom_base = torch.sqrt(torch.multiply(nom_true,nom_pred) + eps)
    nom_top = torch.sum(torch.multiply(pred, ref))
    angle = torch.mean(torch.acos(torch.div(nom_top, (nom_base + eps))))
    angle_loss = torch.div(angle, 3.1416)
    return angle_loss





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
