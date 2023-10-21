from utils import *
import torch

from net3 import *
from net4 import *

import argparse
from torch.utils.data import DataLoader
from data_loader import LoadData





def get_args():
    parser = argparse.ArgumentParser(description='uSDN for HSI-SR')
    parser.add_argument('--cuda', type=str, default='1')




    parser.add_argument('--test_model',type=str, default='./data_cave/M2_dr4_6_22_384.ckpt')          # TODO

    parser.add_argument('--test', type=str, default='./data_cave/test_par.mat')
    parser.add_argument('--test_HR', type=str, default='./data_cave/test_HR_wS.npy')
    parser.add_argument('--test_LR', type=str, default='./data_cave/test_dr4_wS.npy')

    parser.add_argument('--save_dir', type=str, default='./model_test_result/')            # TODO


    # image
    # model
    parser.add_argument('--down_rate', type=int, default=8)
    parser.add_argument('--num_slc', type=int, default=3)
    parser.add_argument('--num_spectral', type=int, default=31)
    parser.add_argument('--slc', type=list, default=[0,6,22])  # -3.5ppm  3.5ppm
    parser.add_argument('--ker', type=tuple, default=(3, 1, 1))




    # rio & rate

    parser.add_argument('--test_batch_size', type=int, default=2)


    args = parser.parse_args()
    return args

def test():
    args = get_args()

    if torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        print("Current device: idx%s | %s" % (
        torch.cuda.current_device(), torch.cuda.get_device_name(torch.cuda.current_device())))
        # assert 0
    else:
        print('cpu')
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)


    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)


    test_loader = DataLoader(
        dataset=LoadData(args.test, args.test_HR, args.test_LR, 'test', args.slc, down_rate=args.down_rate,
                         patch_size=args.patch_size), batch_size=args.test_batch_size, shuffle=False,
        num_workers=10, pin_memory=True, drop_last=True)
    model = OUSC_net_2_full_CNN(config=args).to(dev)

    model.load_state_dict(torch.load(args.test_model, map_location=dev))


###########
    model.eval()
    test_count = 0
    test_psnr = 0
    test_nor_loss = 0
    test_PSNR_core = 0
    test_pred_list = []
    with torch.no_grad():
        for lr_hsi_t, hr_msi_t, hr_hsi_t, mask, S0 in test_loader:
            lr_hsi_t = lr_hsi_t.to(dev)
            hr_msi_t = hr_msi_t.to(dev)
            hr_hsi_t = hr_hsi_t.to(dev)
            # S0 = S0.to(dev)
            # mask = mask.to(dev)
            # S0 = torch.unsqueeze(S0, dim=1)
            # mask = torch.unsqueeze(mask, dim=1)
            # S0_all = torch.repeat_interleave(S0, repeats=args.num_spectral, dim=1)
            # mask_all = torch.repeat_interleave(mask, repeats=args.num_spectral, dim=1)
            test_count += 1
            with torch.no_grad():
                # pred_test,_,_,_,_,_,_ = model(lr_hsi_t, hr_msi_t)     #### M1
                pred_test, _, _, _, _, _, _, _, _, _ = model(lr_hsi_t, hr_msi_t)  #### M2
                # pred_test, _, _, _, _, _, _, _, _, _, _, _, _ = model(lr_hsi_t, hr_msi_t)    #### M3
                test_pred_list.append(pred_test)
                # pred = torch.mul(torch.div(pred_test, S0_all + 1e-08), mask_all)
                # hr_hsi = torch.mul(torch.div(hr_hsi_t, S0_all + 1e-08), mask_all)
                # test_nor_loss = test_nor_loss + torch.mean(
                #     torch.abs(pred[torch.where(mask_all > 0.5)] - hr_hsi[torch.where(mask_all > 0.5)]))
                # core_psnr = PSNR_GPU_mask(pred[:, 1:31, :, :], hr_hsi[:, 1:31, :, :], mask)

                psnr = PSNR_GPU(hr_hsi_t, pred_test)

                test_psnr += psnr
                # test_PSNR_core += core_psnr

    log_test = 'test averagr psnr : {:.4f}  core_PSNR : {:.4f}'.format(test_psnr / (test_count),

                                                                                        test_PSNR_core / test_count)

    print('#####' + log_test + '#####')
    save_mat(args.save_dir, test_pred_list, 603)         # TODO






if __name__ =='__main__':
    test()
