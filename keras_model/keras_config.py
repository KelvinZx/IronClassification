import argparse


class Config(object):
    """
    Manually setting configuration.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_count", default=1, type=int)
    parser.add_argument("--image_per_gpu", type=int, default=16)
    parser.add_argument("--epoch", default=300, type=int)
    parser.add_argument("--backbone", default='xception', type=str)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--out_class", default=8, type=int)
    args = parser.parse_args()
    backbone = args.backbone
    out_class = args.out_class
    image_per_gpu = args.image_per_gpu
    epoch = args.epoch
    lr = args.lr
    gpu_count = args.gpu_count


if __name__ == '__main__':
    a = '/home/zhuyiche/Desktop/cell/CRCHistoPhenotypes_2016_04_28/cls_and_det/train/img1/img1_others.mat'
    inf = '/home/zhuyiche/Desktop/cell/CRCHistoPhenotypes_2016_04_28/cls_and_det/train/img1/img1_inflammatory.mat'
    from scipy.io import loadmat
    othermat = loadmat(a)['detection']
    print(othermat)