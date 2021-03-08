import argparse
import logging
import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import pandas as pd
import cv2
from tqdm import tqdm
from unet import UNet
from utils.data_vis import plot_img_and_mask
from utils.dataset import BasicDataset


def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    print(full_img.size)
    full_img=full_img.convert('RGB')
    print(full_img.size)
    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor))
    print(img.shape)
    img = img.unsqueeze(0)  #增加一个维度 输入网络  当做batchsize
    img = img.to(device=device, dtype=torch.float32)
    print(img.shape)
    with torch.no_grad():
        output = net(img)
        print(output.shape)
        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)

        probs = probs.squeeze(0)
        print(probs.shape)
        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(full_img.size[1]),
                transforms.ToTensor()
            ]
        )

        probs = tf(probs.cpu())
        print(probs.shape)
        full_mask = probs.squeeze().cpu().numpy()
        print(full_mask.shape)
    return full_mask > out_threshold


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='CP_epoch50.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        help='filenames of input images', required=True)

    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
                        help='Filenames of ouput images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minim"
                             "um probability value to consider a mask pixel white",
                        default=0.5)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=1)

    return parser.parse_args()


def get_output_filenames(args):
    in_files = args.input
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        logging.error("Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))


def rle_encode(im):
    '''
    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = im.flatten(order = 'F') # 返回一维数组  101*101
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)
def testImageFetch(test_id):
  image_test = np.zeros((len(test_id), 101, 101), dtype=np.float32)
  for idx, image_id in tqdm(enumerate(test_id), total=len(test_id)):
    image_path = os.path.join(test_image_dir, image_id+'.png')
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
    image_test[idx] = image
  return image_test
if __name__ == "__main__":
    args = get_args()

    test_image_dir = args.input[0]
    in_files = [ os.path.join(test_image_dir, x[:]) for x in os.listdir(test_image_dir) if x[-4:] == '.png']  #取得测试文件夹内所有文件
    test_id = [x[:-4] for x in os.listdir(test_image_dir) if x[-4:] == '.png'] #取得测试文件夹内所有id
    #out_files = get_output_filenames(args)
   # net = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana')
    net = UNet(n_channels=3, n_classes=1)
    logging.info("Loading model {}".format(args.model))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info("Model loaded !")
    overall_pred_101 = np.zeros((len(in_files), 101, 101), dtype=np.float32) #保存所有预测的结果
    for i, fn in enumerate(in_files):
        logging.info("\nPredicting image {} ...".format(fn))
        img = Image.open(fn)
        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)
        overall_pred_101[i]=mask
        # （18000,101,101）

        # if not args.no_save:
        #     out_fn = out_files[i]
        #     result = mask_to_image(mask)
        #     result.save(out_files[i])
        #     logging.info("Mask saved to {}".format(out_files[i]))

        if args.viz:
            logging.info("Visualizing results for image {}, close to continue ...".format(fn))
            plot_img_and_mask(img, mask)
    #最终的结果
    submission = pd.DataFrame({'id': test_id, 'rle_mask': list(overall_pred_101)})
    submission['rle_mask'] = submission['rle_mask'].map(lambda x: rle_encode(x))
    submission.set_index('id', inplace=True)
     #  print(submission)
    sample_submission = pd.read_csv('F:\dataset\competition_data/sample_submission.csv')
    sample_submission.set_index('id', inplace=True)
    submission = submission.reindex(sample_submission.index)
    submission.reset_index(inplace=True)
    submission.to_csv('submission.csv', index=False)