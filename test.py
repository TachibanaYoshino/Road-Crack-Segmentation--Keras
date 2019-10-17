import argparse
import data_loader
import glob, os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from metrics import *
import keras
# from net.Unet import Net
from net.GCUnet import Net
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


parser = argparse.ArgumentParser()
parser.add_argument("--save_weights_path", type=str,default='checkpoint/'+ 'GCUnet/' + 'weights-092-0.1294-0.9792.h5')
parser.add_argument("--test_images", type=str, default='dataset/CRACK500/testcrop/')
parser.add_argument("--output_path", type=str,default='test_result')
parser.add_argument("--img_height", type=int, default=224)
parser.add_argument("--img_width", type=int, default=224)
parser.add_argument("--vis", type=bool, default=False)


args = parser.parse_args()

images_path = args.test_images
img_width =  args.img_width
img_height = args.img_height

batch_size = 4

def output_test_image(x, pr, gt, path):
    fig=plt.figure(figsize=(12, 4))
    fig.add_subplot(1, 3, 1)
    plt.imshow(pr)
    fig.add_subplot(1, 3, 2)
    plt.imshow(x)
    fig.add_subplot(1, 3, 3)
    plt.imshow(gt)
    plt.savefig(path)
    plt.close()

def stats_graph(graph):
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    # params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('FLOPs: {}'.format(flops.total_float_ops))

m = Net()
m.compile(loss='binary_crossentropy',optimizer= keras.optimizers.Adam(lr=1e-4),metrics=['accuracy', f1_score])

# FLOPs
# stats_graph(tf.get_default_graph())

m.load_weights(args.save_weights_path)
print('loaded: ', args.save_weights_path)

test_images = glob.glob(images_path + '*.jpg')
test_images.sort()
gt_seg = glob.glob(images_path + '*.png')
gt_seg.sort()

assert len(test_images) == len(gt_seg)
for im, seg in zip(test_images, gt_seg):
    assert (im.split('/')[-1].split('.')[0] == seg.split('/')[-1].split('.')[0])

test_gen = data_loader.imageSegmentationGenerator(images_path,
               images_path, batch_size, img_height, img_width, False,phase='test')

pred = m.predict_generator(test_gen, steps=len(test_images)//batch_size, verbose=1)
pred = np.argmax(pred, axis=-1)
print(pred.shape)



iou_score = 0.0
f1_scores = 0.0
for idx in range(pred.shape[0]):
    gt = data_loader.getSegmentationArr(gt_seg[idx], img_width, img_height)
    gt = np.argmax(gt, axis=-1)
    pred_label = pred[idx, :, :]
    intersection = np.logical_and(gt, pred_label)
    union = np.logical_or(gt, pred_label)
    iou_score += np.sum(intersection,dtype=np.float) / np.sum(union,dtype=np.float)

    f1_scores += f1_score(gt,pred_label)

with tf.Session() as sess:

    print('mIOU:' + str(iou_score/pred.shape[0]), 'f1_score:' + str(f1_scores.eval()/pred.shape[0]))

if args.vis:
    output_path = args.save_weights_path.split('/')[1]+'_predict/'
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    for i, imgName in enumerate(test_images):

        outName = os.path.join(output_path,os.path.basename(imgName))

        X,gt = data_loader.getImageArr(imgName, gt_seg[i], img_width, img_height,False, phase='test')
        pr = pred[i, :,:]
        gt = np.argmax(gt, axis=-1)
        output_test_image(X, pr, gt, outName)
        
