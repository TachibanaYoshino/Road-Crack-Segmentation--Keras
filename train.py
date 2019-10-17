import argparse, glob,os
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam 
import data_loader
from metrics import *
from net.Unet import Net
# from net.GCUnet import Net
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

parser = argparse.ArgumentParser()
parser.add_argument("--train_images", type=str, default='dataset/CRACK500/traincrop/')
parser.add_argument("--train_annotations", type=str,default='dataset/CRACK500/traincrop/')
parser.add_argument("--img_height", type=int, default=224)
parser.add_argument("--img_width", type=int, default=224)

parser.add_argument("--augment", type=bool, default=True)

parser.add_argument("--val_images", type=str, default='dataset/CRACK500/valcrop/')
parser.add_argument("--val_annotations", type=str, default='dataset/CRACK500/valcrop/')

parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--load_weights", type=str, default=None)
parser.add_argument("--model", type=str, default='checkpoint/Unet',help="path to output model")

args = parser.parse_args()

if not os.path.exists(args.model):
    os.makedirs(args.model)

train_images_path = args.train_images
train_segs_path = args.train_annotations
batch_size = args.batch_size

img_height = args.img_height
img_width = args.img_width

epochs = args.epochs
load_weights = args.load_weights

val_images_path = args.val_images
val_segs_path = args.val_annotations

num_train_images = len(glob.glob(train_images_path + '*.jpg'))
num_valid_images = len(glob.glob(val_images_path + '*.jpg'))


m = Net()
m.compile(loss='binary_crossentropy',optimizer= Adam(lr=1e-4),metrics=['accuracy', f1_score])

if load_weights:
    m.load_weights(load_weights)

print("Model output shape: {}".format(m.output_shape))


train_gen = data_loader.imageSegmentationGenerator(train_images_path,
                  train_segs_path, batch_size, img_height, img_width, args.augment, phase='train')

val_gen = data_loader.imageSegmentationGenerator(val_images_path,
                val_segs_path, batch_size, img_height, img_width, False, phase='test')

filepath = "weights-{epoch:03d}-{val_loss:.4f}-{val_acc:.4f}.h5"
model_weights = os.path.join(args.model, filepath)
checkpoint = ModelCheckpoint(model_weights, monitor='val_loss', verbose=1,save_best_only=False, mode='min', save_weights_only=True)
reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5,  
                                   verbose=1, mode='auto', epsilon=0.0001)

m.fit_generator(train_gen,
                steps_per_epoch = num_train_images//batch_size,
                validation_data = val_gen,
                validation_steps = num_valid_images//batch_size,
                epochs = epochs, 
                verbose = 1, 
                callbacks = [checkpoint, reduceLROnPlat])
