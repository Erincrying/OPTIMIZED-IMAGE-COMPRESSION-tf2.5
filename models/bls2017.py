# Copyright 2018 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Basic nonlinear transform coder for RGB images.

This is a close approximation of the image compression model published in:
J. Ballé, V. Laparra, E.P. Simoncelli (2017):
"End-to-end Optimized Image Compression"
Int. Conf. on Learning Representations (ICLR), 2017
https://arxiv.org/abs/1611.01704

With patches from Victor Xing <victor.t.xing@gmail.com>

This is meant as 'educational' code - you can use this to get started with your
own experiments. To reproduce the exact results from the paper, tuning of hyper-
parameters may be necessary. To compress images with published models, see
`tfci.py`.

This script requires TFC v2 (`pip install tensorflow-compression==2.*`).
"""
# 使用gpu
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0" 
# 限制cpu核数
import tensorflow as tf
os.environ["OMP_NUM_THREADS"] = "4" # cpu核数
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)

# 申请gpu分配内存
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2  # 程序最多只能占用指定gpu50%的显存
config.gpu_options.allow_growth = True  # 设置动态分配GPU内存（可选）
sess = tf.compat.v1.Session(config = config)

import argparse
import glob
import sys
from absl import app
from absl.flags import argparse_flags
# import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_compression as tfc

from numpy import *
import copy

# 查看使用设备
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


def read_png(filename):
  """Loads a PNG image file."""
  string = tf.io.read_file(filename)
  return tf.image.decode_image(string, channels=3)


def write_png(filename, image):
  """Saves an image to a PNG file."""
  string = tf.image.encode_png(image)
  tf.io.write_file(filename, string)


class AnalysisTransform(tf.keras.Sequential):
  # 使用tf.keras.Sequential()来搭建神经网络
  """The analysis transform."""
# kernel 形状 [filter_height, filter_width, in_channels, out_channels]=[5,5,1,36]
# 输入矩阵 [batch, in_height, in_width, in_channels]=[1,256,256,1]
# tf.nn.conv2d执行了以下操作：

# 将滤波器（卷积核）展平为形状为[filter_height * filter_width * in_channels, output_channels]=[25,36]的二维矩阵.
# 从输入张量中提取图像patch,以形成形状为[batch, out_height, out_width, filter_height * filter_width * in_channels]=[1,256,256,25]的虚拟张量.
# 对于每个patch,右乘卷积核矩阵和图像patch矢量.
  def __init__(self, num_filters):
    super().__init__(name="analysis")
    self.add(tf.keras.layers.Lambda(lambda x: x / 255.))
    self.add(tfc.SignalConv2D(
        # num_filters滤波器个数,决定输出特征图的通道数，例如36，输出则为[256,256,36]
        # (9, 9)卷积核尺寸
        # corr=True卷积/互相关
        # strides_down下采样步长
        # same_zeros0填充，为了保证输入输出图片尺寸一致
        # use_bias:Boolean, whether an additive constant will be applied to each output channel.
        # 激活函数GDN
        num_filters, (9, 9), name="layer_0", corr=True, strides_down=4,
        padding="same_zeros", use_bias=True,
        activation=tfc.GDN(name="gdn_0")))
    self.add(tfc.SignalConv2D(
        num_filters, (5, 5), name="layer_1", corr=True, strides_down=2,
        padding="same_zeros", use_bias=True,
        activation=tfc.GDN(name="gdn_1")))
    self.add(tfc.SignalConv2D(
        num_filters, (5, 5), name="layer_2", corr=True, strides_down=2,
        padding="same_zeros", use_bias=False,
        activation=None))


class SynthesisTransform(tf.keras.Sequential):
  """The synthesis transform."""

  def __init__(self, num_filters):
    super().__init__(name="synthesis")
    self.add(tfc.SignalConv2D(
        num_filters, (5, 5), name="layer_0", corr=False, strides_up=2,
        padding="same_zeros", use_bias=True,
        activation=tfc.GDN(name="igdn_0", inverse=True)))
    self.add(tfc.SignalConv2D(
        num_filters, (5, 5), name="layer_1", corr=False, strides_up=2,
        padding="same_zeros", use_bias=True,
        activation=tfc.GDN(name="igdn_1", inverse=True)))
    self.add(tfc.SignalConv2D(
        3, (9, 9), name="layer_2", corr=False, strides_up=4,
        padding="same_zeros", use_bias=True,
        activation=None))
    self.add(tf.keras.layers.Lambda(lambda x: x * 255.))


class BLS2017Model(tf.keras.Model):
  """Main model class."""

  def __init__(self, lmbda, num_filters): # 这里的self就是实例化对象
    super().__init__()
    self.lmbda = lmbda
    self.analysis_transform = AnalysisTransform(num_filters)
    self.synthesis_transform = SynthesisTransform(num_filters)
    self.prior = tfc.NoisyDeepFactorized(batch_shape=(num_filters,)) #先验概率
    self.build((None, None, None, 3))

  def call(self, x, training):
    """Computes rate and distortion losses."""
    # 该库中的熵模型类简化了设计率失真优化代码的过程。在训练期间，它们的行为类似于似然模型。
    entropy_model = tfc.ContinuousBatchedEntropyModel(
        self.prior, coding_rank=3, compression=False)
    y = self.analysis_transform(x)
    y_hat, bits = entropy_model(y, training=training)
    x_hat = self.synthesis_transform(y_hat)
    # Total number of bits divided by total number of pixels.
    #  tf.reduce_prod 计算一个张量的各个维度上元素的乘积.（长乘宽）
    num_pixels = tf.cast(tf.reduce_prod(tf.shape(x)[:-1]), bits.dtype)
    # 码率
    # 码率的单位是bpp，每像素占的bit
    bpp = tf.reduce_sum(bits) / num_pixels
    # Mean squared error across pixels.
    mse = tf.reduce_mean(tf.math.squared_difference(x, x_hat))
    # The rate-distortion Lagrangian.
    loss = bpp + self.lmbda * mse
    return loss, bpp, mse

  def train_step(self, x):
    # x shape=(8,256,256,3)
    with tf.GradientTape() as tape:
      loss, bpp, mse = self(x, training=True)
    variables = self.trainable_variables
    gradients = tape.gradient(loss, variables)
    self.optimizer.apply_gradients(zip(gradients, variables))
    self.loss.update_state(loss)
    self.bpp.update_state(bpp)
    self.mse.update_state(mse)
    print(self.bpp, 'train_step:self.bpp')
    return {m.name: m.result() for m in [self.loss, self.bpp, self.mse]}

  def test_step(self, x):
    loss, bpp, mse = self(x, training=False)
    self.loss.update_state(loss)
    self.bpp.update_state(bpp)
    self.mse.update_state(mse)
    return {m.name: m.result() for m in [self.loss, self.bpp, self.mse]}

  def predict_step(self, x):
    raise NotImplementedError("Prediction API is not supported.")

  # model.compile()方法用于在配置训练方法时，告知训练时用的优化器、损失函数和准确率评测标准
  def compile(self, **kwargs):
    super().compile(
        loss=None,
        metrics=None,
        loss_weights=None,
        weighted_metrics=None,
        **kwargs,
    )
    self.loss = tf.keras.metrics.Mean(name="loss") # tf.keras.metrics.Mean计算给定值的（加权）平均值。
    self.bpp = tf.keras.metrics.Mean(name="bpp")
    self.mse = tf.keras.metrics.Mean(name="mse")

  # 迭代训练模型
  def fit(self, *args, **kwargs):
    retval = super().fit(*args, **kwargs) # 返回值
    # After training, fix range coding tables.
    self.entropy_model = tfc.ContinuousBatchedEntropyModel(
        self.prior, coding_rank=3, compression=True)
    return retval

  @tf.function(input_signature=[
      tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8),
  ])
  def compress(self, x):
    """Compresses an image."""
    # Add batch dimension and cast to float.
    # 维度变化（维度扩展主要是为了适配TensorFlow的函数。或许batch维度可以省略？）
    # 假设输入矩阵维度为[256,256]
    x = tf.expand_dims(x, 0) # tf.expand_dims增加维度，增加了批维度batch [1,256,256] 
    x = tf.cast(x, dtype=tf.float32)
    y = self.analysis_transform(x)
    # Preserve spatial shapes of both image and latents.保留图像和潜在对象的空间形状
    x_shape = tf.shape(x)[1:-1]
    y_shape = tf.shape(y)[1:-1]
    return self.entropy_model.compress(y), x_shape, y_shape

  @tf.function(input_signature=[
      tf.TensorSpec(shape=(1,), dtype=tf.string),
      tf.TensorSpec(shape=(2,), dtype=tf.int32),
      tf.TensorSpec(shape=(2,), dtype=tf.int32),
  ])
  def decompress(self, string, x_shape, y_shape):
    """Decompresses an image."""
    y_hat = self.entropy_model.decompress(string, y_shape)
    x_hat = self.synthesis_transform(y_hat)
    # Remove batch dimension, and crop away any extraneous padding.删除批次维度，并裁剪掉所有无关的填充。
    x_hat = x_hat[0, :x_shape[0], :x_shape[1], :]
    # Then cast back to 8-bit integer.
    return tf.saturate_cast(tf.round(x_hat), tf.uint8)

''' 过滤图片尺寸 '''
def check_image_size(image, patchsize):
  shape = tf.shape(image)
  return shape[0] >= patchsize and shape[1] >= patchsize and shape[-1] == 3 # -1一定是最后一项，三通道

''' 剪裁图片 '''
def crop_image(image, patchsize):
  # random_crop随机裁剪
  image = tf.image.random_crop(image, (patchsize, patchsize, 3))
  return tf.cast(image, tf.float32)

''' 过滤+剪裁获取数据集 '''
def get_dataset(name, split, args):
  """Creates input data pipeline from a TF Datasets dataset."""
  with tf.device("/cpu:0"):
    dataset = tfds.load(name, split=split, shuffle_files=True)
    if split == "train":
      dataset = dataset.repeat()
    # 过滤器
    dataset = dataset.filter(
        lambda x: check_image_size(x["image"], args.patchsize))
    # 裁剪
    dataset = dataset.map(
        lambda x: crop_image(x["image"], args.patchsize))
    dataset = dataset.batch(args.batchsize, drop_remainder=True) # shape：(256,256,3)
  return dataset # shape：(8, 256,256,3)

''' 根据给定地址获取数据集 '''
def get_custom_dataset(split, args):
  """Creates input data pipeline from custom PNG images.从自定义PNG图像创建输入数据管道。"""
  with tf.device("/cpu:0"):
    files = glob.glob(args.train_glob) # args.train_glob路径字符串,类型list
    if not files:
      raise RuntimeError(f"No training images found with glob "
                         f"'{args.train_glob}'.")
    dataset = tf.data.Dataset.from_tensor_slices(files) # from_tensor_slices(从tensor切片读取)
    dataset = dataset.shuffle(len(files), reshuffle_each_iteration=True) # 先shuffle再batch提高随机度
    if split == "train":
      dataset = dataset.repeat()
    dataset = dataset.map(
        # tf.random_crop随机地将张量裁剪为给定的大小.以一致选择的偏移量将一个形状 size 部分从 value 中切出.需要的条件：value.shape >= size.
        # lambda x
        # lambda本质上是个函数功能，是个匿名的函数，表达形式和用法均与一般函数有所不同。普通的函数可以写简单的也可以写复杂的，但lambda函数一般在一行内实现，是个非常简单的函数功能体。
        # 那么，什么时候需要将函数写成lambda形式？
        # 函数功能简单，一句话就可以实现
        # 偶而性使用，不需要考虑复用
        lambda x: crop_image(read_png(x), args.patchsize),
        num_parallel_calls=args.preprocess_threads)
    dataset = dataset.batch(args.batchsize, drop_remainder=True) # 分成 总数/batchsize 个batch
  return dataset


def train(args):
  """Instantiates and trains the model.实例化并训练模型。"""
  if args.check_numerics:
    tf.debugging.enable_check_numerics() # 张量数字有效检查

  model = BLS2017Model(args.lmbda, args.num_filters)
  # 配置训练方法，算bpp、mse、lose的加权平均
  model.compile(
      # optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), # 用优化器传入学习率进行梯度下降
      optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), # 用优化器传入学习率进行梯度下降
  )

  if args.train_glob: # 给了数据集路径（不过滤大小直接裁剪）
    train_dataset = get_custom_dataset("train", args)
    validation_dataset = get_custom_dataset("validation", args)
  else: # 没给数据集路径，默认下载clic数据集（过滤大小、裁剪）
    train_dataset = get_dataset("clic", "train", args)
    validation_dataset = get_dataset("clic", "validation", args)
  validation_dataset = validation_dataset.take(args.max_validation_steps)

  model.fit(
      train_dataset.prefetch(8), # 开启预加载数据
      epochs=args.epochs, # 上限迭代次数（这里设置的1000）
      steps_per_epoch=args.steps_per_epoch,
      validation_data=validation_dataset.cache(),
      validation_freq=1,
      callbacks=[
          tf.keras.callbacks.TerminateOnNaN(),
          tf.keras.callbacks.TensorBoard(
              log_dir=args.train_path,
              histogram_freq=1, update_freq="epoch"), # TensorBoard可视化
          tf.keras.callbacks.experimental.BackupAndRestore(args.train_path), # 断点恢复
      ],
      verbose=int(args.verbose), # 日志显示
  )
  print(args.model_path, 'args.model_path')
  model.save(args.model_path)


def compressAll(args):
  """压缩文件夹的文件"""
  # print(args, 'args')
  files = glob.glob(args.input_folder + '/*png')
  # print(files, 'files')
  perArgs = copy.copy(args) # 浅拷贝，不改变args的值
  # print(perArgs, 'perArgs')
  bpp_list = []
  mse_list = []
  psnr_list = []
  mssim_list = []
  msssim_db_list = []
  # 循环遍历kodak数据集
  for img in files:
    # print(img, 'img')
    # img为图片完整的相对路径
    imgIndexFirst = img.find('/kodim') # 索引
    imgIndexNext = img.find('.png')
    imgName = img[imgIndexFirst: imgIndexNext] # 单独的图片文件名，如kodim01.png
    # print(imgName, 'imgName') # 单独的图片文件名，如/kodim01
    perArgs.input_file = img
    perArgs.output_file = args.output_folder + imgName + '.tfci'
    # print(perArgs, 'perArgs')
    # print(args, 'args')
    bpp, mse, psnr, msssim, msssim_db = perCompress(perArgs)
    print(bpp, mse, psnr, msssim, msssim_db, 'bpp, mse, psnr, msssim, msssim_db')
    bpp_list.append(bpp)
    mse_list.append(mse)
    psnr_list.append(psnr)
    mssim_list.append(msssim)
    msssim_db_list.append(msssim_db)
  print(bpp_list, 'bpp_list')
  print(mse_list, 'mse_list')
  
  bpp_average = mean(bpp_list)
  mse_average = mean(mse_list)
  psnr_average = mean(psnr_list)
  mssim_average = mean(mssim_list)
  msssim_db_average = mean(msssim_db_list)
  
  print(bpp_average, 'bpp_average')
  print(mse_average, 'mse_average')
  print(psnr_average, 'psnr_average')
  print(mssim_average, 'mssim_average')
  print(msssim_db_average, 'msssim_db_average')
  
    
def perCompress(args):
  """Compresses an image."""
  # Load model and use it to compress the image.
  model = tf.keras.models.load_model(args.model_path)
  x = read_png(args.input_file)
  tensors = model.compress(x)

  # Write a binary file with the shape information and the compressed string.
  packed = tfc.PackedTensors()
  packed.pack(tensors)
  with open(args.output_file, "wb") as f:
    f.write(packed.string)

  # If requested, decompress the image and measure performance.
  if args.verbose:
    x_hat = model.decompress(*tensors)

    # Cast to float in order to compute metrics.
    x = tf.cast(x, tf.float32)
    x_hat = tf.cast(x_hat, tf.float32)
    mse = tf.reduce_mean(tf.math.squared_difference(x, x_hat))
    psnr = tf.squeeze(tf.image.psnr(x, x_hat, 255))
    msssim = tf.squeeze(tf.image.ssim_multiscale(x, x_hat, 255))
    msssim_db = -10. * tf.math.log(1 - msssim) / tf.math.log(10.)

    # The actual bits per pixel including entropy coding overhead.
    # 每像素的实际比特数，包括熵编码开销。
    num_pixels = tf.reduce_prod(tf.shape(x)[:-1])
    bpp = len(packed.string) * 8 / num_pixels

    print(f"Mean squared error: {mse:0.4f}")
    print(f"PSNR (dB): {psnr:0.2f}")
    print(f"Multiscale SSIM: {msssim:0.4f}")
    print(f"Multiscale SSIM (dB): {msssim_db:0.2f}")
    print(f"Bits per pixel: {bpp:0.4f}")
    
    return bpp, mse, psnr, msssim, msssim_db

# 原compress方法
def compress(args):
  """Compresses an image."""
  # Load model and use it to compress the image.
  model = tf.keras.models.load_model(args.model_path)
  x = read_png(args.input_file) # kodak数据集shape=(512,768,3)
  tensors = model.compress(x)

  # Write a binary file with the shape information and the compressed string.
  # 这里保存了shape
  packed = tfc.PackedTensors()
  packed.pack(tensors)
  with open(args.output_file, "wb") as f:
    f.write(packed.string)

  # If requested, decompress the image and measure performance.
  if args.verbose:
    x_hat = model.decompress(*tensors)

    # Cast to float in order to compute metrics.
    # x,x_hat都是原图形的shape
    x = tf.cast(x, tf.float32)
    x_hat = tf.cast(x_hat, tf.float32)
    mse = tf.reduce_mean(tf.math.squared_difference(x, x_hat))
    psnr = tf.squeeze(tf.image.psnr(x, x_hat, 255))
    msssim = tf.squeeze(tf.image.ssim_multiscale(x, x_hat, 255))
    msssim_db = -10. * tf.math.log(1 - msssim) / tf.math.log(10.)

    # The actual bits per pixel including entropy coding overhead.
    num_pixels = tf.reduce_prod(tf.shape(x)[:-1])
    bpp = len(packed.string) * 8 / num_pixels

    print(f"Mean squared error: {mse:0.4f}")
    print(f"PSNR (dB): {psnr:0.2f}")
    print(f"Multiscale SSIM: {msssim:0.4f}")
    print(f"Multiscale SSIM (dB): {msssim_db:0.2f}")
    print(f"Bits per pixel: {bpp:0.4f}")


def decompress(args):
  """Decompresses an image."""
  # Load the model and determine the dtypes of tensors required to decompress.
  model = tf.keras.models.load_model(args.model_path)
  dtypes = [t.dtype for t in model.decompress.input_signature]

  # Read the shape information and compressed string from the binary file,
  # and decompress the image using the model.
  with open(args.input_file, "rb") as f:
    packed = tfc.PackedTensors(f.read())
  tensors = packed.unpack(dtypes)
  x_hat = model.decompress(*tensors)

  # Write reconstructed image out as a PNG file.
  write_png(args.output_file, x_hat)


''' 参数设置 '''
def parse_args(argv):
  """Parses command line arguments."""
  # ArgumentParser 对象包含将命令行解析成 Python 数据类型所需的全部信息
  parser = argparse_flags.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  # High-level options.
  parser.add_argument(
      "--verbose", "-V", action="store_true",
      help="Report progress and metrics when training or compressing.")
  parser.add_argument(
      # 训练
      # "--model_path", default="bls2017",
      # "--model_path", default=".bls2017_01", # 第一次训练或基于服务器输入命令压缩,根路径为models(在压缩是需要加上./models，因为根目录不同)
      # "--model_path", default="bls2017_01", # 第一次训练
      # "--model_path", default="bls2017_02", # 第二次训练
      # "--model_path", default="bls2017_03", # 第三次训练
      # "--model_path", default="bls2017_04", # 第四次训练
      # "--model_path", default="bls2017_05", # 第五次训练
      # "--model_path", default="bls2017_06", # 第六次训练
      # "--model_path", default="bls2017_07", # 第七次训练
      
      # 低码率点训练
      # "--model_path", default="bls2017_new1",
      # "--model_path", default="bls2017_new2",
      # "--model_path", default="bls2017_new3",
      # 高码率点训练
      # "--model_path", default="bls2017_new4",
      # "--model_path", default="bls2017_new5",
      # "--model_path", default="bls2017_new6",
      
      # 效果不好的三个点 0.0075、0.015、0.03
      # "--model_path", default="bls2017_model/bls2017_renew2",
      # "--model_path", default="bls2017_model/bls2017_renew3",
      # "--model_path", default="bls2017_model/bls2017_renew4",
      
      # 改变参数，重新训练这三个点
      # "--model_path", default="bls2017_model/bls2017_change3",
      # "--model_path", default="bls2017_model/bls2017_change3_01",
      # "--model_path", default="bls2017_model/bls2017_change2_01",
      # "--model_path", default="bls2017_model/bls2017_change4_01",
      
      
      
      
      
      # "--model_path", default="test",
      
      
      
      
      # 压缩
      # "--model_path", default="./models/bls2017",
      # "--model_path", default="./models/bls2017_01",
      # "--model_path", default="./models/bls2017_02",
      # "--model_path", default="./models/bls2017_03",
      # "--model_path", default="./models/bls2017_04",
      # "--model_path", default="./models/bls2017_05",
      # "--model_path", default="./models/bls2017_06",
      
      # "--model_path", default="./models/bls2017_new1",
      # "--model_path", default="./models/bls2017_new2",
      # "--model_path", default="./models/bls2017_new3",
      # "--model_path", default="./models/bls2017_new4",
      # "--model_path", default="./models/bls2017_new5",
      # "--model_path", default="./models/bls2017_new6",
      
      
      # 效果不好的三个点
      # "--model_path", default="./models/bls2017_model/bls2017_renew2",
      # "--model_path", default="./models/bls2017_model/bls2017_renew3",
      # "--model_path", default="./models/bls2017_model/bls2017_renew4",
      
      # 改变参数，重新训练这三个点
      # "--model_path", default="./models/bls2017_model/bls2017_change3",
      # "--model_path", default="./models/bls2017_model/bls2017_change3_01",
      # "--model_path", default="./models/bls2017_model/bls2017_change2_01",
      "--model_path", default="./models/bls2017_model/bls2017_change4_01",
      
      
      

      
      
      
      
      
      
      
      
      
      help="Path where to save/load the trained model.")
  subparsers = parser.add_subparsers(
      title="commands", dest="command",
      help="What to do: 'train' loads training data and trains (or continues "
           "to train) a new model. 'compress' reads an image file (lossless "
           "PNG format) and writes a compressed binary file. 'decompress' "
           "reads a binary file and reconstructs the image (in PNG format). "
           "input and output filenames need to be provided for the latter "
           "two options. Invoke '<command> -h' for more information.")

  # 'train' subcommand.
  train_cmd = subparsers.add_parser(
      "train",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      description="Trains (or continues to train) a new model. Note that this "
                  "model trains on a continuous stream of patches drawn from "
                  "the training image dataset. An epoch is always defined as "
                  "the same number of batches given by --steps_per_epoch. "
                  "The purpose of validation is mostly to evaluate the "
                  "rate-distortion performance of the model using actual "
                  "quantization rather than the differentiable proxy loss. "
                  "Note that when using custom training images, the validation "
                  "set is simply a random sampling of patches from the "
                  "training set.")
  train_cmd.add_argument(
      # 0.01\0.02\0.04\0.06\0.09\1.1\0.005 # 第一次失败的几个点
      # 新增几个lambda0.0016、0.0032、0.0075对应滤波器数量num_filters=128
      # 0.015、0.03、0.045，对应滤波器数量num_filters=192
      "--lambda", type=float, default=0.015, dest="lmbda",
      help="Lambda for rate-distortion tradeoff.")
  train_cmd.add_argument(
      "--train_glob", type=str, default=None,
      help="Glob pattern identifying custom training data. This pattern must "
           "expand to a list of RGB images in PNG format. If unspecified, the "
           "CLIC dataset from TensorFlow Datasets is used.")
  train_cmd.add_argument(
      # "--num_filters", type=int, default=128, # 低码率
      "--num_filters", type=int, default=192, # 高码率
      help="Number of filters per layer.")
  train_cmd.add_argument(
      # "--train_path", default="/tmp/train_bls2017",
      "--train_path", default="/tmp/train_bls2017_testlog",
      help="Path where to log training metrics for TensorBoard and back up "
           "intermediate model checkpoints.")
  train_cmd.add_argument(
      "--batchsize", type=int, default=8,
      help="Batch size for training and validation.")
  train_cmd.add_argument(
      "--patchsize", type=int, default=256,
      help="Size of image patches for training and validation.")
  train_cmd.add_argument(
      "--epochs", type=int, default=1000,
      help="Train up to this number of epochs. (One epoch is here defined as "
           "the number of steps given by --steps_per_epoch, not iterations "
           "over the full training dataset.)")
  train_cmd.add_argument(
      "--steps_per_epoch", type=int, default=1000,
      help="Perform validation and produce logs after this many batches.")
  train_cmd.add_argument(
      "--max_validation_steps", type=int, default=16,
      help="Maximum number of batches to use for validation. If -1, use one "
           "patch from each image in the training set.")
  train_cmd.add_argument(
      # 进程数（默认16）
      "--preprocess_threads", type=int, default=16,
      help="Number of CPU threads to use for parallel decoding of training "
           "images.")
  train_cmd.add_argument(
      "--check_numerics", action="store_true",
      help="Enable TF support for catching NaN and Inf in tensors.")

  # 'compress' subcommand.
  compress_cmd = subparsers.add_parser(
      "compress",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      description="Reads a PNG file, compresses it, and writes a TFCI file.")

  # 'decompress' subcommand.
  decompress_cmd = subparsers.add_parser(
      "decompress",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      description="Reads a TFCI file, reconstructs the image, and writes back "
                  "a PNG file.")

  # Arguments for both 'compress' and 'decompress'.
  for cmd, ext in ((compress_cmd, ".tfci"), (decompress_cmd, ".png")):
    cmd.add_argument(
        "input_file",
        help="Input filename.")
    cmd.add_argument(
        "output_file", nargs="?",
        help=f"Output filename (optional). If not provided, appends '{ext}' to "
             f"the input filename.")
    
  # 'compressAll' subcommand.
  compressAll_cmd = subparsers.add_parser(
      "compressAll",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      description="读取文件下的文件进行压缩操作")
  
  # Arguments for 'compressAll'.
  compressAll_cmd.add_argument(
    "input_folder",
    help="输入文件夹.")
  compressAll_cmd.add_argument(
    "output_folder",
    help="输出文件夹.")

  # Parse arguments.
  args = parser.parse_args(argv[1:])
  if args.command is None:
    parser.print_usage()
    sys.exit(2)
  return args


def main(args):
  # Invoke subcommand.
  if args.command == "train":
    train(args)
  elif args.command == "compress":
    if not args.output_file:
      args.output_file = args.input_file + ".tfci"
    compress(args)
  elif args.command == "compressAll":
    compressAll(args)
  elif args.command == "decompress":
    if not args.output_file:
      args.output_file = args.input_file + ".png"
    decompress(args)


if __name__ == "__main__":
  app.run(main, flags_parser=parse_args)
