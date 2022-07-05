# 超先验模型
# Copyright 2019 Google LLC. All Rights Reserved.
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
"""Nonlinear transform coder with hyperprior for RGB images.

This is the image compression model published in:
J. Ballé, D. Minnen, S. Singh, S.J. Hwang, N. Johnston:
"Variational Image Compression with a Scale Hyperprior"
Int. Conf. on Learning Representations (ICLR), 2018
https://arxiv.org/abs/1802.01436

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
os.environ["OMP_NUM_THREADS"] = "6" # cpu核数
tf.config.threading.set_intra_op_parallelism_threads(6)
tf.config.threading.set_inter_op_parallelism_threads(6)

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
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_compression as tfc


from numpy import *
import copy


def read_png(filename):
  """Loads a PNG image file."""
  string = tf.io.read_file(filename)
  return tf.image.decode_image(string, channels=3)


def write_png(filename, image):
  """Saves an image to a PNG file."""
  string = tf.image.encode_png(image)
  tf.io.write_file(filename, string)


class AnalysisTransform(tf.keras.Sequential):
  """The analysis transform."""

  def __init__(self, num_filters):
    super().__init__(name="analysis")
    self.add(tf.keras.layers.Lambda(lambda x: x / 255.))
    self.add(tfc.SignalConv2D(
        num_filters, (5, 5), name="layer_0", corr=True, strides_down=2,
        padding="same_zeros", use_bias=True,
        activation=tfc.GDN(name="gdn_0")))
    self.add(tfc.SignalConv2D(
        num_filters, (5, 5), name="layer_1", corr=True, strides_down=2,
        padding="same_zeros", use_bias=True,
        activation=tfc.GDN(name="gdn_1")))
    self.add(tfc.SignalConv2D(
        num_filters, (5, 5), name="layer_2", corr=True, strides_down=2,
        padding="same_zeros", use_bias=True,
        activation=tfc.GDN(name="gdn_2")))
    self.add(tfc.SignalConv2D(
        num_filters, (5, 5), name="layer_3", corr=True, strides_down=2,
        padding="same_zeros", use_bias=True,
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
        num_filters, (5, 5), name="layer_2", corr=False, strides_up=2,
        padding="same_zeros", use_bias=True,
        activation=tfc.GDN(name="igdn_2", inverse=True)))
    self.add(tfc.SignalConv2D(
        3, (5, 5), name="layer_3", corr=False, strides_up=2,
        padding="same_zeros", use_bias=True,
        activation=None))
    self.add(tf.keras.layers.Lambda(lambda x: x * 255.))

'''超先验模型的非线性分析变换'''
class HyperAnalysisTransform(tf.keras.Sequential):
  """The analysis transform for the entropy model parameters."""

  def __init__(self, num_filters):
    super().__init__(name="hyper_analysis")
    self.add(tfc.SignalConv2D(
        num_filters, (3, 3), name="layer_0", corr=True, strides_down=1,
        padding="same_zeros", use_bias=True,
        activation=tf.nn.relu))
    self.add(tfc.SignalConv2D(
        num_filters, (5, 5), name="layer_1", corr=True, strides_down=2,
        padding="same_zeros", use_bias=True,
        activation=tf.nn.relu))
    self.add(tfc.SignalConv2D(
        num_filters, (5, 5), name="layer_2", corr=True, strides_down=2,
        padding="same_zeros", use_bias=False,
        activation=None))

'''超先验模型的非线性综合变换'''
class HyperSynthesisTransform(tf.keras.Sequential):
  """The synthesis transform for the entropy model parameters."""

  def __init__(self, num_filters):
    super().__init__(name="hyper_synthesis")
    self.add(tfc.SignalConv2D(
        num_filters, (5, 5), name="layer_0", corr=False, strides_up=2,
        padding="same_zeros", use_bias=True, kernel_parameter="variable",
        activation=tf.nn.relu))
    self.add(tfc.SignalConv2D(
        num_filters, (5, 5), name="layer_1", corr=False, strides_up=2,
        padding="same_zeros", use_bias=True, kernel_parameter="variable",
        activation=tf.nn.relu))
    self.add(tfc.SignalConv2D(
        num_filters, (3, 3), name="layer_2", corr=False, strides_up=1,
        padding="same_zeros", use_bias=True, kernel_parameter="variable",
        activation=None))


class BMSHJ2018Model(tf.keras.Model):
  """Main model class."""

  def __init__(self, lmbda, num_filters, num_scales, scale_min, scale_max):
    super().__init__()
    self.lmbda = lmbda
    # 构建分组边界，相关变量在LocationScaleIndexedEntropyModel中使用
    # num_scales：整数。值indexes必须在范围内 。 [0, num_scales)
    # scale_fn：可调用。indexes被传递给可调用对象，返回值作为scale关键字参数给出。
    self.num_scales = num_scales
    offset = tf.math.log(scale_min)
    factor = (tf.math.log(scale_max) - tf.math.log(scale_min)) / (
        num_scales - 1.)
    self.scale_fn = lambda i: tf.math.exp(offset + factor * i)
    # 提取特征y的变换网络
    self.analysis_transform = AnalysisTransform(num_filters)
    self.synthesis_transform = SynthesisTransform(num_filters)
    # 超先验网络的变换
    self.hyper_analysis_transform = HyperAnalysisTransform(num_filters)
    self.hyper_synthesis_transform = HyperSynthesisTransform(num_filters)
    # 计算先验概率
    self.hyperprior = tfc.NoisyDeepFactorized(batch_shape=(num_filters,))
    self.build((None, None, None, 3))

  def call(self, x, training):
    """Computes rate and distortion losses."""
    # 位置尺度随机变量族的索引熵模型，指定的分布使用num_scales比例参数的值进行参数化。通过将分布移至零来处理逐个元素的位置参数。
    entropy_model = tfc.LocationScaleIndexedEntropyModel(
        tfc.NoisyNormal, self.num_scales, self.scale_fn, coding_rank=3,
        compression=False)
    # 连续随机变量的批处理熵模型，该库中的熵模型类简化了设计率失真优化代码的过程。在训练期间，它们的行为类似于似然模型。
    # 边信息的建模是通过全分解的熵模型建模（即balle2017的熵模型建模获取p(y)的方式）
    side_entropy_model = tfc.ContinuousBatchedEntropyModel(
        self.hyperprior, coding_rank=3, compression=False)

    y = self.analysis_transform(x)
    z = self.hyper_analysis_transform(abs(y))
    z_hat, side_bits = side_entropy_model(z, training=training)
    indexes = self.hyper_synthesis_transform(z_hat)
    y_hat, bits = entropy_model(y, indexes, training=training)
    x_hat = self.synthesis_transform(y_hat)

    # Total number of bits divided by total number of pixels.
    #  tf.reduce_prod 计算一个张量的各个维度上元素的乘积.（长乘宽）
    num_pixels = tf.cast(tf.reduce_prod(tf.shape(x)[:-1]), bits.dtype)
    # 这里的bit是边信息与原信息的比特数
    bpp = (tf.reduce_sum(bits) + tf.reduce_sum(side_bits)) / num_pixels
    # Mean squared error across pixels.
    mse = tf.reduce_mean(tf.math.squared_difference(x, x_hat))
    # The rate-distortion Lagrangian.
    loss = bpp + self.lmbda * mse
    return loss, bpp, mse

  def train_step(self, x):
    with tf.GradientTape() as tape:
      loss, bpp, mse = self(x, training=True)
    variables = self.trainable_variables
    gradients = tape.gradient(loss, variables)
    self.optimizer.apply_gradients(zip(gradients, variables))
    self.loss.update_state(loss)
    self.bpp.update_state(bpp)
    self.mse.update_state(mse)
    return {m.name: m.result() for m in [self.loss, self.bpp, self.mse]}

  def test_step(self, x):
    loss, bpp, mse = self(x, training=False)
    self.loss.update_state(loss)
    self.bpp.update_state(bpp)
    self.mse.update_state(mse)
    return {m.name: m.result() for m in [self.loss, self.bpp, self.mse]}

  def predict_step(self, x):
    raise NotImplementedError("Prediction API is not supported.")

  def compile(self, **kwargs):
    super().compile(
        loss=None,
        metrics=None,
        loss_weights=None,
        weighted_metrics=None,
        **kwargs,
    )
    self.loss = tf.keras.metrics.Mean(name="loss")
    self.bpp = tf.keras.metrics.Mean(name="bpp")
    self.mse = tf.keras.metrics.Mean(name="mse")

  def fit(self, *args, **kwargs):
    retval = super().fit(*args, **kwargs)
    # After training, fix range coding tables.
    self.entropy_model = tfc.LocationScaleIndexedEntropyModel(
        tfc.NoisyNormal, self.num_scales, self.scale_fn, coding_rank=3,
        compression=True)
    self.side_entropy_model = tfc.ContinuousBatchedEntropyModel(
        self.hyperprior, coding_rank=3, compression=True)
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
    z = self.hyper_analysis_transform(abs(y))
    # Preserve spatial shapes of image and latents.保留图像和潜在对象的空间形状
    x_shape = tf.shape(x)[1:-1]
    y_shape = tf.shape(y)[1:-1]
    z_shape = tf.shape(z)[1:-1]
    z_hat, _ = self.side_entropy_model(z, training=False)
    indexes = self.hyper_synthesis_transform(z_hat)
    indexes = indexes[:, :y_shape[0], :y_shape[1], :]
    side_string = self.side_entropy_model.compress(z) # side_entropy_model.compress:Compresses the tensor to bit strings.模型自带compress方法
    string = self.entropy_model.compress(y, indexes)
    return string, side_string, x_shape, y_shape, z_shape

  @tf.function(input_signature=[
      tf.TensorSpec(shape=(1,), dtype=tf.string),
      tf.TensorSpec(shape=(1,), dtype=tf.string),
      tf.TensorSpec(shape=(2,), dtype=tf.int32),
      tf.TensorSpec(shape=(2,), dtype=tf.int32),
      tf.TensorSpec(shape=(2,), dtype=tf.int32),
  ])
  def decompress(self, string, side_string, x_shape, y_shape, z_shape):
    """Decompresses an image."""
    # 加入边信息进行解压缩
    z_hat = self.side_entropy_model.decompress(side_string, z_shape)
    indexes = self.hyper_synthesis_transform(z_hat)
    indexes = indexes[:, :y_shape[0], :y_shape[1], :]
    y_hat = self.entropy_model.decompress(string, indexes)
    x_hat = self.synthesis_transform(y_hat)
    # Remove batch dimension, and crop away any extraneous padding.
    x_hat = x_hat[0, :x_shape[0], :x_shape[1], :]
    # Then cast back to 8-bit integer.
    return tf.saturate_cast(tf.round(x_hat), tf.uint8)

# 过滤图片尺寸大于patchsize且三通道
def check_image_size(image, patchsize):
  shape = tf.shape(image)
  return shape[0] >= patchsize and shape[1] >= patchsize and shape[-1] == 3 # -1一定是最后一项，三通道


def crop_image(image, patchsize):
  image = tf.image.random_crop(image, (patchsize, patchsize, 3))
  return tf.cast(image, tf.float32)


def get_dataset(name, split, args):
  """Creates input data pipeline from a TF Datasets dataset."""
  with tf.device("/cpu:0"):
    dataset = tfds.load(name, split=split, shuffle_files=True)
    if split == "train":
      dataset = dataset.repeat()
    dataset = dataset.filter(
        lambda x: check_image_size(x["image"], args.patchsize))
    dataset = dataset.map(
        lambda x: crop_image(x["image"], args.patchsize))
    dataset = dataset.batch(args.batchsize, drop_remainder=True)
  return dataset


def get_custom_dataset(split, args):
  """Creates input data pipeline from custom PNG images."""
  with tf.device("/cpu:0"):
    files = glob.glob(args.train_glob)
    if not files:
      raise RuntimeError(f"No training images found with glob "
                         f"'{args.train_glob}'.")
    dataset = tf.data.Dataset.from_tensor_slices(files)
    dataset = dataset.shuffle(len(files), reshuffle_each_iteration=True)
    if split == "train":
      dataset = dataset.repeat()
    dataset = dataset.map(
        lambda x: crop_image(read_png(x), args.patchsize),
        num_parallel_calls=args.preprocess_threads)
    dataset = dataset.batch(args.batchsize, drop_remainder=True)
  return dataset


def train(args):
  """Instantiates and trains the model."""
  if args.check_numerics:
    tf.debugging.enable_check_numerics()

  model = BMSHJ2018Model(
      args.lmbda, args.num_filters, args.num_scales, args.scale_min,
      args.scale_max)
  model.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
  )

  if args.train_glob:
    train_dataset = get_custom_dataset("train", args)
    validation_dataset = get_custom_dataset("validation", args)
  else:
    train_dataset = get_dataset("clic", "train", args)
    validation_dataset = get_dataset("clic", "validation", args)
  validation_dataset = validation_dataset.take(args.max_validation_steps)

  model.fit(
      train_dataset.prefetch(8),
      epochs=args.epochs,
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
      verbose=int(args.verbose),
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
  
# 压缩每一张图片，返回对应值
def perCompress(args):
  """Compresses an image."""
  # Load model and use it to compress the image.
  model = tf.keras.models.load_model(args.model_path)
  x = read_png(args.input_file) # kodak数据集shape=(512,768,3)
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
    num_pixels = tf.reduce_prod(tf.shape(x)[:-1])
    bpp = len(packed.string) * 8 / num_pixels

    print(f"Mean squared error: {mse:0.4f}")
    print(f"PSNR (dB): {psnr:0.2f}")
    print(f"Multiscale SSIM: {msssim:0.4f}")
    print(f"Multiscale SSIM (dB): {msssim_db:0.2f}")
    print(f"Bits per pixel: {bpp:0.4f}")
    return bpp, mse, psnr, msssim, msssim_db
    
#  原compress方法
def compress(args):
  """Compresses an image."""
  # Load model and use it to compress the image.
  model = tf.keras.models.load_model(args.model_path)
  x = read_png(args.input_file) # kodak数据集shape=(512,768,3)
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


def parse_args(argv):
  """Parses command line arguments."""
  parser = argparse_flags.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  # High-level options.
  parser.add_argument(
      "--verbose", "-V", action="store_true",
      help="Report progress and metrics when training or compressing.")
  parser.add_argument(
      # "--model_path", default="bmshj2018",
      # 训练
      # "--model_path", default="bmshj2018Model/bmshj2018_test",
      # "--model_path", default="bmshj2018Model/bmshj2018_01",
      
      
      
      # 压缩
      # "--model_path", default="./models/bmshj2018Model/bmshj2018_test",
      "--model_path", default="./models/bmshj2018Model/bmshj2018_01",
      
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
      # 几个lambda0.0016、0.0032、0.0075对应滤波器数量num_filters=128
      # 0.015、0.03、0.045，对应滤波器数量num_filters=192
      # 初始0.01
      "--lambda", type=float, default=0.0016, dest="lmbda",
      help="Lambda for rate-distortion tradeoff.")
  train_cmd.add_argument(
      "--train_glob", type=str, default=None,
      help="Glob pattern identifying custom training data. This pattern must "
           "expand to a list of RGB images in PNG format. If unspecified, the "
           "CLIC dataset from TensorFlow Datasets is used.")
  train_cmd.add_argument(
      # 初始192
      "--num_filters", type=int, default=128,
      help="Number of filters per layer.")
  train_cmd.add_argument(
      "--num_scales", type=int, default=64,
      help="Number of Gaussian scales to prepare range coding tables for.")
  train_cmd.add_argument(
      "--scale_min", type=float, default=.11,
      # 高斯标准差最小值
      help="Minimum value of standard deviation of Gaussians.")
  train_cmd.add_argument(
      "--scale_max", type=float, default=256.,
      help="Maximum value of standard deviation of Gaussians.")
  train_cmd.add_argument(
      "--train_path", default="/tmp/train_bmshj2018",
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
