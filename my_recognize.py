# -*- coding: utf-8 -*-
import sugartensor as tf
import numpy as np
import librosa
from data import VCTK


__author__ = 'buriburisuri@gmail.com'


# set log level to debug
tf.sg_verbosity(10)

#
# hyper parameters
#

batch_size = 1     # batch size
num_blocks = 3     # dilated blocks
num_dim = 128      # latent dimension

#
# inputs
#

# VCTK corpus input tensor ( with QueueRunner )
data = VCTK(batch_size=batch_size, data_path='/Volumes/Warehouse/VCTK-Corpus/',
    mode='train')

# vocabulary size
voca_size = data.voca_size

# mfcc feature of audio
x = tf.placeholder(dtype=tf.sg_floatx, shape=(batch_size, None, 40))

# sequence length except zero-padding
seq_len = tf.not_equal(x.sg_sum(dims=2), 0.).sg_int().sg_sum(dims=1)


#
# encode graph ( atrous convolution )
#

# residual block
def res_block(tensor, size, rate, dim=num_dim):

    # filter convolution
    conv_filter = tensor.sg_aconv1d(size=size, rate=rate, act='tanh', bn=True)

    # gate convolution
    conv_gate = tensor.sg_aconv1d(size=size, rate=rate,  act='sigmoid', bn=True)

    # output by gate multiplying
    out = conv_filter * conv_gate

    # final output
    out = out.sg_conv1d(size=1, dim=dim, act='tanh', bn=True)

    # residual and skip output
    return out + tensor, out

# expand dimension
z = x.sg_conv1d(size=1, dim=num_dim, act='tanh', bn=True)

# dilated conv block loop
skip = 0  # skip connections
for i in range(num_blocks):
    for r in [1, 2, 4, 8, 16]:
        z, s = res_block(z, size=7, rate=r)
        skip += s

# final logit layers
# logit = (skip
#          .sg_conv1d(size=1, act='tanh', bn=True)
#          .sg_conv1d(size=1, act='tanh', dim=5, bn=True)
#          .sg_conv1d(size=1, dim=2, bn=True)
#          .sg_reshape(shape=(batch_size,200))
#          .sg_dense(in_dim=200, dim=50, act='relu')
#          .sg_dense(in_dim=50, dim=10, act='relu')
#          .sg_dense(in_dim=10, dim=2))
#logit = (skip
#         .sg_conv1d(size=1, act='tanh', bn=True)
#         .sg_conv1d(size=1, dim=2)
#         .sg_reshape(shape=(batch_size,200))
#         .sg_dense(in_dim=200, dim=50, act='tanh')
#         .sg_dense(in_dim=50, dim=10, act='tanh')
#         .sg_dense(in_dim=10, dim=2, act='tanh'))
logit = (skip
         .sg_conv1d(size=1, act='tanh', bn=True)
         .sg_conv1d(size=1, act='tanh', bn=True, dim=20)
         .sg_conv1d(size=1, act='tanh', bn=True, dim=2)
         .sg_reshape(shape=(batch_size,200))
         .sg_dense(in_dim=200, dim=50, act='relu')
         .sg_dense(in_dim=50, dim=10, act='relu')
         .sg_dense(in_dim=10, dim=2))

# ctc decoding
#decoded, _ = tf.nn.ctc_beam_search_decoder(logit.sg_transpose(perm=[1, 0, 2]), seq_len, merge_repeated=False)

# to dense tensor
#y = tf.sparse_to_dense(decoded[0].indices, decoded[0].shape, decoded[0].values) + 1

y = tf.nn.softmax(logit)

#
# regcognize wave file
#

# command line argument for input wave file path
tf.sg_arg_def(file=('', 'speech wave file to recognize.'))

# load wave file
wav, sr = librosa.load(tf.sg_arg().file, mono=True)
# get mfcc feature
mfcc = np.transpose(np.expand_dims(librosa.feature.mfcc(wav, sr,n_mfcc=20), axis=0), [0, 2, 1])
mfcc = mfcc[:,:100]

# run network
with tf.Session() as sess:

    # init variables
    tf.sg_init(sess)

    # restore parameters
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint('asset/train/ckpt'))

    # run session
    #label = sess.run(y, feed_dict={x: mfcc})
    total = 0
    good = 0
    for i in range(0, len(data.labels)):
        wav, sr = librosa.load(data.wave_files[i], mono=True)
        mfcc = np.transpose(np.expand_dims(librosa.feature.mfcc(wav, sr,n_mfcc=40), axis=0), [0, 2, 1])
        mfcc = mfcc[:,:100]
        res = sess.run(y, feed_dict={x: mfcc})

        total += 1
        if np.argmax(res) == np.argmax(data.labels[i]):
            good += 1

        if total % 20 == 0:
            print(total)

    print(good, total)

    # print label
    #data.print_index(label)
