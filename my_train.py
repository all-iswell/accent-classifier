# -*- coding: utf-8 -*-
import sugartensor as tf
from data import VCTK


__author__ = 'buriburisuri@gmail.com'


# set log level to debug
tf.sg_verbosity(10)


#
# hyper parameters
#

batch_size = 16    # batch size
num_blocks = 3     # dilated blocks
num_dim = 128      # latent dimension

#
# inputs
#

# VCTK corpus input tensor ( with QueueRunner )
data = VCTK(batch_size=batch_size, data_path='/Volumes/Warehouse/VCTK-Corpus/')

# vocabulary size
voca_size = data.voca_size

# CLASSIFY SIZE
classify_size = 2

# mfcc feature of audio
x = data.mfcc

# sequence length except zero-padding
seq_len = tf.not_equal(x.sg_sum(dims=2), 0.).sg_int().sg_sum(dims=1)

# target sentence label
#y = data.label

# target classification label
y = data.label

print("SHAPE", tf.shape(y))

#
# encode graph ( atrous convolution )
#

# residual block
def res_block(tensor, size, rate, dim=num_dim):

    # filter convolution
    conv_filter = tensor.sg_aconv1d(size=size, rate=rate, act='tanh', bn=True,dout=0.05)

    # gate convolution
    conv_gate = tensor.sg_aconv1d(size=size, rate=rate,  act='sigmoid', bn=True,dout=0.05)

    # output by gate multiplying
    out = conv_filter * conv_gate

    # final output
    out = out.sg_conv1d(size=1, dim=dim, act='tanh', bn=True, dout=0.20)

    # residual and skip output
    return out + tensor, out

# expand dimension
z = x.sg_conv1d(size=1, dim=num_dim, act='tanh', bn=True, dout=0.20)

# dilated conv block loop
skip = 0  # skip connections
for i in range(num_blocks):
    for r in [1, 2, 4, 8, 16]:
        z, s = res_block(z, size=7, rate=r)
        skip += s

# final logit layers
logit = (skip
         .sg_conv1d(size=1, act='tanh', bn=True, dout=0.25)
         .sg_conv1d(size=1, act='tanh', bn=True, dim=20, dout=0.25)
         .sg_conv1d(size=1, act='tanh', bn=True, dim=2, dout=0.25)
         .sg_reshape(shape=(batch_size,200))
         .sg_dense(in_dim=200, dim=50, act='relu', dout=0.20)
         .sg_dense(in_dim=50, dim=10, act='relu', dout=0.20)
         .sg_dense(in_dim=10, dim=2, dout=0.20))

# CTC loss
#loss = logit.sg_ctc(target=y, seq_len=seq_len)
reg_lambda = 0.0002
trainable = tf.trainable_variables()
lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in trainable])*reg_lambda
loss = logit.sg_ce(target=y, one_hot=True) + lossL2

# train
config = tf.ConfigProto(allow_soft_placement=True,
    inter_op_parallelism_threads=6,
    intra_op_parallelism_threads=6)
sess = tf.Session(config=config)
tf.sg_init(sess)

learning_rate = tf.train.exponential_decay(0.00001, tf.sg_global_step(), 100,
    0.95, staircase=False)

with tf.name_scope('summaries'):
    tf.summary.scalar('global_step', tf.sg_global_step())
    tf.summary.scalar('real_lr', learning_rate)

tf.sg_train(log_interval=30, lr=learning_rate, loss=loss,
            ep_size=data.num_batch, max_ep=8, early_stop=False, lr_reset=True)
