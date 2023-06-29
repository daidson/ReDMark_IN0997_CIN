import numpy as np
import tensorflow as tf


def dot(x, y, sparse=False):
    """
    Wrapper for tf.matmul (sparse vs dense).
    Source: https://github.com/tkipf/gcn
    """
    if sparse:
        res = tf.sparse.sparse_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


def sparse_dropout(x, keep_prob, noise_shape):
    """
    Dropout for sparse tensors.
    Source: https://github.com/tkipf/gcn
    """
    random_tensor = keep_prob
    random_tensor += tf.random.uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse.retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


def partitioning(unpartitioned, p_size):
    up_size = unpartitioned.shape[0]
    parts = np.zeros((int((up_size/p_size)**2), p_size, p_size, 1))
    n=0
    for u in range(int(up_size/p_size)):
        for v in range(int(up_size/p_size)):
            parts[n,:,:,0] = unpartitioned[p_size*u:p_size*u+p_size , p_size*v:p_size*v+p_size]
            n = n + 1
    return parts


def tiling(partitions, rec_size):
    p_size = partitions.shape[1]
    reconstructed_img = np.zeros((rec_size,rec_size))
    n=0
    for u in range(int(rec_size/p_size)):
        for v in range(int(rec_size/p_size)):
            reconstructed_img[p_size*u:p_size*u+p_size , p_size*v:p_size*v+p_size] = partitions[n,:,:,0]
            n = n + 1
    return reconstructed_img


def W_tiling(partitions, rec_size, num_bitplane):
    p_size = partitions.shape[1]
    reconstructed_W = np.zeros((rec_size, rec_size, num_bitplane))
    n=0
    for u in range(int(rec_size/p_size)):
        for v in range(int(rec_size/p_size)):
            reconstructed_W[p_size*u:p_size*u+p_size , p_size*v:p_size*v+p_size, :] = partitions[n,:,:,:]
            n = n + 1
    return reconstructed_W