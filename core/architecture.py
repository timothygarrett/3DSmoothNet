# architecture.py ---
#
# Filename: architecture.py
# Description: defines the architecture of the 3DSmoothNet
# Author: Zan Gojcic, Caifa Zhou
#
# Project: 3DSmoothNet https://github.com/zgojcic/3DSmoothNet
# Created: 04.04.2019
# Version: 1.0

# Copyright (C)
# IGP @ ETHZ

# Code:

# Import python dependencies
import tensorflow as tf
import numpy as np

# Import custom functions
from core import ops


def network_architecture(x_anc,x_pos, dropout_rate, config, reuse=False):

    # Join the 3DSmoothNet structure with the desired output dimension
    # net_structure = [1, 32, 32, 64, 64, 128, 128]
    # net_structure = [1, 128, 256, 512]
    net_structure = [1, 32, 64, 128]
    # net_structure = [1, 8, 16, 32]
    outputDim = config.output_dim
    channels = [item for sublist in [net_structure, [outputDim]] for item in sublist]

    # In the third layer stride is 2
    stride = np.ones(len(channels))
    # stride[2] = 2

    # Apply dropout in the 6th layer
    dropout_flag = np.zeros(len(channels))
    # dropout_flag[5] = 1

    # Apply max pool to the first contracting blocks
    max_pool_flag = np.zeros(len(channels))
    max_pool_flag[0] = 1
    max_pool_filter_stride = {16: None,
                              8: [2, 2],
                              4: [4, 4]}

    # Initalize data
    layer_index = 0
    input_anc_16 = x_anc
    input_pos_16 = x_pos
    input_anc_8 = x_anc
    input_pos_8 = x_pos
    input_anc_4 = x_anc
    input_pos_4 = x_pos

    # shortcut_blocks = {}
    with tf.name_scope('3DIM_cnn') as scope:

        # Build contracting blocks
        for layer in range(len(net_structure) - 1):
            dropout_flag = 0 if layer < len(net_structure) - 2 else 1

            # Build 16-sided contracting blocks
            input_anc_16, input_pos_16 = contracting_block(input_anc_16,
                                                           input_pos_16,
                                                           [channels[layer], channels[layer + 1]],
                                                           16,
                                                           dropout_flag,
                                                           dropout_rate,
                                                           layer_index + layer,
                                                           max_pool_flag[layer],
                                                           max_pool_filter_stride[16],
                                                           stride_input=1,
                                                           reuse=reuse)

            # Build 8-sided contracting blocks
            input_anc_8, input_pos_8 = contracting_block(input_anc_8,
                                                         input_pos_8,
                                                         [channels[layer], channels[layer + 1]],
                                                         8,
                                                         dropout_flag,
                                                         dropout_rate,
                                                         layer_index + layer,
                                                         max_pool_flag[layer],
                                                         max_pool_filter_stride[8],
                                                         stride_input=1,
                                                         reuse=reuse)

            # Build 4-sided contracting blocks
            input_anc_4, input_pos_4 = contracting_block(input_anc_4,
                                                         input_pos_4,
                                                         [channels[layer], channels[layer + 1]],
                                                         4,
                                                         dropout_flag,
                                                         dropout_rate,
                                                         layer_index + layer,
                                                         max_pool_flag[layer],
                                                         max_pool_filter_stride[4],
                                                         stride_input=1,
                                                         reuse=reuse)

            # if layer == 0:
            #     shortcut_blocks[16] = (input_anc_16, input_pos_16)
            #     shortcut_blocks[8] = (input_anc_8, input_pos_8)
            #     shortcut_blocks[4] = (input_anc_4, input_pos_4)

        layer_index += (len(net_structure) - 1)

        # Downsample the 16-sided volumes
        input_anc_16, input_pos_16 = max_pool(input_anc_16, input_pos_16, 2, 2)

        # Upsample the 4-sided volumes
        input_anc_4, input_pos_4 = conv_transpose_block(input_anc_4, input_pos_4,
                                                        [net_structure[-1], net_structure[-1]], 2, 0, 0.0, layer_index,
                                                        reuse=reuse)
        layer_index += 1

        # Add the shortcuts
        # sh_anc_16, sh_pos_16 = shortcut_blocks[16]
        # sh_anc_16, sh_pos_16 = max_pool(sh_anc_16, sh_pos_16, 2, 2)
        # sh_anc_8, sh_pos_8 = shortcut_blocks[8]
        # sh_anc_4, sh_pos_4 = shortcut_blocks[4]
        # sh_anc_4, sh_pos_4 = conv_transpose_block(sh_anc_4, sh_pos_4,
        #                                           [net_structure[1], net_structure[1]], 2, 0, 0.0, layer_index,
        #                                           reuse=reuse)
        # layer_index += 1
        input_ancs = [input_anc_16, input_anc_8, input_anc_4]
        # shortcut_ancs = [sh_anc_16, sh_anc_8, sh_anc_4, input_anc_16, input_anc_8, input_anc_4]
        input_poses = [input_pos_16, input_pos_8, input_pos_4]
        # shortcut_poses = [sh_pos_16, sh_pos_8, sh_pos_4, input_pos_16, input_pos_8, input_pos_4]

        # Concatenate
        input_anc, input_pos = concatenate(input_ancs, input_poses, layer_index)
        # input_anc, input_pos = concatenate(shortcut_ancs, shortcut_poses, layer_index)
        layer_index += 1

        # Encode into a descriptor
        input_anc, input_pos = encode_block(input_anc, input_pos, channels[-1], layer_index, reuse=reuse)

        # Normalize
        return ops.l2_normalize(input_anc), ops.l2_normalize(input_pos)


    # # Loop over the desired layers
    # with tf.name_scope('3DIM_cnn') as scope:
    #     for layer in np.arange(0, len(channels)-2):
    #         scope_name = "3DIM_cnn" + str(layer_index+1)
    #         with tf.name_scope(scope_name) as inner_scope:
    #             input_anc, input_pos = conv_block(input_anc, input_pos, [channels[layer], channels[layer + 1]],
    #                                               dropout_flag[layer], dropout_rate, layer_index,
    #                                               stride_input=stride[layer], reuse=reuse)
    #
    #         layer_index += 1
    #
    #     with tf.name_scope('3DIM_cnn7') as inner_scope:
    #         input_anc, input_pos = out_block(input_anc, input_pos, [channels[-2], channels[-1]],
    #                                          layer_index, reuse=reuse)
    #
    #     return ops.l2_normalize(input_anc), \
    #            ops.l2_normalize(input_pos)


def concatenate(input_ancs, input_poses, layer_idx):
    # assert len(input_ancs) == 6 and len(input_poses) == 6
    assert len(input_ancs) == 3 and len(input_poses) == 3

    with tf.name_scope('concat_{}'.format(layer_idx)) as scope:
        out_anc = tf.concat(input_ancs, -1)
        out_pos = tf.concat(input_poses, -1)

    return out_anc, out_pos


def contracting_block(input_anc, input_pos, channels, dimensions, dropout_flag, dropout_rate, layer_idx,
                      max_pool_flag, max_pool_filter_stride, stride_input=1, k_size=3, padding_type='SAME',
                      reuse=False):

    conv_layer_idx = layer_idx * 2
    input_channels, output_channels = channels

    with tf.name_scope('contracting_{}_{}'.format(dimensions, layer_idx)) as scope:
        print('contracting_{}_{}: {}'.format(dimensions, layer_idx, padding_type))
        conv_output_anc, conv_output_pos = conv_block(input_anc, input_pos, channels, dimensions, 0,
                                                      dropout_rate, conv_layer_idx, stride_input=stride_input,
                                                      reuse=reuse)

        conv_output_anc, conv_output_pos = conv_block(conv_output_anc, conv_output_pos,
                                                      [output_channels, output_channels], dimensions, dropout_flag,
                                                      dropout_rate, conv_layer_idx + 1, stride_input=stride_input,
                                                      reuse=reuse)

        if max_pool_flag == 1 and max_pool_filter_stride is not None:
            assert len(max_pool_filter_stride) == 2
            max_pool_filter, max_pool_stride = max_pool_filter_stride

            conv_output_anc, conv_output_pos = max_pool(conv_output_anc, conv_output_pos, max_pool_filter,
                                                        max_pool_stride)

    return conv_output_anc, conv_output_pos



# def expanding_block(input_anc, input_pos, contracting_block, channels, dropout_flag, dropout_rate, layer_idx,
#                     stride_input=1, k_size=3, padding_type='SAME', reuse=False):
#     cont_anc, cont_pos = contracting_block
#
#     conv_layer_idx = layer_idx * 2
#     output_channels, input_channels = channels
#
#     with tf.name_scope('expanding_{}'.format(layer_idx)) as scope:
#         conv_output_anc, conv_output_pos = conv_transpose_block(input_anc, input_pos, channels, dropout_flag,
#                                                                 dropout_rate, conv_layer_idx, stride_input=stride_input,
#                                                                 reuse=reuse)
#
#     return conv_output_anc, conv_output_pos


def conv_transpose_block(input_anc, input_pos, channels, upscale_factor, dropout_flag, dropout_rate, layer_idx,
                         stride_input=1, k_size=5, padding_type='VALID', reuse=False):

    i_size = input_anc.get_shape().as_list()[-2]/stride_input
    assert padding_type in {'SAME', 'VALID'}
    # if padding_type == 'SAME':
    #     up_size = i_size * upscale_factor
    # else:
    #     up_size = ((i_size - 1) * upscale_factor) + 1
    up_size = i_size * upscale_factor

    weights = ops.weight([k_size, k_size, k_size, channels[0], channels[1]],
                         layer_name='wcnn_transpose' + str(layer_idx+1), reuse=reuse)

    bias = ops.bias([up_size, up_size, up_size, channels[0]], layer_name='bcnn_transpose' + str(layer_idx+1),
                    reuse=reuse)

    conv_output_anc = tf.add(ops.conv3d_transpose(input_anc, weights, stride=[stride_input, stride_input, stride_input],
                                                  upscale_factor=upscale_factor, padding=padding_type), bias)
    conv_output_pos = tf.add(ops.conv3d_transpose(input_pos, weights, stride=[stride_input, stride_input, stride_input],
                                                  upscale_factor=upscale_factor, padding=padding_type), bias)

    conv_output_anc = ops.batch_norm(conv_output_anc)
    conv_output_pos = ops.batch_norm(conv_output_pos)

    conv_output_anc = ops.relu(conv_output_anc)
    conv_output_pos = ops.relu(conv_output_pos)

    if dropout_flag:
        conv_output_anc = ops.dropout(conv_output_anc, dropout_rate=dropout_rate)
        conv_output_pos = ops.dropout(conv_output_pos, dropout_rate=dropout_rate)

    return conv_output_anc, conv_output_pos


def conv_block(input_anc, input_pos, channels, dimensions, dropout_flag, dropout_rate, layer_idx, stride_input=1,
               k_size=3, padding_type='SAME', reuse=False):

    # Traditional 3D conv layer followed by relu activation
    i_size = input_anc.get_shape().as_list()[-2]/stride_input

    weights = ops.weight([k_size, k_size, k_size, channels[0], channels[1]],
                         layer_name='wcnn_{}_{}'.format(dimensions, layer_idx+1), reuse=reuse)

    bias = ops.bias([i_size, i_size, i_size, channels[1]], layer_name='bcnn_{}_{}'.format(dimensions, layer_idx+1),
                    reuse=reuse)

    conv_output_anc = tf.add(ops.conv3d(input_anc, weights, stride=[stride_input,stride_input, stride_input], padding=padding_type),bias)
    conv_output_pos = tf.add(ops.conv3d(input_pos, weights, stride=[stride_input, stride_input, stride_input], padding=padding_type),bias)

    conv_output_anc = ops.batch_norm(conv_output_anc)
    conv_output_pos = ops.batch_norm(conv_output_pos)

    conv_output_anc = ops.relu(conv_output_anc)
    conv_output_pos = ops.relu(conv_output_pos)

    if dropout_flag:
        conv_output_anc = ops.dropout(conv_output_anc, dropout_rate=dropout_rate)
        conv_output_pos = ops.dropout(conv_output_pos, dropout_rate=dropout_rate)

    return conv_output_anc, conv_output_pos


def encode_block(input_anc, input_pos, output_channels, layer_index, reuse=False):
    conv_layer_idx = layer_index * 2
    input_channels = input_anc.get_shape().as_list()[-1]

    with tf.name_scope('conv_{}'.format(layer_index)) as scope:
        input_anc, input_pos = out_block(input_anc, input_pos, [input_channels, output_channels],
                                         conv_layer_idx, reuse=reuse)

    return input_anc, input_pos


def out_block(input_anc, input_pos, channels, layer_idx, stride_input=1, k_size=8, padding_type='VALID', reuse=False):
    print('out_block_{}: {}'.format(layer_idx, padding_type))

    # Last conv layer, flatten the output
    weights = ops.weight([k_size, k_size, k_size, channels[0], channels[1]],
                         layer_name='wcnn' + str(layer_idx+1), reuse=reuse)

    bias = ops.bias([1, 1, 1, channels[1]], layer_name='bcnn' + str(layer_idx + 1), reuse=reuse)
    conv_asc = ops.conv3d(input_anc, weights, stride=[stride_input, stride_input, stride_input], padding=padding_type)
    conv_output_anc = tf.add(conv_asc, bias)
    conv_output_pos = tf.add(ops.conv3d(input_pos, weights, stride=[stride_input, stride_input, stride_input], padding=padding_type), bias)

    conv_output_anc = ops.batch_norm(conv_output_anc)
    conv_output_pos = ops.batch_norm(conv_output_pos)

    conv_output_anc = tf.contrib.layers.flatten(conv_output_anc)
    conv_output_pos = tf.contrib.layers.flatten(conv_output_pos)

    return conv_output_anc, conv_output_pos


def max_pool(input_anc, input_pos, k_size=2, stride_input=2, padding_type='SAME'):

    conv_output_anc = ops.max_pool3d(input_anc, k_size, stride_input, padding_type)
    conv_output_pos = ops.max_pool3d(input_pos, k_size, stride_input, padding_type)

    return conv_output_anc, conv_output_pos


