"""BERT model_fn for CORNERSTONE consumption."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os, sys
import tensorflow as tf


sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import tf_utils


cache = {}

def special_token_ids(vocab_file_path):
    if cache.get('cls_id'):
        return cache['cls_id'], cache['sep_id']
    cls_token, sep_token, cls_id, sep_id = '[CLS]', '[SEP]', None, None
    with open(vocab_file_path) as f:
        for i, line in enumerate(f.readlines()):
            if line.strip() == cls_token:
                cls_id = i
            elif line.strip() == sep_token:
                sep_id = i
            if cls_id is not None and sep_id is not None:
                break
    if cls_id != 101 or sep_id != 102:
        tf.logging.warning('cls_id: %d, sep_id: %d!' % (cls_id, sep_id))
    cache['cls_id'], cache['sep_id'] = cls_id, sep_id
    return cls_id, sep_id


def token_mask(input_ids, input_mask, vocab_file_path):
    cls_id, sep_id = special_token_ids(vocab_file_path)
    special_mask = tf.logical_and(
        tf.not_equal(input_ids, cls_id), tf.not_equal(input_ids, sep_id))
    return tf.logical_and(tf.cast(input_mask, tf.bool), special_mask)


def bert_sparse_to_dense(
    text_a, text_b, seq_length=None, vocab_file_path=None, discard_text_b=False):
    assert seq_length and vocab_file_path
    assert all(isinstance(t, tf.SparseTensor) for t in [text_a, text_b])
    with tf.device('/device:CPU:0'):
        cnt_a, cnt_b = [tf.to_int64(tf_utils.sequence_length_from_sparse_tensor(t))
            for t in [text_a, text_b]]
    batch_size = tf.shape(cnt_a, out_type=tf.int64)[0]
    width_a, width_b = map(tf_utils.reduce_max_with_zero, [cnt_a, cnt_b])
    def change_width(sp_tensor, width):
        return tf.SparseTensor(sp_tensor.indices, sp_tensor.values, tf.stack(
            [sp_tensor.dense_shape[0], tf.to_int64(width)], axis=0))
    text_a = change_width(text_a, width_a)
    text_b = change_width(text_b, width_b)
    text_a, text_b = map(tf_utils.sparse_tensor_to_dense, [text_a, text_b])
    one = tf.ones_like(tf.reshape(cnt_a, [-1, 1]), dtype=tf.int64)
    true = tf.ones_like(tf.reshape(cnt_a, [-1, 1]), dtype=tf.bool)
    cls_id, sep_id = special_token_ids(vocab_file_path)
    cls, sep = cls_id * one, sep_id * one
    if discard_text_b is True:  # if len(tokens_a) > max_seq_length - 2:
        # tokens_a = tokens_a[0:(max_seq_length - 2)]
        max_len = (seq_length - 2) * tf.reshape(one, [-1])
        length_a = tf.minimum(cnt_a, max_len)
        mask_a = tf.sequence_mask(length_a, width_a)
        mask = tf.concat([true, mask_a, true], axis=1)
        flat_mask = tf.reshape(mask, [-1])
        combined_width = width_a + 2
        # combined_width = tf_utils.easy_debug(combined_width)
        combined_length = length_a + 2
        segment_ids = tf.reshape(tf.concat(
            [0 * one, 0 * text_a, 0 * one], axis=1), [-1])
        values = tf.reshape(tf.concat([cls, text_a, sep], axis=1), [-1])
    else:
        max_len = (seq_length - 3) * tf.reshape(one, [-1])
        if discard_text_b == 0.5:   # maximally keep text_a.
            length_a = tf.minimum(cnt_a, max_len)
            length_b = tf.minimum(max_len - length_a, cnt_b)
        else:   # Follow the logic of _truncate_seq_pair
            min_cnt = tf.minimum(cnt_a, cnt_b)
            length_a = tf.where(tf.less_equal(cnt_a + cnt_b, max_len), cnt_a,
                tf.where(tf.less_equal(min_cnt * 2, max_len), tf.where(tf.less(
                cnt_a, cnt_b), cnt_a, max_len - cnt_b), (max_len + 1) // 2))
            length_b = tf.where(tf.less_equal(cnt_a + cnt_b, max_len), cnt_b,
                tf.where(tf.less_equal(min_cnt * 2, max_len), tf.where(tf.less(
                cnt_b, cnt_a), cnt_b, max_len - cnt_a), max_len // 2))
        mask_a = tf.sequence_mask(length_a, width_a)
        mask_b = tf.sequence_mask(length_b, width_b)
        mask = tf.concat([true, mask_a, true, mask_b, true], axis=1)
        flat_mask = tf.reshape(mask, [-1])
        combined_width = width_a + width_b + 3
        combined_length = length_a + length_b + 3
        segment_ids = tf.reshape(tf.concat(
            [0 * one, 0 * text_a, 0 * one, 0 * text_b + 1, one], axis=1), [-1])
        values = tf.reshape(tf.concat(
            [cls, text_a, sep, text_b, sep], axis=1), [-1])

    with tf.device('/device:CPU:0'):
        pos_ids = tf.reshape(tf.tile(tf.reshape(tf.range(combined_width,
            dtype=tf.int64), [1, -1]), [batch_size, 1]), [-1])
        seg_ids = tf.reshape(tf.tile(tf.reshape(tf.range(
            batch_size, dtype=tf.int64), [-1, 1]), [1, combined_width]), [-1])
    pos_mask = tf.reshape(tf.sequence_mask(combined_length, combined_width), [-1])
    pos_ids = tf.boolean_mask(pos_ids, pos_mask)
    seg_ids = tf.boolean_mask(seg_ids, flat_mask)
    indices = tf.stack([seg_ids, pos_ids], axis=1)
    dense_shape = tf.stack([batch_size, seq_length], axis=0)
    def truncate(v):
        untruncated_sp = tf.SparseTensor(indices=indices,
            values=tf.boolean_mask(v, flat_mask), dense_shape=dense_shape)
        return tf_utils.sparse_tensor_to_dense(untruncated_sp)[:, :seq_length]

    input_ids = truncate(values)

    input_mask = tf.sequence_mask(combined_length, tf.maximum(
        tf.to_int64(seq_length), combined_width), dtype=tf.int64)[:, :seq_length]
    segment_ids = truncate(segment_ids)
    return input_ids, input_mask, segment_ids