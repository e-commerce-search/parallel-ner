# -*- coding: utf-8 -*-
import random, string

import tensorflow as tf


def batch_size(mode, config):
    return {tf.estimator.ModeKeys.TRAIN: config.batch_size,
        tf.estimator.ModeKeys.EVAL: config.eval_batch_size,
        tf.estimator.ModeKeys.PREDICT: config.predict_batch_size}[mode]


def randomString(stringLength=10):
    """Generate a random string of fixed length """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))


def SparseTensor(indices, values, dense_shape):
    indices, values, dense_shape = map(
        tf.convert_to_tensor, [indices, values, dense_shape])
    def maybe_to_int64(t):
        if t.dtype != tf.int64:
            return tf.to_int64(t)
        return t
    indices, dense_shape = map(maybe_to_int64, [indices, dense_shape])
    return tf.SparseTensor(indices, values, dense_shape)


def sparse_tensor_to_dense(sp_tensor, default_value=None):
    # tensor_scatter_nd_update not supported; also scatter_update doesn't work
    # because it's applied to a variable, not a tensor
    if default_value in [0, None, False]:
        return tf.scatter_nd(indices=sp_tensor.indices,
        updates=sp_tensor.values, shape=sp_tensor.dense_shape)
    default_value = tf.cast(default_value, dtype=sp_tensor.dtype)
    mask = tf.scatter_nd(indices=sp_tensor.indices, updates=tf.fill(tf.shape(
        sp_tensor.values), True), shape=sp_tensor.dense_shape)
    other_indices = tf.where(tf.logical_not(mask))
    other_values = tf.fill(tf.shape(other_indices)[:1], default_value)
    indices = tf.concat([sp_tensor.indices, other_indices], axis=0)
    updates = tf.concat([sp_tensor.values, other_values], axis=0)
    return tf.scatter_nd(indices=indices, updates=updates, shape=sp_tensor.dense_shape)


def default_default(dtype):
    return {tf.int32: 0, tf.string: '', tf.float32: 0, tf.int64: 0}[dtype]


def count_nonzero(t):
    return tf.to_float(tf.reduce_sum(tf.to_float(t)))


to_vec = lambda t: tf.reshape(t, [-1])
to_row = lambda t: tf.reshape(t, [1, -1])
to_col = lambda t: tf.reshape(t, [-1, 1])


def reduce_max_with_zero(tensor, axis=None, keepdims=None):
    if axis is None:
        return tf.reduce_max(tf.concat([to_vec(tensor), [0]], axis=0))
    zeros = tf.reduce_sum(0 * tensor, axis=axis, keepdims=True)
    return tf.reduce_max(
        tf.concat([tensor, zeros], axis=axis), axis=axis, keepdims=keepdims)


def missing_elements(indices, length):
    # get the complement of indices in range(length).
    mask = tf.logical_not(tf.scatter_nd(indices=to_col(indices), updates=tf.fill(
        [tf.size(indices)], True), shape=[length]))
    return tf.boolean_mask(tf.range(length), mask)


def sequence_length_from_segment_ids(segment_ids, batch_size=None):
    ones = tf.ones_like(segment_ids)
    if batch_size is None:
        return tf.segment_sum(ones, segment_ids)
    return segment_sum_with_batch_size(ones, segment_ids, batch_size)


def sequence_length_from_sparse_tensor(sp_tensor, num_elements=1):
    if cuda_ops_only(): # segment_max not available on gpu.
        row_indices = sp_tensor.indices[:, 0]
        seq_length = sequence_length_from_segment_ids(row_indices)
        # tf.shape(sp_tensor) outputs int32, better than dense_shape.
        n_pad = tf.shape(sp_tensor)[:1] - tf.shape(seq_length)[:1]
        padding = tf.zeros(n_pad, dtype=seq_length.dtype)
        seq_length = tf.concat([seq_length, padding], axis=0)
        if num_elements != 1:
            seq_length = tf.ceil(seq_length / num_elements)
        return seq_length
    else:
        if tf.__version__.startswith('1.12'):
            from tensorflow.python.feature_column.feature_column import \
_sequence_length_from_sparse_tensor as f
        else:
            assert tf.__version__.startswith('1.15')
            from tensorflow.python.feature_column.utils import \
sequence_length_from_sparse_tensor as f
        return f(sp_tensor, num_elements)


# CUDA compatible sparse_fill_empty_rows. Note that contrary to what official doc
# say, tf.sparse_fill_empty_rows actually works for 3d sp_input.
def sparse_fill_empty_rows(sp_input, default_value, name=None):
    # Op implemented in tflite mode, so don't go through complicated logic below.
    # if not cuda_ops_only():
    #     return tf.sparse_fill_empty_rows(sp_input, default_value, name=name)
    d = sparse_tensor_to_dense(sp_input, default_value=default_value)
    ds = sp_input.dense_shape
    seq_len = sequence_length_from_sparse_tensor(sp_input)
    empty_row_indicator = tf.equal(seq_len, 0)
    mask = sparse_tensor_to_dense(tf.SparseTensor(sp_input.indices,
        tf.fill(tf.shape(sp_input.values), True), ds))
    row_idx = tf.where(tf.logical_and(empty_row_indicator,
        tf.greater(tf.size(d), 0)))   # empty input edge case.
    fill_indices = tf.concat([row_idx, tf.zeros([tf.size(row_idx),
        tf.size(ds) - 1], dtype=tf.int64)], axis=1)
    mask = tf.tensor_scatter_nd_update(
        mask, fill_indices, tf.fill(tf.shape(row_idx)[:1], True))
    values, indices = tf.boolean_mask(d, mask), tf.where(mask)
    return tf.SparseTensor(indices, values, ds), empty_row_indicator


def segment_outer_range(segment_lengths, out_idx=tf.int32):
    """Given a list A of lengths, create [i for i, x in enumerate(A) for _ in range(x)]
    For example [2, 3, 1] -> [0, 0, 1, 1, 1, 2]
    """
    segment_lengths = tf.convert_to_tensor(segment_lengths)
    max_length = reduce_max_with_zero(segment_lengths)
    tiled_range = tf.tile(tf.expand_dims(tf.range(tf.size(
        segment_lengths, out_type=out_idx)), 1), [1, max_length])
    return tf.boolean_mask(
        tiled_range, tf.sequence_mask(segment_lengths, max_length))


def segment_inner_range(segment_lengths, out_idx=tf.int32):
    """Given a list A of lengths, create [i for x in A for i in range(x)].
    For example [2, 3, 1] -> [0, 1, 0, 1, 2, 0]
    """
    segment_lengths = tf.convert_to_tensor(segment_lengths)
    if segment_lengths.dtype != out_idx:
        segment_lengths = tf.cast(segment_lengths, out_idx)
    max_length = reduce_max_with_zero(segment_lengths)
    tiled_range = tf.tile(tf.expand_dims(tf.range(max_length), 0),
        [tf.size(segment_lengths), 1])
    return tf.boolean_mask(
        tiled_range, tf.sequence_mask(segment_lengths, max_length))


def segment_lengths_to_sparse_tensor(values, segment_lengths, name_prefix=None):
    values, segment_lengths = map(tf.convert_to_tensor, [values, segment_lengths])
    max_len = tf.to_int64(reduce_max_with_zero(segment_lengths))
    batch_size = tf.to_int64(tf.size(segment_lengths))
    shape = tf.stack([batch_size, max_len], axis=0)
    indices = tf.to_int64(tf.where(tf.sequence_mask(segment_lengths, max_len)))
    if name_prefix is not None:
        values = tf.identity(values, name=name_prefix + '_values')
        indices = tf.identity(indices, name=name_prefix + '_indices')
        shape = tf.identity(shape, name=name_prefix + '_dense_shape')
    return tf.SparseTensor(values=values, indices=indices, dense_shape=shape)


# This can specialize to segment square.
# values, segment_lengths can be extracted from a SparseTensor.
def segment_tile(values, segment_lengths, segment_multiples, inner_tile=True):
    """For example
    outer_tile: [0, 1, 2, 3, 4], [2, 3], [2, 3] ->
    [0,1,0,1,2,3,4,2,3,4,2,3,4].
    inner_tile: [0, 1, 2, 3, 4], [2, 3], [2, 3] ->
    [0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
    """
    values, segment_lengths, segment_multiples = map(tf.convert_to_tensor,
        [values, segment_lengths, segment_multiples])
    sp_tensor = segment_lengths_to_sparse_tensor(values, segment_lengths)
    dense = sparse_tensor_to_dense(sp_tensor)
    mask_values = tf.ones_like(values, dtype=tf.bool)
    dense_mask = sparse_tensor_to_dense(segment_lengths_to_sparse_tensor(
        mask_values, segment_lengths))  # to undo sparse_tensor_to_dense
    max_multiple = reduce_max_with_zero(segment_multiples)
    mult_mask = tf.sequence_mask(segment_multiples)
    max_len = reduce_max_with_zero(segment_lengths)
    batch_size = tf.shape(segment_lengths)[0]
    if inner_tile:
        tile_fn = lambda x: tf.tile(tf.expand_dims(x, axis=2), [1, 1, max_multiple])
        # to undo max_multiple
        tile_mask = tf.tile(tf.expand_dims(mult_mask, axis=1), [1, max_len, 1])
    else:
        tile_fn = lambda x: tf.tile(x, [1, max_multiple])
        tile_mask = tf.reshape(tf.tile(tf.expand_dims(
            mult_mask, axis=2), [1, 1, max_len]), [batch_size, max_len * max_multiple])
    dense_tiled, mask_tiled = tile_fn(dense), tile_fn(dense_mask)
    final_mask = tf.logical_and(mask_tiled,  tile_mask)
    return tf.boolean_mask(dense_tiled, final_mask)


def segment_shuffle(segment_ids):
    # returns a permutation of {0, .., len(segment_ids) - 1}
    segment_ids = tf.convert_to_tensor(segment_ids)
    if segment_ids.dtype != tf.int32:
        segment_ids = tf.to_int32(segment_ids)
    cnt = tf.segment_sum(tf.ones_like(segment_ids), segment_ids)
    all_range = tf.range(tf.reduce_sum(cnt))
    u = tf.random.shuffle(all_range)
    inner_tile = segment_tile(u, cnt, cnt, inner_tile=True)
    outer_tile = segment_tile(u, cnt, cnt, inner_tile=False)
    tiled_segment_ids = segment_tile(all_range, cnt, cnt, inner_tile=True)
    shuffled_ranks = tf.segment_sum(tf.to_int32(
        tf.greater(inner_tile, outer_tile)), tiled_segment_ids)
    starts = tf_repeat(tf.cumsum(cnt, exclusive=True), cnt)
    return starts + shuffled_ranks


# Checks if tensor a depends on tensor b
def tensor_depends(a, b, excluded_middle=None):
    if a.graph is not b.graph:
        assert False, (a.graph, b.graph)
    gd = a.graph.as_graph_def()
    gd_sub = tf.graph_util.extract_sub_graph(gd, [a.op.name])
    if excluded_middle is None:
        return b.op.name in {n.name for n in gd_sub.node}
    if b.graph is not excluded_middle.graph:
        assert False, (b.graph, excluded_middle.graph)
    gd_sub2 = tf.graph_util.extract_sub_graph(gd, [excluded_middle.op.name])
    nodes = {n.name for n in gd_sub.node if n not in gd_sub2.node}
    assert b.op.name in nodes, (b.op.name, nodes)
    return True


def tf_repeat(base, repeats):
    if tf.__version__.startswith('1.15'):
        return tf.repeat(base, repeats)
    base = tf.reshape(base, [-1, 1])
    max_repeat = reduce_max_with_zero(repeats)
    tiled = tf.tile(base, [1, max_repeat])
    mask = tf.sequence_mask(repeats)
    return tf.boolean_mask(tiled, mask)


def unique_2d(tensor):
    from tensorflow.python.ops import gen_array_ops
    ids, idx = gen_array_ops.unique_v2(tensor, axis=[0])
    # static shape of unique_v2 ids (0th) and idx(1th) are swapped by mistake.
    ids = tf.reshape(ids, [-1, tf.shape(tensor)[1]])
    idx = tf.reshape(idx, [-1])
    return ids, idx


def bash_uniq_with_counts(vectors, out_idx=tf.int32):
    assert isinstance(vectors, list)
    assert len(vectors) > 0
    # taking care of empty vectors[i]'s.
    prefix = tf.tile([1], [tf.to_int32(tf.greater(tf.size(vectors[0]), 0))])
    segment_starts = tf.concat([prefix, tf.to_int32(tf.reduce_any(tf.concat(
        [to_row(tf.not_equal(vec[:-1], vec[1:])) for vec in vectors], axis=0),
        axis=0))], axis=0)
    # exclusive so that new_vector does not always start with [0, 1, ..].
    new_vector = tf.cumsum(segment_starts, exclusive=False)
    return tf.unique_with_counts(new_vector, out_idx=out_idx)

def segment_count_unique(inner_ids, segment_ids):
    inner_starts = tf.concat(
        [[True], tf.not_equal(inner_ids[:-1], inner_ids[1:])], axis=0)
    unique_counts = tf.segment_sum(tf.to_int64(inner_starts), segment_ids)
    factored_segment_ids = tf.boolean_mask(segment_ids, inner_starts)
    return factored_segment_ids, unique_counts


def segment_sum_with_batch_size(values, segment_ids, batch_size):
    values, segment_ids = map(tf.convert_to_tensor, [values, segment_ids])
    chip_shape = tf.concat([[1], tf.shape(values)[1:]], axis=0)
    dummy_row = tf.zeros(chip_shape, dtype=values.dtype)
    augmented_ids = tf.concat([segment_ids, [batch_size]], axis=0)
    augmented_values = tf.concat([values, dummy_row], axis=0)
    return tf.segment_sum(augmented_values, augmented_ids)[:-1]


def expand_rank(t, rank):   # append 1's to tf.shape(t) to reach rank.
    target_shape = tf.concat([tf.shape(t), [1] * rank - tf.rank(t)], axis=0)
    return tf.reshape(t, target_shape)


def segment_mean_with_batch_size(values, segment_ids, batch_size):
    seq_len = sequence_length_from_segment_ids(segment_ids, batch_size)
    return segment_sum_with_batch_size(values, segment_ids, batch_size) / (
        expand_rank(tf.maximum(tf.to_float(seq_len), 1.0), tf.rank(values)))


def segment_sqrt_n_with_batch_size(values, segment_ids, batch_size):
    seq_len = sequence_length_from_segment_ids(segment_ids, batch_size)
    return segment_sum_with_batch_size(values, segment_ids, batch_size) / (
        expand_rank(tf.sqrt(tf.maximum(tf.to_float(seq_len), 1.0))),
        tf.rank(values))


def fill_empty_rows_2d(
    embedding_2d, segment_ids, batch_size, default_value=0.0):
    orig_size = tf.to_int64(tf.size(segment_ids))
    segment_lengths = segment_sum_with_batch_size(
        tf.ones_like(segment_ids), segment_ids, batch_size)
    sp_input = segment_lengths_to_sparse_tensor(
        tf.range(orig_size), segment_lengths)
    sp_output = sparse_fill_empty_rows(sp_input, orig_size)[0]
    aug_embedding = tf.concat([embedding_2d, default_value * tf.ones(
        [1, tf.shape(embedding_2d)[1]])], axis=0)
    return tf.gather(aug_embedding, sp_output.values), sp_output.indices[:, 0]


def masked_segment_lengths(segment_lengths, mask):
    segment_ids = segment_outer_range(segment_lengths)
    # be careful that mask size may be different from segment_ids size.
    masked_ids = tf.boolean_mask(segment_ids, mask)
    return segment_sum_with_batch_size(
        tf.ones_like(masked_ids), masked_ids, tf.size(segment_lengths))


def indexed_slices_to_dense(indexed_slices):
    if tf.__version__.startswith('1.15'):
        return tf.convert_to_tensor(indexed_slices)
    else:
        from tensorflow.python.ops import gradients_impl
        return gradients_impl._IndexedSlicesToTensor(indexed_slices)

