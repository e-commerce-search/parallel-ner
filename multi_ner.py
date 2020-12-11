import os, sys
import functools
import tensorflow as tf
from tensorflow.estimator import ModeKeys
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import custom_bert

from bert import modeling
from bert import optimization

from models import metric_utils
from tf_utils import to_vec, to_col, tf_repeat, unique_2d, \
bash_uniq_with_counts, sparse_tensor_to_dense, \
sequence_length_from_sparse_tensor, segment_inner_range, sparse_fill_empty_rows


def sparse_sequence_match(haystack, needle):
    """Outputs pos matches of needle in haystack by row, as indices and shape."""
    batch_size = tf.shape(haystack)[0]
    # NOTE(jyj): otherwise empty needle will match every position in hay
    needle = sparse_fill_empty_rows(needle, -1)[0]  # batch x needle
    needle_d = sparse_tensor_to_dense(needle)
    max_width_h = tf.shape(haystack)[-1]
    max_width_n = tf.shape(needle)[-1]
    # pad haystack by a block of max needle width.
    # batch x (hay + needle)
    haystack_d = tf.concat([sparse_tensor_to_dense(haystack),
        -tf.ones_like(needle_d)], axis=1)
    # batch x hay x needle
    needle_widths = sequence_length_from_sparse_tensor(needle)
    needle_mask = tf.tile(tf.expand_dims(
        tf.sequence_mask(needle_widths), axis=1), [1, max_width_h, 1])
    haystack_mask = tf.sequence_mask(
        sequence_length_from_sparse_tensor(haystack) - needle_widths + 1,
        maxlen=tf.reduce_max(sequence_length_from_sparse_tensor(haystack)))
    # batch x hay x (hay + needle)
    h = tf.tile(tf.expand_dims(haystack_d, axis=1), [1, max_width_h, 1])
    # hay x (hay + needle)
    ones_mat = tf.ones([max_width_h, max_width_h + max_width_n], dtype=tf.bool)
    # batch x hay x (needle + hay)
    shift_mask = tf.tile(tf.expand_dims(tf.matrix_band_part(
        ones_mat, 0, tf.to_int64(max_width_n - 1)), axis=0), [batch_size, 1, 1])
    # batch x hay x needle
    shifted_tile = tf.reshape(tf.boolean_mask(h, shift_mask),
                              [batch_size, max_width_h, max_width_n])
    # batch x hay x needle
    needle_b = tf.where(needle_mask,
        x=tf.equal(tf.expand_dims(needle_d, axis=1), shifted_tile),
        y=tf.cast(tf.ones_like(shifted_tile), tf.bool))
    # reduce any along needle dim: batch x hay.
    found = tf.logical_and(tf.reduce_all(needle_b, axis=2), haystack_mask)
    indices = tf.where(found)
    batch_idx, start_pos = tf.split(indices, num_or_size_splits=2, axis=1)
    batch_widths = bash_uniq_with_counts([batch_idx], out_idx=tf.int64)[2]
    needle_idx = segment_inner_range(batch_widths, out_idx=tf.int64)
    return batch_idx, needle_idx, start_pos, tf.reshape(needle_widths, [-1, 1])


def flatten_needle(batch_idx, needle_idx, start_pos, needle_widths):
    # suffix 2 stands for segment flattened by needle widths.
    batch_widths = bash_uniq_with_counts([batch_idx])[2]
    needle_widths2 = tf_repeat(needle_widths, batch_widths)
    batch_idx2 = tf_repeat(batch_idx, needle_widths2)
    needle_idx2 = tf_repeat(needle_idx, needle_widths2)
    needle_pos = tf_repeat(start_pos, needle_widths2
        ) + segment_inner_range(needle_widths2)
    return batch_idx2, needle_idx2, needle_pos


def get_dense_mask(needle_pos, segment_idx, dense_shape):
    # segment_idx is basically row idx.
    from_tensor = tf.ones(dense_shape)
    indices = tf.to_int64(tf.stack([segment_idx, needle_pos], axis=1))
    values = tf.fill(tf.shape(needle_pos), True)
    sp_tensor = tf.SparseTensor(indices, values, tf.to_int64(dense_shape))
    to_mask = sparse_tensor_to_dense(sp_tensor, False)
    return modeling.create_attention_mask_from_input_mask(from_tensor, to_mask)


def aggregate_embedding(embeddings, segment_idx, aggregator, config=None, aux=None, name=None):
    # segment_idx denotes different needles, rather than rows.
    if aggregator == 'segment_sqrt_n':
        denom = to_col(tf.sqrt(tf.to_float(tf.segment_sum(tf.ones_like(segment_idx), segment_idx))))
        output_layer = tf.div_no_nan(tf.segment_sum(
            embeddings, segment_idx), denom, name=name)
    elif aggregator in ['segment_sum', 'segment_mean']:
        output_layer = getattr(tf, aggregator)(
            embeddings, segment_idx, name=name)
    else:
        del embeddings
        assert aggregator.startswith('transformer')
        flags = {}
        if '^' in aggregator:
            flags = [kv.split('@') for kv in filter(None, aggregator.split('^')[1].split(','))]
            flags = {k: eval(v) for k, v in flags}
        assert config is not None and aux is not None
        needle_pos = aux['needle_pos']
        embedding_output = aux['sequence_output']
        batch_idx2 = aux['batch_idx2']  # different rows.
        is_training = aux['is_training']
        attention_mask = get_dense_mask(
            needle_pos, batch_idx2, tf.shape(embedding_output)[:2])
        with tf.variable_scope('final_transformer'):
            all_encoder_layers = modeling.transformer_model(
                input_tensor=embedding_output,
                attention_mask=attention_mask,
                hidden_size=config.hidden_size, # this must agree with input width.
                num_hidden_layers=flags.get('num_hidden_layers', 1),
                num_attention_heads=flags.get('num_attention_heads', config.num_attention_heads),
                intermediate_size=flags.get('intermediate_size', config.intermediate_size),
                intermediate_act_fn=modeling.get_activation(config.hidden_act),
                hidden_dropout_prob=flags.get('hidden_dropout_prob', config.hidden_dropout_prob)
                    * int(is_training),
                attention_probs_dropout_prob=int(is_training) *
                    flags.get('attention_probs_dropout_prob', config.attention_probs_dropout_prob),
                initializer_range=config.initializer_range,
                do_return_all_layers=True)

            first_token_tensor = all_encoder_layers[-1][:, 0, :]
            output_layer = tf.layers.dense(
                first_token_tensor,
                config.hidden_size,
                activation=tf.tanh,
                kernel_initializer=modeling.create_initializer(
                    config.initializer_range))

    return output_layer


def fill_missing(batch_idx, needle_idx, start_pos, text_a, num_labels):
    bd, nd, sd = batch_idx.dtype, needle_idx.dtype, start_pos.dtype
    dense_shape = tf.to_int64(
        [text_a.dense_shape[0], num_labels, text_a.dense_shape[1]])
    sp_input = tf.SparseTensor(indices=tf.concat(list(map(tf.to_int64,[
        batch_idx, to_col(needle_idx), start_pos])), axis=1),
        values=tf.zeros_like(batch_idx[:, 0]), dense_shape=dense_shape)
    dummy = -1  # the dummy start_pos will all be 0.
    filled_sp = sparse_fill_empty_rows(sp_input, dummy)[0]
    tmp = tf.split(filled_sp.indices, axis=1, num_or_size_splits=3)
    tf.summary.scalar('missing_needles', tf.size(
        tf.where(tf.equal(filled_sp.values, dummy))))
    tf.summary.scalar('found_needles', tf.size(
        tf.where(tf.not_equal(filled_sp.values, dummy))))
    return tf.cast(tmp[0], bd), tf.cast(tmp[1], nd), tf.cast(tmp[2], sd)


def model_fn(features, labels, mode, params):
    is_training = mode == ModeKeys.TRAIN
    config, args = params['config'], params['args']
    model_config = config.model_config
    query_answer_feature_names = (
        model_config.query_answer_feature_names.split(','))
    feature_stats = {k: v for k, v in config.feature_stats.items() if k in
        query_answer_feature_names}
    num_labels, learning_rate, use_tpu, use_one_hot_embeddings = (
        model_config.num_labels, model_config.learning_rate,
        model_config.use_tpu, model_config.use_one_hot_embeddings)
    init_checkpoint = (model_config.init_checkpoint and
        model_config.init_checkpoint.split(':')[-1])

    steps_per_epoch = model_config.num_training_examples / config.batch_size
    num_train_steps = int(steps_per_epoch * config.epochs)
    num_warmup_steps = int(steps_per_epoch * model_config.warmup_proportion)

    tmp = [features[feature_stats[k]['feature_id']] for k
            in query_answer_feature_names]
    assert model_config.num_queries == 2
    text_a = features['sentence']
    text_b = features['entity']
    label = features['label']

    with tf.device('/device:CPU:0'):
        vocab_file_paths = [v['vocab_file'].split(':')[-1] for v in
          feature_stats.values()]
        to_dense = functools.partial(custom_bert.bert_sparse_to_dense, text_a,
            seq_length=model_config.seq_length,
            vocab_file_path=vocab_file_paths[0],
            discard_text_b=True)
        input_ids, input_mask, segment_ids = map(tf.to_int32, to_dense(text_a))
        concat_int = tf.concat([input_ids, input_mask, segment_ids], axis=1)
        # unique along the batch dimension.
        unique_int, segment_idx = unique_2d(concat_int)
        input_ids, input_mask, segment_ids = tf.split(
            unique_int, axis=1, num_or_size_splits=3)
    model = modeling.BertModel(
            config=model_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings)

    sequence_output = tf.gather(model.get_sequence_output(), segment_idx)
    orig_width = sequence_output.shape[-1].value
    hidden_size = orig_width
    expt_flags = dict(t.split(':') for t in filter(
        None, config.experimental_flags.split(';')))
    # only use the first column. Later pword columns are treated as features.
    # label > -1 as context feature mask.
    pword_context_aggregator = expt_flags.get('pword_context_aggregator')
    needle_embedding_aggregator = expt_flags.get('needle_embedding_aggregator')
    output_dim = int(expt_flags.get('output_dim', 1))   # 2 means using softmax
    if pword_context_aggregator:
        hidden_size += orig_width
    if eval(expt_flags.get('concat_first_embedding', 'False')):
        hidden_size += orig_width
    output_weights, output_biases = [], []
    with tf.variable_scope("loss"):
        output_layers = list(map(int, filter(None, expt_flags.get(
            'output_layers', '').split(','))))
        output_layers = [hidden_size] + output_layers + [output_dim]
        for i, (a, b) in enumerate(zip(output_layers[:-1], output_layers[1:])):
            suffix = '' if i == 0 else '_%d' % i
            output_weights.append(tf.get_variable('output_weights' + suffix,
                [b, a], initializer=tf.truncated_normal_initializer(stddev=0.02)))
            output_biases.append(tf.get_variable('output_bias' + suffix,
                [b], initializer=tf.zeros_initializer()))
    def mlp(net, weights, biases):
        for i, (w, b) in enumerate(zip(weights, biases)):
            dropout_rate = float(expt_flags.get('mlp_dropout_rate', 0.0))
            if dropout_rate > 0.0 and is_training:
                net = modeling.dropout(net, dropout_rate)
            if eval(expt_flags.get('mlp_layer_norm', 'False')):
                net = modeling.layer_norm(net)
            net = tf.nn.bias_add(tf.matmul(net, w, transpose_b=True), b)
            if i < len(weights) - 1:
                net = modeling.gelu(net)
        return net
    output_layers = []

    # batch x num needles
    batch_idx, needle_idx, start_pos, needle_widths = map(tf.to_int32, (
        sparse_sequence_match(text_a, text_b)))
    # even with preprocessing some sentences may not have any matched entity.
    if eval(expt_flags.get('fill_missing', 'False')):
        batch_idx, needle_idx, start_pos = fill_missing(
            batch_idx, needle_idx, start_pos, text_a, num_labels)

    # batch x num needles x needle width
    batch_idx2, needle_idx2, needle_pos = flatten_needle(
        batch_idx, needle_idx, start_pos, needle_widths)
    # batch x needle idx x haystack pos
    # sequence output: batch x sequence length x embedding dim.
    indices = tf.stack([batch_idx2, needle_pos], axis=1)
    # (batch x num needles x needle width) x embedding_dim
    embeddings = tf.gather_nd(sequence_output, indices)
    needle_idx3 = bash_uniq_with_counts([batch_idx2, needle_idx2])[1]

    # (batch x num_needles) x embedding_dim
    output_aggregator = expt_flags.get('output_aggregator', 'segment_mean')
    output_layer = aggregate_embedding(embeddings, needle_idx3, output_aggregator,
        name='aggregated_entity_embedding',
        aux={'needle_pos': needle_pos, 'sequence_output': sequence_output,
            'batch_idx2': batch_idx2, 'is_training': is_training},
        config=model_config)
    if eval(expt_flags.get('concat_first_embedding', 'False')):
        pooled_layer = tf.gather(model.get_pooled_output(), segment_idx)
        if not output_aggregator.startswith('transformer'):
            pooled_layer = tf.gather(pooled_layer, to_vec(batch_idx))
        output_layer = tf.concat([output_layer, pooled_layer], axis=1)
    if needle_embedding_aggregator:
        assert not output_aggregator.startswith('transformer'), (
            'transformer aggregator already aggregates needles in the same row!')
        output_layer = aggregate_embedding(
            output_layer, to_vec(batch_idx), needle_embedding_aggregator)
    if is_training and float(expt_flags.get('mlp_dropout_rate', 0.0)) <= 0.0:
        # I.e., 0.1 dropout
        output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

    logits2 = mlp(output_layer, output_weights, output_biases)
    if needle_embedding_aggregator or output_aggregator.startswith('transformer'):
        logits = tf.identity(logits2, name='mean_logits')
    else:
        logits = tf.segment_mean(tf.reshape(
            logits2, [-1, output_dim]), to_vec(batch_idx), name='mean_logits')
    loss, per_example_loss = None, None
    if mode != ModeKeys.PREDICT:
        if output_dim == 1:
            per_example_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=to_vec(labels[:, 0]), logits=to_vec(logits))
        else:   # pairwise softmax
            assert output_dim == 2
            dense_log_probs = tf.nn.log_softmax(logits, axis=-1)
            one_hot_labels = tf.one_hot(
                tf.to_int32(labels[:, 0]), depth=output_dim, dtype=tf.float32)
            per_example_loss = -tf.reduce_sum(
                one_hot_labels * dense_log_probs, axis=-1)
            dense_probs = tf.maximum(1e-8, tf.minimum(1.0 - 1e-8,
                tf.exp(dense_log_probs[:, 1])))
            logits = - tf.log(1. / dense_probs - 1.)
        per_query_loss = tf.reduce_sum(per_example_loss, axis=-1)
        loss = tf.reduce_mean(per_query_loss)

    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    if init_checkpoint:
        (assignment_map, initialized_variable_names
         ) = modeling.get_assignment_map_from_checkpoint(
             tvars, init_checkpoint)
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
        init_string = ""
        if var.name in initialized_variable_names:
            init_string = ", *INIT_FROM_CKPT*"
        tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                        init_string)
    output_spec = None
    if mode == ModeKeys.TRAIN:
        train_op = optimization.create_optimizer(loss, learning_rate,
            num_train_steps, num_warmup_steps, use_tpu, config=config)
        output_spec = tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op)
    elif mode == ModeKeys.EVAL:
        def metric_fn(per_example_loss, labels, logits):
            metrics = {}
            def update_metrics(labels, logits, metrics, scope=None):
                probabilities = to_vec(tf.sigmoid(logits))
                predicted_classes = tf.to_int32(probabilities > 0.5)
                tmp = metric_utils.binary_classification_metrics(
                to_vec(labels), {'predicted_classes': predicted_classes,
                'probabilities': probabilities, 'logits': to_vec(logits)}, config)
                for k, v in tmp.items():
                    metrics['%s/%s' % (scope, k) if scope else k] = v

            update_metrics(labels, logits, metrics)
            for i in range(num_labels):
                update_metrics(labels[:, i], logits[:, i], metrics, 'column_%d' % i)

            if args.role == 'evaluator' and args.index > 0:
                scope = 'secondary_eval'    # to be backward compatible.
                scope += '_%d' % args.index if args.index > 1 else ''
                metrics = {'%s/%s' % (scope, k): v for k, v in metrics.items()}

            return metrics

        eval_metric_ops = metric_fn(per_example_loss, labels, logits)
        output_spec = tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            eval_metric_ops=eval_metric_ops)
    else:   # PREDICT
        predictions = {
            "probabilities": tf.sigmoid(logits),
            "logits": logits,
        }
        output_spec = tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions)
    return output_spec