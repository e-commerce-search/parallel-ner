import os, sys
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import tensorflow as tf
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from bert.run_classifier import *
from bert import tokenization
import multi_ner
import tf_utils
import multiprocessing


class NamedEntityRecognitionProcessor(DataProcessor):
    """Base Processor for NER data set, with sentence/entity/label."""
    def __init__(self, tokenizer):
        self._tokenizer = tokenizer

    def name(self):
        return ''
        raise NotImplementedError()

    def get_examples(self, data_dir, split_type='train'):
        """See base class."""
        base_dir = os.path.join(data_dir, self.name())
        num_examples_file = os.path.join(base_dir, 'num_%s_examples.txt' % split_type)
        if os.path.exists(num_examples_file):   # tokenization and tfrecord already done!
            with open(num_examples_file) as f:
                return int(f.readline().strip())

        prefix = os.path.join(base_dir, "{}.{}.tsv".format(self.name(), split_type))
        def create_bytes_feature(values):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=list(values)))

        def create_float_feature(values):
            return tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))

        def create_tf_record_file(input_file, output_file):
            with tf.python_io.TFRecordWriter(output_file) as w, open(input_file) as f:
                ex_index = 0
                for ex_index, line in enumerate(f):
                    if ex_index % 10000 == 0:
                        tf.logging.info("Writing % example %d of %s" % (
                            split_type, ex_index, input_file))
                    tmp = line.strip('\r\n').split('\t')
                    text_a, text_b, label = tmp[0], tmp[1], tmp[2:]
                    features = collections.OrderedDict()
                    features['sentence'] = create_bytes_feature(
                        self._tokenizer.tokenize(text_a))
                    features['entity'] = create_bytes_feature(
                        self._tokenizer.tokenize(text_b))
                    features['label'] = create_float_feature(
                        list(map(float, label)))
                    tf_example = tf.train.Example(features=tf.train.Features(
                        feature=features))
                    w.write(tf_example.SerializeToString())
                return ex_index

        pool, results = multiprocessing.Pool(), []
        for input_file in tf_utils.glob_files(prefix + '*'):
            suffix = input_file[len(prefix):]
            output_file = os.path.join(base_dir, split_type + '.tf_record' + suffix)
            results.append(pool.apply_async(create_tf_record_file, (
                input_file, output_file)))
        num_examples = sum(result.get() for result in results)
        setattr(self, '_num_%s_examples' % split_type, num_examples)
        [pool.close(), pool.join()]
        with open(num_examples_file, 'w') as f:
            f.write(str(num_examples))
        return num_examples

    def get_train_examples(self, data_dir):
        return self.get_examples(data_dir, 'train')

    def get_dev_examples(self, data_dir):
        return self.get_examples(data_dir, 'dev')

    def get_test_examples(self, data_dir):
        return self.get_examples(data_dir, 'test')


class PwordProcessor(NamedEntityRecognitionProcessor):
    """Processor for the product word data set."""
    def name(self):
        return 'pword'


class MsraProcessor(NamedEntityRecognitionProcessor):
    """Processor for the product word data set."""
    def name(self):
        return 'msra'


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    processors = {
        "pword": PwordProcessor,
        "msra": MsraProcessor,
    }
    task_name = FLAGS.task_name.lower()
    processor = processors[task_name]()

    tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file,
                                           do_lower_case=FLAGS.do_lower_case)

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    train_examples = None
    num_train_steps = None
    num_train_examples = None
    if FLAGS.do_train:
        num_train_examples = processor.get_train_examples(FLAGS.data_dir)
        num_train_steps = int(
            num_train_examples / FLAGS.train_batch_size * FLAGS.num_train_epochs)


    model_fn = multi_ner.model_fn
    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=False,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)

    if FLAGS.do_train:
        # tfrecord stores text rather than token ids.
        train_file = tf_utils.global_fiiles(os.path.join(FLAGS.output_dir, "train.tf_record*"))
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", num_train_examples)
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    if FLAGS.do_eval:
        num_eval_examples = processor.get_dev_examples(FLAGS.data_dir)

        eval_file = tf_utils.glob_files(os.path.join(FLAGS.output_dir, "eval.tf_record*"))

        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

        # This tells the estimator to run through the entire set.
        eval_steps = None

        eval_drop_remainder =  False
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=eval_drop_remainder)

        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    if FLAGS.do_predict:
        num_predict_examples = processor.get_test_examples(FLAGS.data_dir)

        predict_file = tf_utils.glob_files(os.path.join(FLAGS.output_dir, "predict.tf_record*"))

        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

        predict_drop_remainder = True if FLAGS.use_tpu else False
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=predict_drop_remainder)

        result = estimator.predict(input_fn=predict_input_fn)

        output_predict_file = os.path.join(FLAGS.output_dir,
                                            "test_results.tsv")
        with tf.gfile.GFile(output_predict_file, "w") as writer:
            num_written_lines = 0
            tf.logging.info("***** Predict results *****")
            for (i, prediction) in enumerate(result):
                probabilities = prediction["probabilities"]
                if i >= num_predict_examples:
                    break
                output_line = "\t".join(
                    str(class_probability)
                    for class_probability in probabilities) + "\n"
                writer.write(output_line)
                num_written_lines += 1
        assert num_written_lines == num_predict_examples
