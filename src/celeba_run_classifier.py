"""
Script to reproduce the few-shot classification results in:
"Meta-Learning Probabilistic Inference For Prediction"
https://arxiv.org/pdf/1805.09921.pdf
The following command lines will reproduce the published results within error-bars:
Omniglot 5-way, 5-shot
----------------------
python run_classifier.py
Omniglot 5-way, 1-shot
----------------------
python run_classifier.py --shot 1
Omniglot 20-way, 5-shot
-----------------------
python run_classifier.py --way 20 --iterations 60000
Omniglot 20-way, 1-shot
-----------------------
python run_classifier.py --way 20 --shot 1 --iterations 100000
minImageNet 5-way, 5-shot
-------------------------
python run_classifier.py --dataset miniImageNet --tasks_per_batch 4 --iterations 100000 --dropout 0.5
minImageNet 5-way, 1-shot
-------------------------
python run_classifier.py --dataset miniImageNet --shot 1 --tasks_per_batch 8 --iterations 50000 --dropout 0.5 -lr 0.00025
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import argparse
import celeba_test_utils
from features import extract_features_omniglot, extract_features_mini_imagenet, extract_features_celeba
from inference import infer_classifier
from utilities import sample_normal, multinoulli_log_density, print_and_log, get_log_files
from data import get_data
from itertools import cycle

"""
parse_command_line: command line parser
"""


def parse_command_line():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--dataset", "-d", choices=["Omniglot", "miniImageNet", "celebA"],
    #                     default="Omniglot", help="Dataset to use")
    # parser.add_argument("--mode", choices=["train", "test", "train_test"], default="train_test",
    #                     help="Whether to run traing only, testing only, or both training and testing.")
    # parser.add_argument("--d_theta", type=int, default=256,
    #                     help="Size of the feature extractor output.")
    # parser.add_argument("--shot", type=int, default=5,
    #                     help="Number of training examples.")
    # parser.add_argument("--way", type=int, default=5,
    #                     help="Number of classes.")
    # parser.add_argument("--test_shot", type=int, default=None,
    #                     help="Shot to be used at evaluation time. If not specified 'shot' will be used.")
    # parser.add_argument("--test_way", type=int, default=None,
    #                     help="Way to be used at evaluation time. If not specified 'way' will be used.")
    # parser.add_argument("--tasks_per_batch", type=int, default=16,
    #                     help="Number of tasks per batch.")
    # parser.add_argument("--samples", type=int, default=10,
    #                     help="Number of samples from q.")
    # parser.add_argument("--learning_rate", "-lr", type=float, default=1e-4,
    #                     help="Learning rate.")
    # parser.add_argument("--iterations", type=int, default=80000,
    #                     help="Number of training iterations.")
    # parser.add_argument("--checkpoint_dir", "-c", default='./checkpoint',
    #                     help="Directory to save trained models.")
    # parser.add_argument("--dropout", type=float, default=0.9,
    #                     help="Dropout keep probability.")
    # parser.add_argument("--test_model_path", "-m", default=None,
    #                     help="Model to load and test.")
    # parser.add_argument("--print_freq", type=int, default=200,
    #                     help="Frequency of summary results (in iterations).")
    args = parser.parse_args()

    args.dataset = 'celebA'
    args.mode = "test_celeba"
    args.d_theta = 64  # "Size of the feature extractor output."
    args.shot = 5  # "Number of training examples."
    args.way = 2  # "Number of classes."
    args.test_shot = 5  # Shot to be used at evaluation time. If not specified 'shot' will be used.")
    args.test_way = 2  # "Way to be used at evaluation time. If not specified 'way' will be used.")
    args.tasks_per_batch = 6
    args.samples = 10  # "Number of samples from q.")
    args.learning_rate = 1e-4
    args.iterations = 10000
    args.checkpoint_dir = './models/way2shot5'
    args.dropout = 0.9
    args.test_model_path = './models/way2shot5/theta64_bs6_s10_lr1e4_10k/fully_trained'
    args.print_freq = 2

    # adjust test_shot and test_way if necessary
    if args.test_shot is None:
        args.test_shot = args.shot
    if args.test_way is None:
        args.test_way = args.way

    return args


def main(unused_argv):
    tf.logging.set_verbosity(tf.logging.ERROR)

    args = parse_command_line()

    logfile, checkpoint_path_validation, checkpoint_path_final = get_log_files(args.checkpoint_dir)

    print_and_log(logfile, "Options: %s\n" % args)

    # Load training and eval data
    data = get_data(args.dataset)

    # set the feature extractor based on the dataset
    feature_extractor_fn = extract_features_mini_imagenet
    if args.dataset == "Omniglot":
        feature_extractor_fn = extract_features_omniglot
    if args.dataset == "celebA":
        feature_extractor_fn = extract_features_celeba

    # evaluation samples
    eval_samples_train = 15
    eval_samples_test = args.shot

    # testing parameters
    test_iterations = 600
    test_args_per_batch = 1  # always use a batch size of 1 for testing

    # tf placeholders
    train_images = tf.placeholder(tf.float32, [None,  # tasks per batch
                                               None,  # shot
                                               data.get_image_height(),
                                               data.get_image_width(),
                                               data.get_image_channels()],
                                  name='train_images')
    test_images = tf.placeholder(tf.float32, [None,  # tasks per batch
                                              None,  # num test images
                                              data.get_image_height(),
                                              data.get_image_width(),
                                              data.get_image_channels()],
                                 name='test_images')
    train_labels = tf.placeholder(tf.float32, [None,  # tasks per batch
                                               None,  # shot
                                               args.way],
                                  name='train_labels')
    test_labels = tf.placeholder(tf.float32, [None,  # tasks per batch
                                              None,  # num test images
                                              args.way],
                                 name='test_labels')
    dropout_keep_prob = tf.placeholder(tf.float32, [], name='dropout_keep_prob')
    L = tf.constant(args.samples, dtype=tf.float32, name="num_samples")

    # Relevant computations for a single task
    def evaluate_task(inputs):
        train_inputs, train_outputs, test_inputs, test_outputs = inputs
        with tf.variable_scope('shared_features'):
            # extract features from train and test data
            features_train = feature_extractor_fn(images=train_inputs,
                                                  output_size=args.d_theta,
                                                  use_batch_norm=True,
                                                  dropout_keep_prob=dropout_keep_prob)
            features_test = feature_extractor_fn(images=test_inputs,
                                                 output_size=args.d_theta,
                                                 use_batch_norm=True,
                                                 dropout_keep_prob=dropout_keep_prob)
        # Infer classification layer from q
        with tf.variable_scope('classifier'):
            classifier = infer_classifier(features_train, train_outputs, args.d_theta, args.way)

        # Local reparameterization trick
        # Compute parameters of q distribution over logits
        weight_mean, bias_mean = classifier['weight_mean'], classifier['bias_mean']
        weight_log_variance, bias_log_variance = classifier['weight_log_variance'], classifier['bias_log_variance']
        logits_mean_test = tf.matmul(features_test, weight_mean) + bias_mean
        logits_log_var_test = \
            tf.log(tf.matmul(features_test ** 2, tf.exp(weight_log_variance)) + tf.exp(bias_log_variance))
        logits_sample_test = sample_normal(logits_mean_test, logits_log_var_test, args.samples)
        test_labels_tiled = tf.tile(tf.expand_dims(test_outputs, 0), [args.samples, 1, 1])
        task_log_py = multinoulli_log_density(inputs=test_labels_tiled, logits=logits_sample_test)
        averaged_predictions = tf.reduce_logsumexp(logits_sample_test, axis=0) - tf.log(L)
        task_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(test_outputs, axis=-1),
                                                        tf.argmax(averaged_predictions, axis=-1)), tf.float32))
        task_score = tf.reduce_logsumexp(task_log_py, axis=0) - tf.log(L)
        task_loss = -tf.reduce_mean(task_score, axis=0)

        return [task_loss, task_accuracy]

    # tf mapping of batch to evaluation function
    batch_output = tf.map_fn(fn=evaluate_task,
                             elems=(train_images, train_labels, test_images, test_labels),
                             dtype=[tf.float32, tf.float32],
                             parallel_iterations=args.tasks_per_batch)

    # average all values across batch
    batch_losses, batch_accuracies = batch_output
    loss = tf.reduce_mean(batch_losses)
    accuracy = tf.reduce_mean(batch_accuracies)

    def evaluate_task_celeba(inputs):
        train_inputs, train_outputs, test_inputs, test_outputs = inputs
        with tf.variable_scope('shared_features'):
            # extract features from train and test data
            features_train = feature_extractor_fn(images=train_inputs,
                                                  output_size=args.d_theta,
                                                  use_batch_norm=True,
                                                  dropout_keep_prob=dropout_keep_prob)
            features_test = feature_extractor_fn(images=test_inputs,
                                                 output_size=args.d_theta,
                                                 use_batch_norm=True,
                                                 dropout_keep_prob=dropout_keep_prob)
        # Infer classification layer from q
        with tf.variable_scope('classifier'):
            classifier = infer_classifier(features_train, train_outputs, args.d_theta, args.way)

        # Local reparameterization trick
        # Compute parameters of q distribution over logits
        weight_mean, bias_mean = classifier['weight_mean'], classifier['bias_mean']
        weight_log_variance, bias_log_variance = classifier['weight_log_variance'], classifier['bias_log_variance']

        return [features_test, weight_mean, bias_mean, weight_log_variance, bias_log_variance]

    # tf mapping of batch to evaluation function
    batch_output_celeba = tf.map_fn(fn=evaluate_task_celeba,
                                    elems=(train_images, train_labels, test_images, test_labels),
                                    dtype=[tf.float32, tf.float32, tf.float32, tf.float32, tf.float32],
                                    parallel_iterations=args.tasks_per_batch)

    # average all values across batch
    features_test, weight_mean, bias_mean, weight_log_variance, bias_log_variance = batch_output_celeba

    with tf.Session() as sess:
        saver = tf.train.Saver()

        if args.mode == 'train' or args.mode == 'train_test':
            # train the model
            optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
            train_step = optimizer.minimize(loss)

            validation_batches = 200
            iteration = 0
            best_validation_accuracy = 0.0
            train_iteration_accuracy = []
            sess.run(tf.global_variables_initializer())
            # Main training loop
            while iteration < args.iterations:
                train_inputs, test_inputs, train_outputs, test_outputs = \
                    data.get_batch('train', args.tasks_per_batch, args.shot, args.way, eval_samples_train)

                feed_dict = {train_images: train_inputs, test_images: test_inputs,
                             train_labels: train_outputs, test_labels: test_outputs,
                             dropout_keep_prob: args.dropout}
                _, iteration_loss, iteration_accuracy = sess.run([train_step, loss, accuracy], feed_dict)
                train_iteration_accuracy.append(iteration_accuracy)
                if (iteration > 0) and (iteration % args.print_freq == 0):
                    # compute accuracy on validation set
                    validation_iteration_accuracy = []
                    validation_iteration = 0
                    while validation_iteration < validation_batches:
                        train_inputs, test_inputs, train_outputs, test_outputs = \
                            data.get_batch('validation', args.tasks_per_batch, args.shot, args.way, eval_samples_test)
                        feed_dict = {train_images: train_inputs, test_images: test_inputs,
                                     train_labels: train_outputs, test_labels: test_outputs,
                                     dropout_keep_prob: 1.0}
                        iteration_accuracy = sess.run(accuracy, feed_dict)
                        validation_iteration_accuracy.append(iteration_accuracy)
                        validation_iteration += 1
                    validation_accuracy = np.array(validation_iteration_accuracy).mean()
                    train_accuracy = np.array(train_iteration_accuracy).mean()

                    # save checkpoint if validation is the best so far
                    if validation_accuracy > best_validation_accuracy:
                        best_validation_accuracy = validation_accuracy
                        saver.save(sess=sess, save_path=checkpoint_path_validation)

                    print_and_log(logfile, 'Iteration: {}, Loss: {:5.3f}, Train-Acc: {:5.3f}, Val-Acc: {:5.3f}'
                                  .format(iteration, iteration_loss, train_accuracy, validation_accuracy))
                    train_iteration_accuracy = []

                iteration += 1
            # save the checkpoint from the final epoch
            saver.save(sess, save_path=checkpoint_path_final)
            print_and_log(logfile, 'Fully-trained model saved to: {}'.format(checkpoint_path_final))
            print_and_log(logfile, 'Best validation accuracy: {:5.3f}'.format(best_validation_accuracy))
            print_and_log(logfile, 'Best validation model saved to: {}'.format(checkpoint_path_validation))

        def test_model(model_path, load=True):
            if load:
                saver.restore(sess, save_path=model_path)
            test_iteration = 0
            test_iteration_accuracy = []
            while test_iteration < test_iterations:
                train_inputs, test_inputs, train_outputs, test_outputs = \
                    data.get_batch('test', test_args_per_batch, args.test_shot, args.test_way,
                                   eval_samples_test)
                feedDict = {train_images: train_inputs, test_images: test_inputs,
                            train_labels: train_outputs, test_labels: test_outputs,
                            dropout_keep_prob: 1.0}
                iter_acc = sess.run(accuracy, feedDict)
                test_iteration_accuracy.append(iter_acc)
                test_iteration += 1
            test_accuracy = np.array(test_iteration_accuracy).mean() * 100.0
            confidence_interval_95 = \
                (196.0 * np.array(test_iteration_accuracy).std()) / np.sqrt(len(test_iteration_accuracy))
            print_and_log(logfile, 'Held out accuracy: {0:5.3f} +/- {1:5.3f} on {2:}'
                          .format(test_accuracy, confidence_interval_95, model_path))

        if args.mode == 'train_test':
            print_and_log(logfile, 'Train Shot: {0:d}, Train Way: {1:d}, Test Shot {2:d}, Test Way {3:d}'
                          .format(args.shot, args.way, args.test_shot, args.test_way))
            # test the model on the final trained model
            # no need to load the model, it was just trained
            test_model(checkpoint_path_final, load=False)

            # test the model on the best validation checkpoint so far
            test_model(checkpoint_path_validation)

        if args.mode == 'test':
            test_model(args.test_model_path)

        def test_celeba(model_path, load=True):
            if load:
                saver.restore(sess, save_path=model_path)
            test_eval_samples = 5  # number of query points
            MAX_TEST_TASKS = 100

            output_tensors = [features_test, weight_mean, bias_mean, weight_log_variance, bias_log_variance]
            all_accuracy = []
            all_clusters = []
            all_log_probs = []

            data.init_testing_params(args.test_way)

            # compute test accuracy
            n_test_examples = 0
            for i in cycle(range(data.n_test_triplets)):
                n_test_examples += 1
                train_inputs, test_inputs, train_outputs, test_outputs = data.get_test_triplet_batch(
                    args.test_shot, args.test_way, test_eval_samples, i)
                # split out batch
                feedDict = {train_images: train_inputs,  # [3, 2*5, 84, 84, 3] support inputs (three same things)
                            test_images: test_inputs,  # [3, 2*5, 84, 84, 3] query inputs (three different things)
                            train_labels: train_outputs,  # [3, 2*5, 2] support outputs
                            test_labels: test_outputs,  # [3, 2*5, 2] query outputs
                            dropout_keep_prob: 1.0}

                ft, wm, bm, wlv, blv = sess.run(output_tensors, feedDict)

                n_samples = 40
                n_marginal = 100

                counter = 0
                cluster = []
                all_preds = []

                # weight_normal = np.random.normal(loc=wm, scale=np.sqrt(np.exp(wlv)))
                # bias_normal = np.random.normal(loc=bm, scale=np.sqrt(np.exp(blv)))
                print("Tested on test_example {}".format(n_test_examples))
                while counter < n_samples:
                    counter += 1
                    # TODO: are w and b correct?
                    w = wm + np.random.normal(loc=0.0, scale=1.0, size=1) * np.sqrt(np.exp(wlv))  # sample from normal(weight_mean, exp(weight_log_variance))
                    b = bm + np.random.normal(loc=0.0, scale=1.0, size=1) * np.sqrt(np.exp(blv))  # sample from normal (bias_mean, exp(bias_log_variance))
                    print(wm.shape, wlv.shape, w.shape)
                    print(bm.shape, blv.shape, b.shape)
                    print(ft.shape)
                    outputs = np.matmul(ft, w) + b
                    # cluster assignments can also be done with marginal
                    task_probs = celeba_test_utils.np_softmax(outputs[0])
                    _task_log_probs = np.log(task_probs[np.where(test_outputs == 1)]).reshape([3, -1])
                    task_log_probs = np.mean(_task_log_probs, axis=-1)
                    most_likely_idx = task_log_probs.argmax()
                    cluster.append(most_likely_idx)
                    all_preds.append(outputs[0][most_likely_idx].argmax(axis=-1))
                    accuracy = celeba_test_utils.np_accuracy(outputs[0], test_outputs, average=False)
                    all_accuracy.append(accuracy[most_likely_idx].mean())
                    all_clusters.append(cluster)
                    # compute marginal for that task

                def compute_marginal_batch(sess, input_tensors, n_samples, inputa, labela, inputb, labelb):
                    """
                    Compute the marginal for a group of tasks
                    """
                    # TODO: move this function to celeba_test_utils
                    ndims = len(inputa.shape)
                    n_tasks = inputa.shape[0]
                    tile_shape = (n_samples,) + (ndims) * (1,)
                    # tile and collapse
                    inputa_tiled = np.reshape(
                        np.tile(inputa[None, :], tile_shape), (-1,) + inputa.shape[1:])
                    inputb_tiled = np.reshape(
                        np.tile(inputb[None, :], tile_shape), (-1,) + inputb.shape[1:])
                    labela_tiled = np.reshape(
                        np.tile(labela[None, :], tile_shape), (-1,) + labela.shape[1:])
                    labelb_tiled = np.reshape(
                        np.tile(labelb[None, :], tile_shape), (-1,) + labelb.shape[1:])
                    feedDict = {train_images: inputa_tiled,
                                test_images: inputb_tiled,
                                train_labels: labela_tiled,
                                test_labels: labelb_tiled,
                                dropout_keep_prob: 1.0}
                    ft, wm, bm, wlv, blv = sess.run(input_tensors, feedDict)
                    # TODO: are w and b correct?
                    w = wm + np.random.normal(loc=0.0, scale=1.0, size=1) * np.sqrt(np.exp(wlv))  # sample from normal(weight_mean, exp(weight_log_variance))
                    b = bm + np.random.normal(loc=0.0, scale=1.0, size=1) * np.sqrt(np.exp(blv))  # sample from normal (bias_mean, exp(bias_log_variance))
                    outputs = np.matmul(ft, w) + b
                    probs = celeba_test_utils.np_softmax(outputs[0].reshape([-1, args.test_way]))
                    probs = probs.reshape((n_samples, n_tasks) + outputs[0].shape[1:])
                    # average over samples
                    marginal_probs = probs.mean(axis=0)
                    preds = np.argmax(marginal_probs, axis=-1)
                    return marginal_probs, preds

                # TODO: call celeba_test_utils.compute_marginal_batch
                marginal_p, _ = compute_marginal_batch(sess, output_tensors, n_marginal, train_inputs, train_outputs, test_inputs, test_outputs)
                all_log_probs.append(np.mean(np.log(marginal_p[np.where(test_outputs == 1)])))
                if n_test_examples >= MAX_TEST_TASKS:
                    break

            print("Test")
            print("all_accuracy: {}".format(np.mean(all_accuracy)))
            print("all_accuracy confidence: {}".format(np.std(all_accuracy) * 1.96 / np.sqrt(len(all_accuracy))))
            coverage = [len(np.unique(cl)) for cl in all_clusters]
            print("all_coverage: {}".format(np.mean(coverage)))
            print("all_coverage confidence: {}".format(np.std(coverage) * 1.96 / np.sqrt(len(coverage))))
            print("all_logprob: {}".format(np.mean(all_log_probs)))
            print("all_logprobs confidence: {}".format(np.std(all_log_probs) * 1.96 / np.sqrt(len(all_log_probs))))

        if args.mode == 'test_celeba':
            test_celeba(args.test_model_path)


if __name__ == "__main__":
    tf.app.run()
