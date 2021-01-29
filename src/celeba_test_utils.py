import tensorflow as tf
import numpy as np
from itertools import cycle


def np_softmax(logits, axis=-1):
    probs = np.exp(logits - np.max(logits, axis=axis, keepdims=True))
    probs = probs / np.sum(probs, axis=axis, keepdims=True)
    return probs


def np_accuracy(probs, labels, axis=-1, average=True):
    """
    Labels should be one hot
    """
    assert probs.shape == labels.shape, "{} shape != {} shape".format(probs, labels)
    if average:
        return np.mean(probs.argmax(axis=axis) == labels.argmax(axis=axis))
    else:
        return probs.argmax(axis=axis) == labels.argmax(axis=axis)


def compute_marginal_batch(model, sess, test_way, test_shot, input_tensors, n_samples, inputa, labela, inputb, labelb):
    """
    Compute the marginal for a group of tasks
    """    
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
    feed_dict = {model.inputa: inputa_tiled,
                 model.inputb: inputb_tiled,
                 model.labela: labela_tiled,
                 model.labelb: labelb_tiled,
                 model.meta_lr:0.0}
    outputs = sess.run(input_tensors, feed_dict)
    probs = np_softmax(outputs[0].reshape([-1, 2]))
    probs = probs.reshape((n_samples, n_tasks) + outputs[0].shape[1:])
    # average over samples
    marginal_probs = probs.mean(axis=0)
    preds = np.argmax(marginal_probs, axis=-1)
    return marginal_probs, preds


# def celeba_test(data, test_shot, test_way, test_eval_samples, model, sess):
#     MAX_TEST_TASKS = 100

#     output_tensors = [model.output]  # model.output is the predicted results corresponding to query inputs
#     all_accuracy = []
#     all_clusters = []
#     all_log_probs = []
    
#     dataset.init_testing_params(way)

#     # compute test accuracy
#     n_test_examples = 0
#     for i in cycle(range(data.n_test_triplets)):  # data_generator.n_test_triplets is 50 when num_support=num_query=5
#         n_test_examples += 1
#         train_inputs, test_inputs, train_outputs, test_outputs = data.get_test_triplet_batch(
#             test_shot, test_way, test_eval_samples, i)
#         # split out batch
#         feedDict = {train_images: train_inputs, # [3, 2*5, 84, 84, 3] support inputs (three same things)
#                     test_images: test_inputs,   # [3, 2*5, 84, 84, 3] query inputs (three different things)
#                     train_labels: train_outputs, # [3, 2*5, 2] support outputs
#                     test_labels: test_outputs,   # [3, 2*5, 2] query outputs
#                     dropout_keep_prob: 1.0}

#         n_samples = 40
#         n_marginal = 100

#         counter = 0
#         cluster = []
#         all_preds = []

#         print("Tested on test_example {}".format(n_test_examples))
#         while counter < n_samples:
#             counter += 1
#             outputs = sess.run([averaged_predictions], feed_dict)
#             # cluster assignments can also be done with marginal
#             task_probs = np_softmax(outputs[0])
#             _task_log_probs = np.log(task_probs[np.where(test_outputs == 1)]).reshape([3, -1])
#             task_log_probs = np.mean(_task_log_probs, axis=-1)
#             most_likely_idx = task_log_probs.argmax()
#             cluster.append(most_likely_idx)
#             all_preds.append(outputs[0][most_likely_idx].argmax(axis=-1))
#             accuracy = np_accuracy(outputs[0], test_outputs, average=False)
#             all_accuracy.append(accuracy[most_likely_idx].mean())
#             all_clusters.append(cluster)
#             # compute marginal for that task

#         marginal_p, _ = compute_marginal_batch(
#             model, sess, output_tensors, n_marginal, inputa,
#             labela, inputb, labelb, n_test_examples)
#         all_log_probs.append(np.mean(np.log(marginal_p[np.where(labelb == 1)])))

#         if n_test_examples >= MAX_TEST_TASKS:
#             break

#     print("Test")
#     print("all_accuracy: {}".format(np.mean(all_accuracy)))
#     print("all_accuracy confidence: {}".format(np.std(all_accuracy) * 1.96 / np.sqrt(len(all_accuracy))))
#     coverage = [len(np.unique(cl)) for cl in all_clusters]
#     print("all_coverage: {}".format(np.mean(coverage)))
#     print("all_coverage confidence: {}".format(np.std(coverage) * 1.96 / np.sqrt(len(coverage))))
#     print("all_logprob: {}".format(np.mean(all_log_probs)))
#     print("all_logprobs confidence: {}".format(np.std(all_log_probs) * 1.96 / np.sqrt(len(all_log_probs))))


# if __name__ == "__main__":

#     data_generator = ...  # initialize dataloader (num_support=num_query=5)
#     model = ...  # initialize model
#     checkpoint_folder = ...  # it may store files like model60000.meta, model60000.index, model60000.data-00000-of-00001, checkpoint
#     test_iter = ...
#     saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), max_to_keep=...)
#     sess = tf.InteractiveSession()

#     model_file = tf.train.latest_checkpoint(checkpoint_folder)
#     if test_iter > 0:
#         model_file = model_file[:model_file.index('model')] + 'model' + str(test_iter)
#     if model_file:
#         ind1 = model_file.index('model')
#         resume_itr = int(model_file[ind1+5:])
#         print("Restoring model weights from " + model_file)
#         saver.restore(sess, model_file)

#     celeba_test(model, data_generator, sess)
