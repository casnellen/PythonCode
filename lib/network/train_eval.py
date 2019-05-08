"""
Functions for training and/or evaluating Keras models
"""

__author__ = 'clee'
# Python 3.6.1

# Standard lib
import os
import sys
import datetime
from itertools import cycle

# Additional lib
import numpy as np
import matplotlib.pyplot as plt
import keras
from sklearn.metrics import confusion_matrix, roc_curve, auc
from lib.plot import plot_cm

# lib imports
from lib.util import cnf_matrix

def batch_gen(spectrograms, labels, batch_size=4):
    """
    Batch generator for Keras's fit_generator()

    :param spectrograms:            (n, x, y)-shaped array-like         n samples, x=time axis, y=frequency axis
    :param labels:                  (n,)-shaped array-like              Corresponding class labels
    :param batch_size:              int
    :return:
    """

    num_cls = len(np.unique(labels))

    while True:
        for i in range(0, spectrograms.shape[0]-batch_size, batch_size):
            if (i % spectrograms.shape[0] == 0) or (i > spectrograms.shape[0]):
                rand_perm = np.random.permutation(len(spectrograms))
            samp_ind = list(rand_perm[i:i+batch_size])
            samp_ind.sort()
            batch_sg = spectrograms[samp_ind]
            batch_tgt = labels[samp_ind]
            yield(batch_sg.reshape(batch_sg.shape+(1,)), keras.utils.to_categorical(batch_tgt, num_classes=num_cls))
            
def batch_gen_binary(spectrograms, labels, batch_size=4):
    """
    Batch generator for Keras's fit_generator()

    :param spectrograms:            (n, x, y)-shaped array-like         n samples, x=time axis, y=frequency axis
    :param labels:                  (n,)-shaped array-like              Corresponding class labels
    :param batch_size:              int
    :return:
    """

    # num_cls = len(np.unique(labels))

    while True:
        for i in range(0, spectrograms.shape[0]-batch_size, batch_size):
            if (i % spectrograms.shape[0] == 0) or (i > spectrograms.shape[0]):
                rand_perm = np.random.permutation(len(spectrograms))
            samp_ind = list(rand_perm[i:i+batch_size])
            samp_ind.sort()
            batch_sg = spectrograms[samp_ind]
            batch_tgt = labels[samp_ind]
            yield(batch_sg.reshape(batch_sg.shape+(1,)), batch_tgt)


def train_eval(model, train_set, test_set, run_epochs, path_results, num_cls, cls_names,
               val_set=None,
               start_epoch=0,
               verbose=1,
               batch_size=4,
               experiment_notes="",
               show=False,
               prev_hist=None):
    """
    Train and evaluate Keras model
    *_set[0] should have shape ([n samples], [x=time axis], [y=frequency axis]) containing the input spectrograms
    *_set[1] should have shape ([n samples],) containing the corresponding class labels
    If using in a loop, feed return values into start_epoch and prev_hist to keep accurate epoch count

    :param model:               keras.engine.training.Model         Compiled Keras model
    :param train_set:           2-tuple of array-likes              Training set (see above)
    :param test_set:            2-tuple of array-likes              Testing set (see above)
    :param path_results         str                                 Output directory
    :param num_cls              int                                 Total number of classes
    :param cls_names            list of str                         List of class names ordered by corresponding class label
    :param run_epochs:          int                                 Number of epochs to run
    :param val_set:             2-tuple of array-likes              Validation set (see above)
    :param start_epoch:         int                                 Current epoch of model before training
    :param verbose:             int                                 Keras verbosity mode (0, 1, or 2)
    :param batch_size:          int
    :param experiment_notes:    str                                 Notes about experiment
    :param show:                bool                                Whether to show plots
    :param prev_hist:           dict of arrays                      Previous interval's training history

    :return count_epoch:        int                                 Current epoch of model after training
    :return curr_hist:          dict of arrays                      Current interval's training history
    """

    # Set up results directory
    rundate = datetime.datetime.now().strftime('%b%d%y-%H%M%S')
    os.mkdir(path_results + '/' + rundate)
    os.mkdir(path_results + '/' + rundate + '/models')

    # Save model summary
    orig_stdout = sys.stdout
    summ_file = open(path_results + '/' + rundate + '/models/modelsummary.txt', 'w')
    sys.stdout = summ_file
    print(model.summary())
    sys.stdout = orig_stdout
    summ_file.close()

    # Save experiment notes
    orig_stdout = sys.stdout
    experiment_notes_file = open(path_results + '/' + rundate + '/_experiment_notes.txt', 'w')
    sys.stdout = experiment_notes_file
    print(experiment_notes)
    sys.stdout = orig_stdout
    experiment_notes_file.close()

    # Perform training
    count_epoch = start_epoch + run_epochs
    history = model.fit_generator(
        generator=batch_gen(train_set[0], train_set[1], batch_size),
        steps_per_epoch=train_set[0].shape[0]//batch_size,
        epochs=count_epoch,
        initial_epoch=start_epoch,
        verbose=verbose,
        workers=1,
        validation_data=[val_set[0][...].reshape(val_set[0].shape + (1,)), keras.utils.to_categorical(val_set[1], num_classes=num_cls)])

    # Save trained model and training info
    model.save(path_results + '/' + rundate + '/models/epochs' + str(start_epoch) + '-' + str(count_epoch) +'.h5')

    if prev_hist is None:
        acc = np.asarray(history.history['acc'])
        val_acc = np.asarray(history.history['val_acc'])
        loss = np.asarray(history.history['loss'])
        val_loss = np.asarray(history.history['val_loss'])
    else:
        acc = np.append(prev_hist['acc'], np.asarray(history.history['acc']))
        val_acc = np.append(prev_hist['val_acc'], np.asarray(history.history['val_acc']))
        loss = np.append(prev_hist['loss'], np.asarray(history.history['loss']))
        val_loss = np.append(prev_hist['val_loss'], np.asarray(history.history['val_loss']))

    curr_hist = {'acc':acc, 'val_acc':val_acc, 'loss':loss, 'val_loss':val_loss}

    np.save(path_results + '/' + rundate + '/history_acc.npy', acc)
    np.save(path_results + '/' + rundate + '/history_valacc.npy', val_acc)
    np.save(path_results + '/' + rundate + '/history_loss.npy', loss)
    np.save(path_results + '/' + rundate + '/history_valloss.npy', val_loss)

    plt.figure(figsize=(10, 5))
    plt.plot(acc, 'b', label="Training")
    plt.plot(val_acc, 'g', label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc='upper left')
    plt.savefig(path_results + '/' + rundate + '/accuracy.png', bbox_inches='tight')

    plt.figure(figsize=(10, 5))
    plt.plot(loss, 'b', label="Training")
    plt.plot(val_loss, 'g', label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='upper left')
    plt.savefig(path_results + '/' + rundate + '/loss.png', bbox_inches='tight')
    if show:
        plt.show()

    # Evaluate on test set
    print("Evaluating model on test set")
    test_loss_acc = model.evaluate(test_set[0][...].reshape(test_set[0].shape + (1,)),
                                   keras.utils.to_categorical(test_set[1], num_classes=num_cls),
                                   batch_size=batch_size,
                                   verbose=verbose)
    print("\nComputing scores on test set")
    scores = model.predict(test_set[0][...].reshape(test_set[0].shape + (1,)),
                           batch_size=batch_size,
                           verbose=verbose)
    arr_scores = np.asarray(scores)
    np.save(path_results + '/' + rundate + '/scores.npy', arr_scores)

    cm = confusion_matrix(test_set[1], np.argmax(scores, axis=-1))
    plot_cm(cm, cls_names=cls_names, normalize=True,
            title="Loss: "+str(test_loss_acc[0])+" | Acc: "+str(test_loss_acc[1]),
            path_save=path_results + '/' + rundate + '/cm.png',
            show=show)

    print('\n\n')

    return count_epoch, curr_hist


def test_eval(model, test_set, path_results, num_cls, cls_names,
              verbose=1,
              batch_size=4,
              experiment_notes="",
              show=False,
              cm_figsize=(10,10)):
    """
    Evaluate a pretrained Keras model

    :param model:               keras.engine.training.Model         Compiled Keras model
    :param test_set:            2-tuple of array-likes              Testing set (see above)
    :param path_results         str                                 Output directory
    :param num_cls              int                                 Total number of classes
    :param cls_names            list of str                         List of class names ordered by corresponding class label
    :param verbose:             int                                 Keras verbosity mode (0, 1, or 2)
    :param batch_size:          int
    :param experiment_notes:    str                                 Notes about experiment
    :param show:                bool                                Whether to show plots
    :param cm_figsize:          2-tuple of ints                     Matplotlib figure size for confusion matrix

    :return:                    None
    """

    # Set up results directory
    rundate = datetime.datetime.now().strftime('%b%d%y-%H%M%S')
    os.mkdir(path_results + '/' + rundate)
    os.mkdir(path_results + '/' + rundate + '/models')

    # Save model summary
    orig_stdout = sys.stdout
    summ_file = open(path_results + '/' + rundate + '/models/modelsummary.txt', 'w')
    sys.stdout = summ_file
    print(model.summary())
    sys.stdout = orig_stdout
    summ_file.close()

    # Save experiment notes
    orig_stdout = sys.stdout
    experiment_notes_file = open(path_results + '/' + rundate + '/_experiment_notes.txt', 'w')
    sys.stdout = experiment_notes_file
    print(experiment_notes)
    sys.stdout = orig_stdout
    experiment_notes_file.close()

    # Evaluate on test set
    print("Evaluating model on test set")
    test_loss_acc = model.evaluate(test_set[0][...].reshape(test_set[0].shape + (1,)),
                                   keras.utils.to_categorical(test_set[1], num_classes=num_cls),
                                   batch_size=batch_size,
                                   verbose=verbose)
    print("\nComputing scores on test set")
    scores = model.predict(test_set[0][...].reshape(test_set[0].shape + (1,)),
                           batch_size=batch_size,
                           verbose=verbose)
    arr_scores = np.asarray(scores)
    np.save(path_results + '/' + rundate + '/scores.npy', arr_scores)

    cm = confusion_matrix(test_set[1], np.argmax(scores, axis=-1))
    plot_cm(cm, cls_names=cls_names, normalize=True,
            title="Loss: "+str(test_loss_acc[0])+" | Acc: "+str(test_loss_acc[1]),
            path_save = path_results + '/' + rundate + '/cm.png',
            show=show,
            figsize=cm_figsize)


def train_eval_close_emitter(model, train_set, same_set, simi_set, run_epochs, path_results, num_cls, cls_names,
                             val_set=None,
                             start_epoch=0,
                             verbose=1,
                             batch_size=4,
                             experiment_notes="",
                             show=False,
                             prev_hist=None):
    """
    Train and evaluate Keras model
    *_set[0] should have shape ([n samples], [x=time axis], [y=frequency axis]) containing the input spectrograms
    *_set[1] should have shape ([n samples],) containing the corresponding class labels
    same/simi_set[2] should have shape ([n samples],) containing the corresponding true class names
    If using in a loop, feed return values into start_epoch and prev_hist to keep accurate epoch count

    :param model:               keras.engine.training.Model         Compiled Keras model
    :param train_set:           2-tuple of array-likes              Training set (see above)
    :param same_set:            3-tuple of array-likes              Testing set (see above)
    :param simi_set:            3-tuple of array-likes              Testing set (see above)
    :param path_results         str                                 Output directory
    :param num_cls              int                                 Total number of classes
    :param cls_names            list of str                         List of class names ordered by corresponding class label
    :param run_epochs:          int                                 Number of epochs to run
    :param val_set:             2-tuple of array-likes              Validation set (see above)
    :param start_epoch:         int                                 Current epoch of model before training
    :param verbose:             int                                 Keras verbosity mode (0, 1, or 2)
    :param batch_size:          int
    :param experiment_notes:    str                                 Notes about experiment
    :param show:                bool                                Whether to show plots
    :param prev_hist:           dict of arrays                      Previous interval's training history

    :return count_epoch:        int                                 Current epoch of model after training
    :return curr_hist:          dict of arrays                      Current interval's training history
    """

    # Set up results directory
    rundate = datetime.datetime.now().strftime('%b%d%y-%H%M%S')
    os.mkdir(path_results + '/' + rundate)
    os.mkdir(path_results + '/' + rundate + '/models')

    # Save model summary
    orig_stdout = sys.stdout
    summ_file = open(path_results + '/' + rundate + '/models/modelsummary.txt', 'w')
    sys.stdout = summ_file
    print(model.summary())
    sys.stdout = orig_stdout
    summ_file.close()

    # Save experiment notes
    orig_stdout = sys.stdout
    experiment_notes_file = open(path_results + '/' + rundate + '/_experiment_notes.txt', 'w')
    sys.stdout = experiment_notes_file
    print(experiment_notes)
    sys.stdout = orig_stdout
    experiment_notes_file.close()

    # Perform training
    count_epoch = start_epoch + run_epochs
    history = model.fit_generator(
        generator=batch_gen(train_set[0], train_set[1], batch_size),
        steps_per_epoch=train_set[0].shape[0] // batch_size,
        epochs=count_epoch,
        initial_epoch=start_epoch,
        verbose=verbose,
        workers=1,
        validation_data=[val_set[0][...].reshape(val_set[0].shape + (1,)),
                         keras.utils.to_categorical(val_set[1], num_classes=num_cls)])

    # Save trained model and training info
    model.save(path_results + '/' + rundate + '/models/epochs' + str(start_epoch) + '-' + str(count_epoch) + '.h5')

    if prev_hist is None:
        acc = np.asarray(history.history['acc'])
        val_acc = np.asarray(history.history['val_acc'])
        loss = np.asarray(history.history['loss'])
        val_loss = np.asarray(history.history['val_loss'])
    else:
        acc = np.append(prev_hist['acc'], np.asarray(history.history['acc']))
        val_acc = np.append(prev_hist['val_acc'], np.asarray(history.history['val_acc']))
        loss = np.append(prev_hist['loss'], np.asarray(history.history['loss']))
        val_loss = np.append(prev_hist['val_loss'], np.asarray(history.history['val_loss']))

    curr_hist = {'acc': acc, 'val_acc': val_acc, 'loss': loss, 'val_loss': val_loss}

    np.save(path_results + '/' + rundate + '/history_acc.npy', acc)
    np.save(path_results + '/' + rundate + '/history_valacc.npy', val_acc)
    np.save(path_results + '/' + rundate + '/history_loss.npy', loss)
    np.save(path_results + '/' + rundate + '/history_valloss.npy', val_loss)

    plt.figure(figsize=(10, 5))
    plt.plot(acc, 'b', label="Training")
    plt.plot(val_acc, 'g', label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc='upper left')
    plt.savefig(path_results + '/' + rundate + '/accuracy.png', bbox_inches='tight')

    plt.figure(figsize=(10, 5))
    plt.plot(loss, 'b', label="Training")
    plt.plot(val_loss, 'g', label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='upper left')
    plt.savefig(path_results + '/' + rundate + '/loss.png', bbox_inches='tight')
    if show:
        plt.show()

    # Evaluate on validation set
    print("Evaluating model on validation set")
    val_loss_acc = model.evaluate(val_set[0][...].reshape(val_set[0].shape + (1,)),
                                   keras.utils.to_categorical(val_set[1], num_classes=num_cls),
                                   batch_size=batch_size,
                                   verbose=verbose)
    print("\nComputing scores on validation set")
    scores = model.predict(val_set[0][...].reshape(val_set[0].shape + (1,)),
                           batch_size=batch_size,
                           verbose=verbose)
    arr_scores = np.asarray(scores)
    np.save(path_results + '/' + rundate + '/val_scores.npy', arr_scores)

    cm = confusion_matrix(val_set[1], np.argmax(scores, axis=-1))
    plot_cm(cm, cls_names=cls_names, normalize=True,
            title="Loss: " + str(val_loss_acc[0]) + " | Acc: " + str(val_loss_acc[1]),
            path_save=path_results + '/' + rundate + '/val_cm.png',
            show=show)

    print('\n\n')

    # Evaluate on same set
    print("Evaluating model on same set")
    same_loss_acc = model.evaluate(same_set[0][...].reshape(same_set[0].shape + (1,)),
                                   keras.utils.to_categorical(same_set[1], num_classes=num_cls),
                                   batch_size=batch_size,
                                   verbose=verbose)
    print("\nComputing scores on same set")
    same_scores = model.predict(same_set[0][...].reshape(same_set[0].shape + (1,)),
                                batch_size=batch_size,
                                verbose=verbose)
    same_arr_scores = np.asarray(same_scores)
    np.save(path_results + '/' + rundate + '/same_scores.npy', same_arr_scores)

    # Build same confusion matrix
    cm = cnf_matrix(same_set[1], np.argmax(same_scores, axis=-1), num_cls)
    same_cls_preds = cls_names.copy()
    same_cls_trues = []
    list_same_cls = list(same_set[1])
    same_classes = sorted(np.unique(same_set[1]))
    for cls in same_classes:
        true_emit = same_set[2][list_same_cls.index(cls)]
        same_cls_preds[cls] = same_cls_preds[cls] + "(" + str(true_emit) + ")"
        same_cls_trues.append(str(true_emit))
    cm = cm[same_classes]
    same_cls_cm = cm[:, same_classes]
    same_cls_labels = [same_cls_preds[cls] for cls in same_classes]
    other_cls_cm = cm[:, list(set(range(num_cls)) - set(same_classes))]
    other_cls_labels = [same_cls_preds[cls] for cls in list(set(range(num_cls)) - set(same_classes))]
    cm = np.concatenate([same_cls_cm, other_cls_cm], axis=1)
    same_cls_preds = same_cls_labels + other_cls_labels

    plot_cm(cm, cls_names=same_cls_preds, normalize=True,
            title="Loss: " + str(same_loss_acc[0]) + " | Acc: " + str(same_loss_acc[1]),
            path_save=path_results + '/' + rundate + '/same_cm.png',
            show=show,
            true_cls_names=same_cls_trues)

    print('\n\n')

    # Evaluate on similar set
    print("Evaluating model on simi set")
    simi_loss_acc = model.evaluate(simi_set[0][...].reshape(simi_set[0].shape + (1,)),
                                   keras.utils.to_categorical(simi_set[1], num_classes=num_cls),
                                   batch_size=batch_size,
                                   verbose=verbose)
    print("\nComputing scores on simi set")
    simi_scores = model.predict(simi_set[0][...].reshape(simi_set[0].shape + (1,)),
                                batch_size=batch_size,
                                verbose=verbose)
    simi_arr_scores = np.asarray(simi_scores)
    np.save(path_results + '/' + rundate + '/simi_scores.npy', simi_arr_scores)

    # Build simi confusion matrix
    cm = cnf_matrix(simi_set[1], np.argmax(simi_scores, axis=-1), num_cls)
    simi_cls_preds = cls_names.copy()
    simi_cls_trues = []
    list_simi_cls = list(simi_set[1])
    simi_classes = sorted(np.unique(simi_set[1]))
    for cls in simi_classes:
        true_emit = simi_set[2][list_simi_cls.index(cls)]
        simi_cls_preds[cls] = simi_cls_preds[cls] + "(" + str(true_emit) + ")"
        simi_cls_trues.append(str(true_emit))
    cm = cm[simi_classes]
    simi_cls_cm = cm[:, simi_classes]
    simi_cls_labels = [simi_cls_preds[cls] for cls in simi_classes]
    other_cls_cm = cm[:, list(set(range(num_cls)) - set(simi_classes))]
    other_cls_labels = [simi_cls_preds[cls] for cls in list(set(range(num_cls)) - set(simi_classes))]
    cm = np.concatenate([simi_cls_cm, other_cls_cm], axis=1)
    simi_cls_preds = simi_cls_labels + other_cls_labels

    plot_cm(cm, cls_names=simi_cls_preds, normalize=True,
            title="Loss: " + str(simi_loss_acc[0]) + " | Acc: " + str(simi_loss_acc[1]),
            path_save=path_results + '/' + rundate + '/simi_cm.png',
            show=show,
            true_cls_names=simi_cls_trues)

    print('\n\n')
    return count_epoch, curr_hist


def test_eval_close_emitter(model, val_set, same_set, simi_set, path_results, num_cls, cls_names,
              verbose=1,
              batch_size=4,
              experiment_notes="",
              show=False,
              cm_figsize=(10,10),
              hist_figsize=(10,5),
              bins=100,
              hist_xlim=(0, 1.05),
              threat_names=None):
    """
    Evaluate a pretrained Keras model
    *_set[0] should have shape ([n samples], [x=time axis], [y=frequency axis]) containing the input spectrograms
    *_set[1] should have shape ([n samples],) containing the corresponding class labels
    same/simi_set[2] should have shape ([n samples],) containing the corresponding true class names
    If using in a loop, feed return values into start_epoch and prev_hist to keep accurate epoch count

    :param model:               keras.engine.training.Model         Compiled Keras model
    :param val_set:             2-tuple of array-likes              Testing set (see above)
    :param same_set:            3-tuple of array-likes              Testing set (see above)
    :param simi_set:            3-tuple of array-likes              Testing set (see above)
    :param path_results         str                                 Output directory
    :param num_cls              int                                 Total number of classes
    :param cls_names            list of str                         List of class names ordered by corresponding class label
    :param verbose:             int                                 Keras verbosity mode (0, 1, or 2)
    :param batch_size:          int
    :param experiment_notes:    str                                 Notes about experiment
    :param show:                bool                                Whether to show plots
    :param cm_figsize:          2-tuple of ints                     Matplotlib figure size for confusion matrix & ROC plot
    :param hist_figsize         2-tuple of ints                     Matplotlib figure size for histograms
    :param bins                 int                                 Number of bins in score histograms
    :param hist_xlim            2-tuple of floats                   X-axis limits of histograms
    :param threat_names         dict of strings                     key: Emitter number, value: Corresponding threat

    :return:                    None
    """

    # Set up results directory
    rundate = datetime.datetime.now().strftime('%b%d%y-%H%M%S')
    os.mkdir(path_results + '/' + rundate)
    os.mkdir(path_results + '/' + rundate + '/models')

    # Save model summary
    orig_stdout = sys.stdout
    summ_file = open(path_results + '/' + rundate + '/models/modelsummary.txt', 'w')
    sys.stdout = summ_file
    print(model.summary())
    sys.stdout = orig_stdout
    summ_file.close()

    # Save experiment notes
    orig_stdout = sys.stdout
    experiment_notes_file = open(path_results + '/' + rundate + '/_experiment_notes.txt', 'w')
    sys.stdout = experiment_notes_file
    print(experiment_notes)
    sys.stdout = orig_stdout
    experiment_notes_file.close()

    # Evaluate on validation set
    print("Evaluating model on validation set")
    val_loss_acc = model.evaluate(val_set[0][...].reshape(val_set[0].shape + (1,)),
                                  keras.utils.to_categorical(val_set[1], num_classes=num_cls),
                                  batch_size=batch_size,
                                  verbose=verbose)
    print("\nComputing scores on validation set")
    scores = model.predict(val_set[0][...].reshape(val_set[0].shape + (1,)),
                           batch_size=batch_size,
                           verbose=verbose)
    arr_scores = np.asarray(scores)
    np.save(path_results + '/' + rundate + '/val_scores.npy', arr_scores)

    if threat_names is not None:
        val_cm_cls_names = [threat_names[cls] for cls in cls_names]
    else:
        val_cm_cls_names = cls_names

    cm = confusion_matrix(val_set[1], np.argmax(scores, axis=-1))
    plot_cm(cm, cls_names=val_cm_cls_names, normalize=True,
            title="Total Accuracy: " + "%.2f" % (val_loss_acc[1]*100) + "%",
            path_save=path_results + '/' + rundate + '/val_cm.png',
            show=show)

    print('\n\n')

    # Evaluate on same set
    print("Evaluating model on same set")
    same_loss_acc = model.evaluate(same_set[0][...].reshape(same_set[0].shape + (1,)),
                                   keras.utils.to_categorical(same_set[1], num_classes=num_cls),
                                   batch_size=batch_size,
                                   verbose=verbose)
    print("\nComputing scores on same set")
    same_scores = model.predict(same_set[0][...].reshape(same_set[0].shape + (1,)),
                                batch_size=batch_size,
                                verbose=verbose)
    same_arr_scores = np.asarray(same_scores)
    np.save(path_results + '/' + rundate + '/same_scores.npy', same_arr_scores)

    # Build same confusion matrix
    cm = cnf_matrix(same_set[1], np.argmax(same_scores, axis=-1), num_cls)
    same_cls_preds = cls_names.copy()
    same_cls_trues = []
    list_same_cls = list(same_set[1])
    same_classes = sorted(np.unique(same_set[1]))
    for cls in same_classes:
        true_emit = same_set[2][list_same_cls.index(cls)]
        same_cls_preds[cls] = same_cls_preds[cls] + "(" + str(true_emit) + ")"
        same_cls_trues.append(str(true_emit))
    cm = cm[same_classes]
    same_cls_cm = cm[:, same_classes]
    same_cls_labels = [same_cls_preds[cls] for cls in same_classes]
    other_cls_cm = cm[:, list(set(range(num_cls)) - set(same_classes))]
    other_cls_labels = [same_cls_preds[cls] for cls in list(set(range(num_cls)) - set(same_classes))]
    cm = np.concatenate([same_cls_cm, other_cls_cm], axis=1)
    same_cls_preds = same_cls_labels + other_cls_labels

    plot_cm(cm, cls_names=same_cls_preds, normalize=True,
            title="Total Accuracy: " + "%.2f" % (same_loss_acc[1]*100) + "%",
            path_save=path_results + '/' + rundate + '/same_cm.png',
            show=show,
            true_cls_names=same_cls_trues)

    print('\n\n')

    # Evaluate on similar set
    print("Evaluating model on simi set")
    simi_loss_acc = model.evaluate(simi_set[0][...].reshape(simi_set[0].shape + (1,)),
                                   keras.utils.to_categorical(simi_set[1], num_classes=num_cls),
                                   batch_size=batch_size,
                                   verbose=verbose)
    print("\nComputing scores on simi set")
    simi_scores = model.predict(simi_set[0][...].reshape(simi_set[0].shape + (1,)),
                                batch_size=batch_size,
                                verbose=verbose)
    simi_arr_scores = np.asarray(simi_scores)
    np.save(path_results + '/' + rundate + '/simi_scores.npy', simi_arr_scores)

    # Build simi confusion matrix
    cm = cnf_matrix(simi_set[1], np.argmax(simi_scores, axis=-1), num_cls)
    simi_cls_preds = cls_names.copy()
    simi_cls_trues = []
    list_simi_cls = list(simi_set[1])
    simi_classes = sorted(np.unique(simi_set[1]))
    for cls in simi_classes:
        true_emit = simi_set[2][list_simi_cls.index(cls)]
        simi_cls_preds[cls] = simi_cls_preds[cls] + "(" + str(true_emit) + ")"
        simi_cls_trues.append(str(true_emit))
    cm = cm[simi_classes]
    simi_cls_cm = cm[:, simi_classes]
    simi_cls_labels = [simi_cls_preds[cls] for cls in simi_classes]
    other_cls_cm = cm[:, list(set(range(num_cls)) - set(simi_classes))]
    other_cls_labels = [simi_cls_preds[cls] for cls in list(set(range(num_cls)) - set(simi_classes))]
    cm = np.concatenate([simi_cls_cm, other_cls_cm], axis=1)
    simi_cls_preds = simi_cls_labels + other_cls_labels

    plot_cm(cm, cls_names=simi_cls_preds, normalize=True,
            title="Total Accuracy: " + "%.2f" % (simi_loss_acc[1]*100) + "%",
            path_save=path_results + '/' + rundate + '/simi_cm.png',
            show=show,
            true_cls_names=simi_cls_trues)

    # Score histograms
    # ASSUMES EVEN-CLASS, CLASS-NAME-SORTED, UNSHUFFLED SETS
    if arr_scores.shape[0] % num_cls != 0:
        raise ValueError("Number of validation samples does not evenly divide into number of classes")
    else:
        samp_per_emit = arr_scores.shape[0]/num_cls
    fig = plt.figure(figsize=hist_figsize)
    for i in range(num_cls):
        cls_scores = arr_scores[int(i*samp_per_emit):int(i*samp_per_emit+samp_per_emit)]
        plt.hist(np.max(cls_scores, axis=1), range=(0, 1.05), bins=bins, alpha=1/num_cls, label=cls_names[i])
    plt.title("Validation Set Per-class Score Distributions")
    plt.xlim(hist_xlim)
    plt.ylim(0, arr_scores.shape[0]/num_cls)
    plt.legend(loc="upper left")
    plt.savefig(path_results + '/' + rundate + '/val_score_hist.png', bbox_inches='tight')
    plt.show()

    same_cls_names = np.unique(same_set[2])
    same_num_cls = same_cls_names.shape[0]
    if same_arr_scores.shape[0] % same_num_cls !=0:
        raise ValueError("Number of same set samples does not evenly divide into number of classes")
    else:
        same_samp_per_emit = same_arr_scores.shape[0]/same_num_cls
    fig = plt.figure(figsize=hist_figsize)
    for i in range(same_num_cls):
        same_cls_scores = same_arr_scores[int(i*same_samp_per_emit):int(i*same_samp_per_emit+same_samp_per_emit)]
        plt.hist(np.max(same_cls_scores, axis=1), range=(0, 1.05), bins=bins, alpha=1/same_num_cls, label=str(same_cls_names[i]))
    plt.title("Same Set Per-class Score Distributions")
    plt.xlim(hist_xlim)
    plt.ylim(0, same_arr_scores.shape[0]/same_num_cls)
    plt.legend(loc='upper left')
    plt.savefig(path_results + '/' + rundate + '/same_score_hist.png', bbox_inches='tight')
    plt.show()

    simi_cls_names = np.unique(simi_set[2])
    simi_num_cls = simi_cls_names.shape[0]
    if simi_arr_scores.shape[0] % simi_num_cls != 0:
        raise ValueError("Number of simi set samples does not evenly divide into number of classes")
    else:
        simi_samp_per_emit = simi_arr_scores.shape[0] / simi_num_cls
    fig = plt.figure(figsize=hist_figsize)
    for i in range(simi_num_cls):
        simi_cls_scores = simi_arr_scores[int(i * simi_samp_per_emit):int(i * simi_samp_per_emit + simi_samp_per_emit)]
        plt.hist(np.max(simi_cls_scores, axis=1), range=(0, 1.05), bins=bins, alpha=1/simi_num_cls, label=str(simi_cls_names[i]))
    plt.title("Similar Set Per-class Score Distributions")
    plt.xlim(hist_xlim)
    plt.ylim(0, simi_arr_scores.shape[0] / simi_num_cls)
    plt.legend(loc='upper left')
    plt.savefig(path_results + '/' + rundate + '/simi_score_hist.png', bbox_inches='tight')
    plt.show()

    # Plot combined same/similar ROC curves
    same_simi_scores = np.concatenate([same_arr_scores, simi_arr_scores], axis=0)
    same_targets = keras.utils.to_categorical(same_set[1], num_classes=num_cls)
    simi_targets = keras.utils.to_categorical(simi_set[1], num_classes=num_cls)
    same_simi_targets = np.concatenate([same_targets, simi_targets], axis=0)

    fpr = {}
    tpr = {}
    roc_auc = {}

    for cls in range(num_cls):
        fpr[cls], tpr[cls], _ = roc_curve(same_simi_targets[:, cls], same_simi_scores[:, cls])
        roc_auc[cls] = auc(fpr[cls], tpr[cls])

    fig = plt.figure(figsize=cm_figsize)
    for i in range(num_cls):
        if i in same_classes and i in simi_classes:
            plt.plot(fpr[i], tpr[i], '-.', lw=2, label='Emitter {0} (AUC={1:0.2f})'''.format(cls_names[i], roc_auc[i]))
        elif i in same_classes:
            plt.plot(fpr[i], tpr[i], '-', lw=2, label='Emitter {0} (AUC={1:0.2f})'''.format(cls_names[i], roc_auc[i]))
        elif i in simi_classes:
            plt.plot(fpr[i], tpr[i], '--', lw=2, label='Emitter {0} (AUC={1:0.2f})'''.format(cls_names[i], roc_auc[i]))
        else:
            raise ValueError("Class not in same or similar set")

    plt.plot([0, 1], [0, 1], 'k:', lw=2)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Per-class ROC Curves')
    plt.legend(loc="lower right")
    plt.savefig(path_results + '/' + rundate + '/roc.png', bbox_inches='tight')
    plt.show()

def train_eval_binary_relevace (model, train_set, test_set, run_epochs, path_results, num_cls, cls_names,target_emitter,rundate,
               val_set=None,
               start_epoch=0,
               verbose=1,
               batch_size=4,
               experiment_notes="",
               show=False,
               prev_hist=None):
    """
    Train and evaluate Keras model
    *_set[0] should have shape ([n samples], [x=time axis], [y=frequency axis]) containing the input spectrograms
    *_set[1] should have shape ([n samples],) containing the corresponding class labels
    If using in a loop, feed return values into start_epoch and prev_hist to keep accurate epoch count

    :param model:               keras.engine.training.Model         Compiled Keras model
    :param train_set:           2-tuple of array-likes              Training set (see above)
    :param test_set:            2-tuple of array-likes              Testing set (see above)
    :param run_epochs:          int                                 Number of epochs to run
    :param path_results:        str                                 Output directory
    :param num_cls:             int                                 Total number of classes
    :param cls_names:           list of str                         List of class names ordered by corresponding class label
    :param target_emitter:      str                                 Positive class for the network
    :param rundate:             date time                           The rundate of the current training session
    :param val_set:             2-tuple of array-likes              Validation set (see above)
    :param start_epoch:         int                                 Current epoch of model before training
    :param verbose:             int                                 Keras verbosity mode (0, 1, or 2)
    :param batch_size:          int
    :param experiment_notes:    str                                 Notes about experiment
    :param show:                bool                                Whether to show plots
    :param prev_hist:           dict of arrays                      Previous interval's training history

    :return count_epoch:        int                                 Current epoch of model after training
    :return curr_hist:          dict of arrays                      Current interval's training history
    """

    # Set up results directory
    if not os.path.exists(path_results + '/' + target_emitter):
        os.mkdir(path_results + '/' + target_emitter)
        os.mkdir(path_results + '/' + target_emitter +'/' + rundate )
        os.mkdir(path_results + '/' + target_emitter +'/' + rundate + '/models/' )
        
    if not os.path.exists(path_results + '/' + target_emitter +'/' + rundate ):
        os.mkdir(path_results + '/' + target_emitter +'/' + rundate )
        os.mkdir(path_results + '/' + target_emitter +'/' + rundate + '/models/' )
    
    save_path = path_results + '/' + target_emitter +'/' + rundate

    # Save model summary
    orig_stdout = sys.stdout
    summ_file = open(save_path + '/models/modelsummary.txt', 'w')
    sys.stdout = summ_file
    print(model.summary())
    sys.stdout = orig_stdout
    summ_file.close()

    # Save experiment notes
    orig_stdout = sys.stdout
    experiment_notes_file = open(save_path + '/models/_experiment_notes.txt', 'w')
    sys.stdout = experiment_notes_file
    print(experiment_notes)
    sys.stdout = orig_stdout
    experiment_notes_file.close()

    # Perform training
    count_epoch = start_epoch + run_epochs
    history = model.fit_generator(
        generator=batch_gen_binary(train_set[0], train_set[1], batch_size),
        steps_per_epoch=train_set[0].shape[0]//batch_size,
        epochs=count_epoch,
        initial_epoch=start_epoch,
        verbose=verbose,
        workers=1,
        validation_data=[val_set[0][...].reshape(val_set[0].shape + (1,)), val_set[1]])

    # Save trained model and training info
    save_name = save_path + '/models/epochs' + str(start_epoch) + '-' + str(count_epoch) + '.h5'
    model.save(save_name)

    if prev_hist is None:
        acc = np.asarray(history.history['acc'])
        val_acc = np.asarray(history.history['val_acc'])
        loss = np.asarray(history.history['loss'])
        val_loss = np.asarray(history.history['val_loss'])
    else:
        acc = np.append(prev_hist['acc'], np.asarray(history.history['acc']))
        val_acc = np.append(prev_hist['val_acc'], np.asarray(history.history['val_acc']))
        loss = np.append(prev_hist['loss'], np.asarray(history.history['loss']))
        val_loss = np.append(prev_hist['val_loss'], np.asarray(history.history['val_loss']))

    curr_hist = {'acc':acc, 'val_acc':val_acc, 'loss':loss, 'val_loss':val_loss}

    np.save(save_path + '/history_acc.npy', acc)
    np.save(save_path + '/history_valacc.npy', val_acc)
    np.save(save_path + '/history_loss.npy', loss)
    np.save(save_path + '/history_valloss.npy', val_loss)

    plt.figure(figsize=(10, 5))
    plt.plot(acc, 'b', label="Training")
    plt.plot(val_acc, 'g', label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc='upper left')
    plt.savefig(save_path + '/accuracy.png', bbox_inches='tight')

    plt.figure(figsize=(10, 5))
    plt.plot(loss, 'b', label="Training")
    plt.plot(val_loss, 'g', label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='upper left')
    plt.savefig(save_path + '/loss.png', bbox_inches='tight')
    if show:
        plt.show()

    # Evaluate on test set
    print("Evaluating model on test set")
    test_loss_acc = model.evaluate(test_set[0][...].reshape(test_set[0].shape + (1,)),
                                   test_set[1], 
                                   batch_size=batch_size,
                                   verbose=verbose)
    print("\nComputing scores on test set")
    scores = model.predict(test_set[0][...].reshape(test_set[0].shape + (1,)),
                           batch_size=batch_size,
                           verbose=verbose)
    arr_scores = np.asarray(scores)
    np.save(save_path + '/scores.npy', arr_scores)

    cm = confusion_matrix(test_set[1], np.argmax(scores, axis=-1))
    plot_cm(cm, cls_names=cls_names, normalize=True,
            title="Loss: "+str(test_loss_acc[0])+" | Acc: "+str(test_loss_acc[1]),
            path_save=save_path + '/cm.png',
            show=show)

    print('\n\n')

    return count_epoch, curr_hist

def train_eval_multi_label (model, train_set, test_set, run_epochs, path_results, num_cls, cls_names,rundate,
               val_set=None,
               start_epoch=0,
               verbose=1,
               batch_size=4,
               experiment_notes="",
               show=False,
               prev_hist=None):
    """
    Train and evaluate Keras model
    *_set[0] should have shape ([n samples], [x=time axis], [y=frequency axis]) containing the input spectrograms
    *_set[1] should have shape ([n samples],) containing the corresponding class labels
    If using in a loop, feed return values into start_epoch and prev_hist to keep accurate epoch count

    :param model:               keras.engine.training.Model         Compiled Keras model
    :param train_set:           2-tuple of array-likes              Training set (see above)
    :param test_set:            2-tuple of array-likes              Testing set (see above)
    :param path_results         str                                 Output directory
    :param num_cls              int                                 Total number of classes
    :param cls_names            list of str                         List of class names ordered by corresponding class label
    :param run_epochs:          int                                 Number of epochs to run
    :param val_set:             2-tuple of array-likes              Validation set (see above)
    :param start_epoch:         int                                 Current epoch of model before training
    :param verbose:             int                                 Keras verbosity mode (0, 1, or 2)
    :param batch_size:          int
    :param experiment_notes:    str                                 Notes about experiment
    :param show:                bool                                Whether to show plots
    :param prev_hist:           dict of arrays                      Previous interval's training history

    :return count_epoch:        int                                 Current epoch of model after training
    :return curr_hist:          dict of arrays                      Current interval's training history
    """

    # Set up results directory
    if not os.path.exists(path_results + '/' + rundate):
        os.mkdir(path_results + '/' + rundate )
        os.mkdir(path_results + '/' + rundate + '/models' )
        
    
    save_path = path_results + '/' + rundate

    # Save model summary
    orig_stdout = sys.stdout
    summ_file = open(save_path + '/models/modelsummary.txt', 'w')
    sys.stdout = summ_file
    print(model.summary())
    sys.stdout = orig_stdout
    summ_file.close()

    # Save experiment notes
    orig_stdout = sys.stdout
    experiment_notes_file = open(save_path + '/models/_experiment_notes.txt', 'w')
    sys.stdout = experiment_notes_file
    print(experiment_notes)
    sys.stdout = orig_stdout
    experiment_notes_file.close()

    # Perform training
    count_epoch = start_epoch + run_epochs
    history = model.fit_generator(
        generator=batch_gen_binary(train_set[0], train_set[1], batch_size),
        steps_per_epoch=train_set[0].shape[0]//batch_size,
        epochs=count_epoch,
        initial_epoch=start_epoch,
        verbose=verbose,
        workers=1,
        validation_data=[val_set[0][...].reshape(val_set[0].shape + (1,)), val_set[1]])

    # Save trained model and training info
    save_name = save_path + '/models/epochs' + str(start_epoch) + '-' + str(count_epoch) + '.h5'
    model.save(save_name)

    if prev_hist is None:
        acc = np.asarray(history.history['acc'])
        val_acc = np.asarray(history.history['val_acc'])
        loss = np.asarray(history.history['loss'])
        val_loss = np.asarray(history.history['val_loss'])
    else:
        acc = np.append(prev_hist['acc'], np.asarray(history.history['acc']))
        val_acc = np.append(prev_hist['val_acc'], np.asarray(history.history['val_acc']))
        loss = np.append(prev_hist['loss'], np.asarray(history.history['loss']))
        val_loss = np.append(prev_hist['val_loss'], np.asarray(history.history['val_loss']))

    curr_hist = {'acc':acc, 'val_acc':val_acc, 'loss':loss, 'val_loss':val_loss}

    np.save(save_path + '/history_acc.npy', acc)
    np.save(save_path + '/history_valacc.npy', val_acc)
    np.save(save_path + '/history_loss.npy', loss)
    np.save(save_path + '/history_valloss.npy', val_loss)

    plt.figure(figsize=(10, 5))
    plt.plot(acc, 'b', label="Training")
    plt.plot(val_acc, 'g', label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc='upper left')
    plt.savefig(save_path + '/accuracy.png', bbox_inches='tight')

    plt.figure(figsize=(10, 5))
    plt.plot(loss, 'b', label="Training")
    plt.plot(val_loss, 'g', label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='upper left')
    plt.savefig(save_path + '/loss.png', bbox_inches='tight')
    if show:
        plt.show()

    # Evaluate on test set
    print("Evaluating model on test set")
    test_loss_acc = model.evaluate(test_set[0][...].reshape(test_set[0].shape + (1,)),
                                   test_set[1], 
                                   batch_size=batch_size,
                                   verbose=verbose)
    print("\nComputing scores on test set")
    scores = model.predict(test_set[0][...].reshape(test_set[0].shape + (1,)),
                           batch_size=batch_size,
                           verbose=verbose)
    arr_scores = np.asarray(scores)
    np.save(save_path + '/scores.npy', arr_scores)

    cm = confusion_matrix(test_set[1], np.argmax(scores, axis=-1))
    plot_cm(cm, cls_names=cls_names, normalize=True,
            title="Loss: "+str(test_loss_acc[0])+" | Acc: "+str(test_loss_acc[1]),
            path_save=save_path + '/cm.png',
            show=show)

    print('\n\n')

    return count_epoch, curr_hist

            
