"""
Functions for organizing data in HDF5 files into training/validation/testing datasets
"""

__author__ = 'clee'
# Python 3.6.1
"""
Changed: 02/12/2019 by C.Snellen
Added form_binary_relevance_datasets which creates a binary dataset with a target emitter
as the positive value and all other emitters as the negative value
"""
# Standard lib
import os
import random

# Additional lib
import h5py
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

def form_even_datasets(path_sg, path_meta, path_datasets, samps_per_class, train_val_test,
                       seed=256,
                       shuffle=False,
                       fname_ds_sg='/sg_datasets.h5',
                       fname_ds_meta='/meta_datasets.h5'):

    """
    Randomly form even-class training, validation, and testing datasets; assign class labels (0 to [# of classes]-1)

    :param path_sg:                 str                 Path to spectrogram HDF5 file
    :param path_meta:               str                 Path to metadata HDF5 file
    :param path_datasets:           str                 Path to dataset output directory
    :param samps_per_class:         int                 Number of samples per class
    :param train_val_test:          3-tuple of ints     ([# of train samples], [# of val samples], [# of test samples])
                                                            (Per class)
    :param seed:                    int                 Seed for random operations
    :param shuffle                  bool                Whether or not to shuffle datasets after creation
    :param fname_ds_sg              str                 Name of spectrograms dataset file
    :param fname_ds_meta            str                 Name of metadata datasets file

    :return:                        2-tuple of str      Paths of dataset-organized spectrogram and metadata files
    """

    if sum(train_val_test) != samps_per_class:
        raise ValueError("Set values do not add up")

    hdf5_sg = h5py.File(path_sg, 'r')
    hdf5_meta = h5py.File(path_meta, 'r')
    dataset_sg = hdf5_sg['sg']
    dataset_emitter = hdf5_meta['emitter']
    dataset_file = hdf5_meta['file']
    list_ds_emit = list(dataset_emitter)
    str_dt = h5py.special_dtype(vlen=str)
    shape_sg = dataset_sg[0].shape
    
    # Set up hdf5 files and datasets for training/val/testing
    if os.path.isfile(path_datasets + fname_ds_sg):
        os.remove(path_datasets + fname_ds_sg)
    if os.path.isfile(path_datasets + fname_ds_meta):
        os.remove(path_datasets + fname_ds_meta)
    
    hdf5_ds_sg = h5py.File(path_datasets + fname_ds_sg, 'w')
    hdf5_ds_meta = h5py.File(path_datasets + fname_ds_meta, 'w')
    
    ds_train_sg = hdf5_ds_sg.create_dataset('train_sg', shape=(1, shape_sg[0], shape_sg[1]), maxshape=(None, shape_sg[0], shape_sg[1]))
    ds_train_emit = hdf5_ds_meta.create_dataset('train_emitter', shape=(1, ), maxshape=(None, ), dtype='int16')
    ds_train_cls = hdf5_ds_meta.create_dataset('train_class', shape=(1, ), maxshape=(None, ), dtype='int8')
    ds_train_file = hdf5_ds_meta.create_dataset('train_file', shape=(1, ), maxshape=(None, ), dtype=str_dt)
    
    ds_val_sg = hdf5_ds_sg.create_dataset('val_sg', shape=(1, shape_sg[0], shape_sg[1]), maxshape=(None, shape_sg[0], shape_sg[1]))
    ds_val_emit = hdf5_ds_meta.create_dataset('val_emitter', shape=(1, ), maxshape=(None, ), dtype='int16')
    ds_val_cls = hdf5_ds_meta.create_dataset('val_class', shape=(1, ), maxshape=(None, ), dtype='int8')
    ds_val_file = hdf5_ds_meta.create_dataset('val_file', shape=(1, ), maxshape=(None, ), dtype=str_dt)
    
    ds_test_sg = hdf5_ds_sg.create_dataset('test_sg', shape=(1, shape_sg[0], shape_sg[1]), maxshape=(None, shape_sg[0], shape_sg[1]))
    ds_test_emit = hdf5_ds_meta.create_dataset('test_emitter', shape=(1, ), maxshape=(None, ), dtype='int16')
    ds_test_cls = hdf5_ds_meta.create_dataset('test_class', shape=(1, ), maxshape=(None, ), dtype='int8')
    ds_test_file = hdf5_ds_meta.create_dataset('test_file', shape=(1, ), maxshape=(None, ), dtype=str_dt)


    trainset_ind = 0
    valset_ind = 0
    testset_ind = 0
    cls_label = 0
    for emitter in sorted(np.unique(list_ds_emit)):
        # Get index range of emitter
        first_ind = list_ds_emit.index(emitter)
        last_ind = len(list_ds_emit) - 1 - list_ds_emit[::-1].index(emitter)
        if (last_ind - first_ind + 2) < samps_per_class:
            raise ValueError("Emitter " + str(emitter) + " has too few samples")
        emitter_inds = set(range(first_ind, last_ind+1))
        
        # Randomly sample for each set's indices
        train_inds = sorted(random.Random(seed).sample(emitter_inds, train_val_test[0]))
        val_inds = sorted(random.Random(seed).sample(emitter_inds-set(train_inds), train_val_test[1]))
        test_inds = sorted(random.Random(seed).sample(emitter_inds-set(train_inds)-set(val_inds), train_val_test[2]))
        
        # Add emitter samples to each set
        ds_train_sg.resize(trainset_ind+train_val_test[0], axis=0)
        ds_train_emit.resize(trainset_ind+train_val_test[0], axis=0)
        ds_train_cls.resize(trainset_ind+train_val_test[0], axis=0)
        ds_train_file.resize(trainset_ind+train_val_test[0], axis=0)
        
        ds_train_sg[trainset_ind:trainset_ind+train_val_test[0]] = dataset_sg[train_inds]
        ds_train_emit[trainset_ind:trainset_ind+train_val_test[0]] = dataset_emitter[train_inds]
        ds_train_cls[trainset_ind:trainset_ind+train_val_test[0]] = np.int8(cls_label)
        ds_train_file[trainset_ind:trainset_ind+train_val_test[0]] = dataset_file[train_inds]
        trainset_ind += train_val_test[0]
        
        
        ds_val_sg.resize(valset_ind+train_val_test[1], axis=0)
        ds_val_emit.resize(valset_ind+train_val_test[1], axis=0)
        ds_val_cls.resize(valset_ind+train_val_test[1], axis=0)
        ds_val_file.resize(valset_ind+train_val_test[1], axis=0)
        
        ds_val_sg[valset_ind:valset_ind+train_val_test[1]] = dataset_sg[val_inds]
        ds_val_emit[valset_ind:valset_ind+train_val_test[1]] = dataset_emitter[val_inds]
        ds_val_cls[valset_ind:valset_ind+train_val_test[1]] = np.int8(cls_label)
        ds_val_file[valset_ind:valset_ind+train_val_test[1]] = dataset_file[val_inds]
        valset_ind += train_val_test[1]
        
        
        ds_test_sg.resize(testset_ind+train_val_test[2], axis=0)
        ds_test_emit.resize(testset_ind+train_val_test[2], axis=0)
        ds_test_cls.resize(testset_ind+train_val_test[2], axis=0)
        ds_test_file.resize(testset_ind+train_val_test[2], axis=0)
        
        ds_test_sg[testset_ind:testset_ind+train_val_test[2]] = dataset_sg[test_inds]
        ds_test_emit[testset_ind:testset_ind+train_val_test[2]] = dataset_emitter[test_inds]
        ds_test_cls[testset_ind:testset_ind+train_val_test[2]] = np.int8(cls_label)
        ds_test_file[testset_ind:testset_ind+train_val_test[2]] = dataset_file[test_inds]
        testset_ind += train_val_test[2]
        
        cls_label += 1

    if shuffle:
        # Rely on seed to maintain correct row integrity
        random.Random(seed).shuffle(ds_train_sg)
        random.Random(seed).shuffle(ds_train_emit)
        random.Random(seed).shuffle(ds_train_cls)
        random.Random(seed).shuffle(ds_train_file)

        random.Random(seed).shuffle(ds_val_sg)
        random.Random(seed).shuffle(ds_val_emit)
        random.Random(seed).shuffle(ds_val_cls)
        random.Random(seed).shuffle(ds_val_file)

        random.Random(seed).shuffle(ds_test_sg)
        random.Random(seed).shuffle(ds_test_emit)
        random.Random(seed).shuffle(ds_test_cls)
        random.Random(seed).shuffle(ds_test_file)
    
    hdf5_ds_sg.close()
    hdf5_ds_meta.close()

    return path_datasets+fname_ds_sg, path_datasets+fname_ds_meta


def form_close_emitter_datasets(path_sg, path_meta, path_datasets, samps_per_class, train_val, train_emits, same_emits, simi_emits,
                                seed=256,
                                fname_ds_sg='/sg_datasets.h5',
                                fname_ds_meta='/meta_datasets.h5'):

    """
    Randomly form even class datasets with training and validation from the same set of emitters
    "Same" and "similar" testing datasets chosen from different sets of emitters

    :param path_sg:                 str                 Path to spectrogram HDF5 file
    :param path_meta:               str                 Path to metadata HDF5 file
    :param path_datasets:           str                 Path to dataset output directory
    :param samps_per_class:         int                 Number of samples per class
    :param train_val:               2-tuple of ints     ([# of train samples], [# of val samples])
                                                            (Per class)
    :param train_emits:             list of ints        Emitters to put in the training and validation sets
    :param same_emits:              dict of ints        Emitters to put in the "same" test set
                                                            With comparable training emitters as values
    :param simi_emits:           dict of ints        Emitters to put in the "similar" test set
                                                            With comparable training emitters as values
    :param seed:                    int                 Seed for random operations
    :param fname_ds_sg              str                 Name of spectrograms dataset file
    :param fname_ds_meta            str                 Name of metadata datasets file

    :return:                        2-tuple of str      Paths of dataset-organized spectrogram and metadata files
    """

    if sum(train_val) != samps_per_class:
        raise ValueError("Set values do not add up")

    hdf5_sg = h5py.File(path_sg, 'r')
    hdf5_meta = h5py.File(path_meta, 'r')
    dataset_sg = hdf5_sg['sg']
    dataset_emitter = hdf5_meta['emitter']
    list_ds_emit = list(dataset_emitter)
    dataset_file = hdf5_meta['file']
    str_dt = h5py.special_dtype(vlen=str)
    shape_sg = dataset_sg[0].shape

    # Set up hdf5 files and datasets for training/val/testing
    if os.path.isfile(path_datasets + fname_ds_sg):
        os.remove(path_datasets + fname_ds_sg)
    if os.path.isfile(path_datasets + fname_ds_meta):
        os.remove(path_datasets + fname_ds_meta)

    hdf5_ds_sg = h5py.File(path_datasets + fname_ds_sg, 'w')
    hdf5_ds_meta = h5py.File(path_datasets + fname_ds_meta, 'w')

    ds_train_sg = hdf5_ds_sg.create_dataset('train_sg', shape=(1, shape_sg[0], shape_sg[1]), maxshape=(None, shape_sg[0], shape_sg[1]))
    ds_train_emit = hdf5_ds_meta.create_dataset('train_emitter', shape=(1,), maxshape=(None,), dtype='int16')
    ds_train_cls = hdf5_ds_meta.create_dataset('train_class', shape=(1,), maxshape=(None,), dtype='int8')
    ds_train_file = hdf5_ds_meta.create_dataset('train_file', shape=(1,), maxshape=(None,), dtype=str_dt)

    ds_val_sg = hdf5_ds_sg.create_dataset('val_sg', shape=(1, shape_sg[0], shape_sg[1]), maxshape=(None, shape_sg[0], shape_sg[1]))
    ds_val_emit = hdf5_ds_meta.create_dataset('val_emitter', shape=(1,), maxshape=(None,), dtype='int16')
    ds_val_cls = hdf5_ds_meta.create_dataset('val_class', shape=(1,), maxshape=(None,), dtype='int8')
    ds_val_file = hdf5_ds_meta.create_dataset('val_file', shape=(1,), maxshape=(None,), dtype=str_dt)

    ds_same_sg = hdf5_ds_sg.create_dataset('same_sg', shape=(1, shape_sg[0], shape_sg[1]), maxshape=(None, shape_sg[0], shape_sg[1]))
    ds_same_emit = hdf5_ds_meta.create_dataset('same_emitter', shape=(1,), maxshape=(None,), dtype='int16')
    ds_same_comp = hdf5_ds_meta.create_dataset('same_comp', shape=(1,), maxshape=(None,), dtype='int16')
    ds_same_cls = hdf5_ds_meta.create_dataset('same_class', shape=(1,), maxshape=(None,), dtype='int8')
    ds_same_file = hdf5_ds_meta.create_dataset('same_file', shape=(1,), maxshape=(None,), dtype=str_dt)

    ds_simi_sg = hdf5_ds_sg.create_dataset('simi_sg', shape=(1, shape_sg[0], shape_sg[1]), maxshape=(None, shape_sg[0], shape_sg[1]))
    ds_simi_emit = hdf5_ds_meta.create_dataset('simi_emitter', shape=(1,), maxshape=(None,), dtype='int16')
    ds_simi_comp = hdf5_ds_meta.create_dataset('simi_comp', shape=(1,), maxshape=(None,), dtype='int16')
    ds_simi_cls = hdf5_ds_meta.create_dataset('simi_class', shape=(1,), maxshape=(None,), dtype='int8')
    ds_simi_file = hdf5_ds_meta.create_dataset('simi_file', shape=(1,), maxshape=(None,), dtype=str_dt)

    # Form training and validation sets
    trainset_ind = 0
    valset_ind = 0
    cls_label = 0
    for emitter in sorted(np.unique(train_emits)):
        # Get index range of emitter
        first_ind = list_ds_emit.index(emitter)
        last_ind = len(list_ds_emit) - 1 - list_ds_emit[::-1].index(emitter)
        if (last_ind - first_ind + 2) < samps_per_class:
            raise ValueError("Emitter " + str(emitter) + " has too few samples")

        emitter_inds = set(range(first_ind, last_ind + 1))

        # Randomly sample for each set's indices
        train_inds = sorted(random.Random(seed).sample(emitter_inds, train_val[0]))
        val_inds = sorted(random.Random(seed).sample(emitter_inds - set(train_inds), train_val[1]))

        # Add emitter samples to each set
        ds_train_sg.resize(trainset_ind + train_val[0], axis=0)
        ds_train_emit.resize(trainset_ind + train_val[0], axis=0)
        ds_train_cls.resize(trainset_ind + train_val[0], axis=0)
        ds_train_file.resize(trainset_ind + train_val[0], axis=0)

        ds_train_sg[trainset_ind:trainset_ind + train_val[0]] = dataset_sg[train_inds]
        ds_train_emit[trainset_ind:trainset_ind + train_val[0]] = dataset_emitter[train_inds]
        ds_train_cls[trainset_ind:trainset_ind + train_val[0]] = np.int8(cls_label)
        ds_train_file[trainset_ind:trainset_ind + train_val[0]] = dataset_file[train_inds]
        trainset_ind += train_val[0]

        ds_val_sg.resize(valset_ind + train_val[1], axis=0)
        ds_val_emit.resize(valset_ind + train_val[1], axis=0)
        ds_val_cls.resize(valset_ind + train_val[1], axis=0)
        ds_val_file.resize(valset_ind + train_val[1], axis=0)

        ds_val_sg[valset_ind:valset_ind + train_val[1]] = dataset_sg[val_inds]
        ds_val_emit[valset_ind:valset_ind + train_val[1]] = dataset_emitter[val_inds]
        ds_val_cls[valset_ind:valset_ind + train_val[1]] = np.int8(cls_label)
        ds_val_file[valset_ind:valset_ind + train_val[1]] = dataset_file[val_inds]
        valset_ind += train_val[1]

        cls_label += 1
        
    sameset_ind = 0
    for emitter in sorted(np.unique(list(same_emits.keys()))):
        # Get index range of emitter
        first_ind = list_ds_emit.index(emitter)
        last_ind = len(list_ds_emit) - 1 - list_ds_emit[::-1].index(emitter)
        if (last_ind - first_ind + 2) < samps_per_class:
            raise ValueError("Emitter " + str(emitter) + " has too few samples")
        emitter_inds = set(range(first_ind, last_ind + 1))

        # Randomly sample for each set's indices
        same_inds = sorted(random.Random(seed).sample(emitter_inds, samps_per_class))

        # Add emitter samples to each set
        ds_same_sg.resize(sameset_ind + samps_per_class, axis=0)
        ds_same_emit.resize(sameset_ind + samps_per_class, axis=0)
        ds_same_comp.resize(sameset_ind + samps_per_class, axis=0)
        ds_same_cls.resize(sameset_ind + samps_per_class, axis=0)
        ds_same_file.resize(sameset_ind + samps_per_class, axis=0)

        ds_same_sg[sameset_ind:sameset_ind + samps_per_class] = dataset_sg[same_inds]
        ds_same_emit[sameset_ind:sameset_ind + samps_per_class] = dataset_emitter[same_inds]
        ds_same_comp[sameset_ind:sameset_ind + samps_per_class] = np.int16(same_emits[emitter])
        comp_class = sorted(np.unique(train_emits)).index(same_emits[emitter])
        ds_same_cls[sameset_ind:sameset_ind + samps_per_class] = np.int8(comp_class)
        ds_same_file[sameset_ind:sameset_ind + samps_per_class] = dataset_file[same_inds]
        sameset_ind += samps_per_class

    simiset_ind = 0
    for emitter in sorted(np.unique(list(simi_emits.keys()))):
        # Get index range of emitter
        first_ind = list_ds_emit.index(emitter)
        last_ind = len(list_ds_emit) - 1 - list_ds_emit[::-1].index(emitter)
        if (last_ind - first_ind + 2) < samps_per_class:
            raise ValueError("Emitter " + str(emitter) + " has too few samples")
        emitter_inds = set(range(first_ind, last_ind + 1))

        # Randomly sample for each set's indices
        simi_inds = sorted(random.Random(seed).sample(emitter_inds, samps_per_class))

        # Add emitter samples to each set
        ds_simi_sg.resize(simiset_ind + samps_per_class, axis=0)
        ds_simi_emit.resize(simiset_ind + samps_per_class, axis=0)
        ds_simi_comp.resize(simiset_ind + samps_per_class, axis=0)
        ds_simi_cls.resize(simiset_ind + samps_per_class, axis=0)
        ds_simi_file.resize(simiset_ind + samps_per_class, axis=0)

        ds_simi_sg[simiset_ind:simiset_ind + samps_per_class] = dataset_sg[simi_inds]
        ds_simi_emit[simiset_ind:simiset_ind + samps_per_class] = dataset_emitter[simi_inds]
        ds_simi_comp[simiset_ind:simiset_ind + samps_per_class] = np.int16(simi_emits[emitter])
        comp_class = sorted(np.unique(train_emits)).index(simi_emits[emitter])
        ds_simi_cls[simiset_ind:simiset_ind + samps_per_class] = np.int8(comp_class)
        ds_simi_file[simiset_ind:simiset_ind + samps_per_class] = dataset_file[simi_inds]
        simiset_ind += samps_per_class

    hdf5_ds_sg.close()
    hdf5_ds_meta.close()

    return path_datasets + fname_ds_sg, path_datasets + fname_ds_meta

def form_binary_relevance_dataset(path_sg, path_meta, path_datasets, samps_per_class, target_train_val_test, other_train_val_test, target_set,other_set,
                       seed=256,
                       shuffle=False,
                       fname_ds_sg= '/sg_datasets.h5',
                       fname_ds_meta= '/meta_datasets.h5'):

    """
    Randomly form even-class training, validation, and testing datasets; assign class labels (0 to [# of classes]-1)

    :param path_sg:                 str                 Path to spectrogram HDF5 file
    :param path_meta:               str                 Path to metadata HDF5 file
    :param path_datasets:           str                 Path to dataset output directory
    :param samps_per_class:         int                 Number of samples per class
    :param train_val_test:          3-tuple of ints     ([# of train samples], [# of val samples], [# of test samples])
                                                            (Per class)
    :param seed:                    int                 Seed for random operations
    :param shuffle                  bool                Whether or not to shuffle datasets after creation
    :param fname_ds_sg              str                 Name of spectrograms dataset file
    :param fname_ds_meta            str                 Name of metadata datasets file

    :return:                        2-tuple of str      Paths of dataset-organized spectrogram and metadata files
    """
    
     

   # if sum(target_train_val_test) != samps_per_class:
   #    raise ValueError("Set values do not add up")

    hdf5_sg = h5py.File(path_sg, 'r')
    hdf5_meta = h5py.File(path_meta, 'r')
    dataset_sg = hdf5_sg['sg']
    dataset_emitter = hdf5_meta['emitter']
    dataset_file = hdf5_meta['file']
    list_ds_emit = list(dataset_emitter)
    str_dt = h5py.special_dtype(vlen=str)
    shape_sg = dataset_sg[0].shape
    
    # Set up hdf5 files and datasets for training/val/testing
    if os.path.isfile(path_datasets + fname_ds_sg):
        os.remove(path_datasets + fname_ds_sg)
    if os.path.isfile(path_datasets + fname_ds_meta):
        os.remove(path_datasets + fname_ds_meta)
    
    hdf5_ds_sg = h5py.File(path_datasets + fname_ds_sg, 'w')
    hdf5_ds_meta = h5py.File(path_datasets + fname_ds_meta, 'w')
    
    ds_train_sg = hdf5_ds_sg.create_dataset('train_sg', shape=(1, shape_sg[0], shape_sg[1]), maxshape=(None, shape_sg[0], shape_sg[1]))
    ds_train_emit = hdf5_ds_meta.create_dataset('train_emitter', shape=(1, ), maxshape=(None, ), dtype=str_dt)
    ds_train_cls = hdf5_ds_meta.create_dataset('train_class', shape=(1, ), maxshape=(None, ), dtype='int8')
    ds_train_file = hdf5_ds_meta.create_dataset('train_file', shape=(1, ), maxshape=(None, ), dtype=str_dt)
    
    ds_val_sg = hdf5_ds_sg.create_dataset('val_sg', shape=(1, shape_sg[0], shape_sg[1]), maxshape=(None, shape_sg[0], shape_sg[1]))
    ds_val_emit = hdf5_ds_meta.create_dataset('val_emitter', shape=(1, ), maxshape=(None, ), dtype=str_dt)
    ds_val_cls = hdf5_ds_meta.create_dataset('val_class', shape=(1, ), maxshape=(None, ), dtype='int8')
    ds_val_file = hdf5_ds_meta.create_dataset('val_file', shape=(1, ), maxshape=(None, ), dtype=str_dt)
    
    ds_test_sg = hdf5_ds_sg.create_dataset('test_sg', shape=(1, shape_sg[0], shape_sg[1]), maxshape=(None, shape_sg[0], shape_sg[1]))
    ds_test_emit = hdf5_ds_meta.create_dataset('test_emitter', shape=(1, ), maxshape=(None, ), dtype=str_dt)
    ds_test_cls = hdf5_ds_meta.create_dataset('test_class', shape=(1, ), maxshape=(None, ), dtype='int8')
    ds_test_file = hdf5_ds_meta.create_dataset('test_file', shape=(1, ), maxshape=(None, ), dtype=str_dt)


    trainset_ind = 0
    valset_ind = 0
    testset_ind = 0
    for emitter in sorted(np.unique(list_ds_emit)):
        # Get index range of emitter
        first_ind = list_ds_emit.index(emitter)
        last_ind = len(list_ds_emit) - 1 - list_ds_emit[::-1].index(emitter)
        if (last_ind - first_ind + 2) < samps_per_class:
            raise ValueError("Emitter " + str(emitter) + " has too few samples")
        emitter_inds = set(range(first_ind, last_ind+1))
        if emitter in target_set:
            # Randomly sample for each set's indices
            train_inds = sorted(random.Random(seed).sample(emitter_inds, target_train_val_test[0]))
            val_inds = sorted(random.Random(seed).sample(emitter_inds-set(train_inds), target_train_val_test[1]))
            test_inds = sorted(random.Random(seed).sample(emitter_inds-set(train_inds)-set(val_inds), target_train_val_test[2]))
            cls_label = 1;
            # Add emitter samples to each set
            ds_train_sg.resize(trainset_ind + target_train_val_test[0], axis=0)
            ds_train_emit.resize(trainset_ind + target_train_val_test[0], axis=0)
            ds_train_cls.resize(trainset_ind + target_train_val_test[0], axis=0)
            ds_train_file.resize(trainset_ind + target_train_val_test[0], axis=0)
        
            ds_train_sg[trainset_ind : trainset_ind + target_train_val_test[0]] = dataset_sg[train_inds]
            ds_train_emit[trainset_ind : trainset_ind + target_train_val_test[0]] = dataset_emitter[train_inds]
            ds_train_cls[trainset_ind : trainset_ind + target_train_val_test[0]] = np.int8(cls_label)
            ds_train_file[trainset_ind : trainset_ind + target_train_val_test[0]] = dataset_file[train_inds]
            trainset_ind += target_train_val_test[0]
        
        
            ds_val_sg.resize(valset_ind + target_train_val_test[1], axis=0)
            ds_val_emit.resize(valset_ind + target_train_val_test[1], axis=0)
            ds_val_cls.resize(valset_ind + target_train_val_test[1], axis=0)
            ds_val_file.resize(valset_ind+ target_train_val_test[1], axis=0)
        
            ds_val_sg[valset_ind : valset_ind + target_train_val_test[1]] = dataset_sg[val_inds]
            ds_val_emit[valset_ind : valset_ind + target_train_val_test[1]] = dataset_emitter[val_inds]
            ds_val_cls[valset_ind : valset_ind + target_train_val_test[1]] = np.int8(cls_label)
            ds_val_file[valset_ind : valset_ind + target_train_val_test[1]] = dataset_file[val_inds]
            valset_ind += target_train_val_test[1]
        
        
            ds_test_sg.resize(testset_ind + target_train_val_test[2], axis=0)
            ds_test_emit.resize(testset_ind + target_train_val_test[2], axis=0)
            ds_test_cls.resize(testset_ind + target_train_val_test[2], axis=0)
            ds_test_file.resize(testset_ind + target_train_val_test[2], axis=0)
        
            ds_test_sg[testset_ind : testset_ind + target_train_val_test[2]] = dataset_sg[test_inds]
            ds_test_emit[testset_ind : testset_ind + target_train_val_test[2]] = dataset_emitter[test_inds]
            ds_test_cls[testset_ind : testset_ind + target_train_val_test[2]] = np.int8(cls_label)
            ds_test_file[testset_ind : testset_ind + target_train_val_test[2]] = dataset_file[test_inds]
            testset_ind += target_train_val_test[2]
        elif emitter in other_set: 
            train_inds = sorted(random.Random(seed).sample(emitter_inds, other_train_val_test[0]))
            val_inds = sorted(random.Random(seed).sample(emitter_inds-set(train_inds), other_train_val_test[1]))
            test_inds = sorted(random.Random(seed).sample(emitter_inds-set(train_inds)-set(val_inds), other_train_val_test[2]))
            cls_label = 0;
            
            # Add emitter samples to each set
            ds_train_sg.resize(trainset_ind + other_train_val_test[0], axis=0)
            ds_train_emit.resize(trainset_ind + other_train_val_test[0], axis=0)
            ds_train_cls.resize(trainset_ind + other_train_val_test[0], axis=0)
            ds_train_file.resize(trainset_ind + other_train_val_test[0], axis=0)
        
            ds_train_sg[trainset_ind : trainset_ind + other_train_val_test[0]] = dataset_sg[train_inds]
            ds_train_emit[trainset_ind : trainset_ind + other_train_val_test[0]] = dataset_emitter[train_inds]
            ds_train_cls[trainset_ind : trainset_ind + other_train_val_test[0]] = np.int8(cls_label)
            ds_train_file[trainset_ind : trainset_ind + other_train_val_test[0]] = dataset_file[train_inds]
            trainset_ind += other_train_val_test[0]
        
        
            ds_val_sg.resize(valset_ind + other_train_val_test[1], axis=0)
            ds_val_emit.resize(valset_ind + other_train_val_test[1], axis=0)
            ds_val_cls.resize(valset_ind + other_train_val_test[1], axis=0)
            ds_val_file.resize(valset_ind+ other_train_val_test[1], axis=0)
        
            ds_val_sg[valset_ind : valset_ind + other_train_val_test[1]] = dataset_sg[val_inds]
            ds_val_emit[valset_ind : valset_ind + other_train_val_test[1]] = dataset_emitter[val_inds]
            ds_val_cls[valset_ind : valset_ind + other_train_val_test[1]] = np.int8(cls_label)
            ds_val_file[valset_ind : valset_ind + other_train_val_test[1]] = dataset_file[val_inds]
            valset_ind += other_train_val_test[1]
        
        
            ds_test_sg.resize(testset_ind + other_train_val_test[2], axis=0)
            ds_test_emit.resize(testset_ind + other_train_val_test[2], axis=0)
            ds_test_cls.resize(testset_ind + other_train_val_test[2], axis=0)
            ds_test_file.resize(testset_ind + other_train_val_test[2], axis=0)
        
            ds_test_sg[testset_ind : testset_ind + other_train_val_test[2]] = dataset_sg[test_inds]
            ds_test_emit[testset_ind : testset_ind + other_train_val_test[2]] = dataset_emitter[test_inds]
            ds_test_cls[testset_ind : testset_ind + other_train_val_test[2]] = np.int8(cls_label)
            ds_test_file[testset_ind : testset_ind + other_train_val_test[2]] = dataset_file[test_inds]
            testset_ind += other_train_val_test[2]
        else:
            print("Emitter combination %s not used" % (emitter))
      
        

    if shuffle:
        # Rely on seed to maintain correct row integrity
        random.Random(seed).shuffle(ds_train_sg)
        random.Random(seed).shuffle(ds_train_emit)
        random.Random(seed).shuffle(ds_train_cls)
        random.Random(seed).shuffle(ds_train_file)

        random.Random(seed).shuffle(ds_val_sg)
        random.Random(seed).shuffle(ds_val_emit)
        random.Random(seed).shuffle(ds_val_cls)
        random.Random(seed).shuffle(ds_val_file)

        random.Random(seed).shuffle(ds_test_sg)
        random.Random(seed).shuffle(ds_test_emit)
        random.Random(seed).shuffle(ds_test_cls)
        random.Random(seed).shuffle(ds_test_file)
    
    hdf5_ds_sg.close()
    hdf5_ds_meta.close()

    return path_datasets+fname_ds_sg, path_datasets+fname_ds_meta

def form_multi_label_datasets(path_sg, path_meta, path_datasets,path_data, samps_per_class, train_val_test, test_specs, num_classes,
                       seed=256,
                       shuffle=False,
                       fname_ds_sg= '/sg_datasets.h5',
                       fname_ds_meta= '/meta_datasets.h5'):

    """
    Randomly form even-class training, validation, and testing datasets; assign class labels (0 to [# of classes]-1)

    :param path_sg:                 str                 Path to spectrogram HDF5 file
    :param path_meta:               str                 Path to metadata HDF5 file
    :param path_datasets:           str                 Path to dataset output directory
    :param samps_per_class:         int                 Number of samples per class
    :param train_val_test:          3-tuple of ints     ([# of train samples], [# of val samples], [# of test samples])
                                                            (Per class)
    :param seed:                    int                 Seed for random operations
    :param shuffle                  bool                Whether or not to shuffle datasets after creation
    :param fname_ds_sg              str                 Name of spectrograms dataset file
    :param fname_ds_meta            str                 Name of metadata datasets file

    :return:                        2-tuple of str      Paths of dataset-organized spectrogram and metadata files
    """
    
     

   # if sum(target_train_val_test) != samps_per_class:
   #    raise ValueError("Set values do not add up")
    # Fit a multi label binarizer for the labels 
    labels =[]
    for root, subdirs, files in os.walk(path_data):
        subdirs.sort()
        files.sort()

        # Check for the right directory
        if (root != path_data and files != []):

            for file in files:
                if file.split('.')[-1] == 'mat':
                    fullLabel = root.split('/')[-1]
                    label = fullLabel.split("_")
                    labels.append(label)
                    
    hdf5_sg = h5py.File(path_sg, 'r')
    hdf5_meta = h5py.File(path_meta, 'r')
    dataset_sg = hdf5_sg['sg']
    dataset_emitter = hdf5_meta['emitter']
    dataset_file = hdf5_meta['file']
    list_ds_emit = list(dataset_emitter)
    str_dt = h5py.special_dtype(vlen=str)
    shape_sg = dataset_sg[0].shape
    
    # Set up hdf5 files and datasets for training/val/testing
    if os.path.isfile(path_datasets + fname_ds_sg):
        os.remove(path_datasets + fname_ds_sg)
    if os.path.isfile(path_datasets + fname_ds_meta):
        os.remove(path_datasets + fname_ds_meta)
    
    hdf5_ds_sg = h5py.File(path_datasets + fname_ds_sg, 'w')
    hdf5_ds_meta = h5py.File(path_datasets + fname_ds_meta, 'w')
    
    ds_train_sg = hdf5_ds_sg.create_dataset('train_sg', shape=(1, shape_sg[0], shape_sg[1]), maxshape=(None, shape_sg[0], shape_sg[1]))
    ds_train_emit = hdf5_ds_meta.create_dataset('train_emitter', shape=(1, ), maxshape=(None, ), dtype=str_dt)
    ds_train_cls = hdf5_ds_meta.create_dataset('train_class', shape=(1,num_classes,), maxshape=(None,num_classes,), dtype='int8')
    ds_train_file = hdf5_ds_meta.create_dataset('train_file', shape=(1, ), maxshape=(None, ), dtype=str_dt)
    
    ds_val_sg = hdf5_ds_sg.create_dataset('val_sg', shape=(1, shape_sg[0], shape_sg[1]), maxshape=(None, shape_sg[0], shape_sg[1]))
    ds_val_emit = hdf5_ds_meta.create_dataset('val_emitter', shape=(1, ), maxshape=(None, ), dtype=str_dt)
    ds_val_cls = hdf5_ds_meta.create_dataset('val_class',shape=(1,num_classes,),maxshape=(None,num_classes,), dtype='int8')
    ds_val_file = hdf5_ds_meta.create_dataset('val_file', shape=(1, ), maxshape=(None, ), dtype=str_dt)
    
    ds_test_sg = hdf5_ds_sg.create_dataset('test_sg', shape=(1, shape_sg[0], shape_sg[1]), maxshape=(None, shape_sg[0], shape_sg[1]))
    ds_test_emit = hdf5_ds_meta.create_dataset('test_emitter', shape=(1, ), maxshape=(None, ), dtype=str_dt)
    ds_test_cls = hdf5_ds_meta.create_dataset('test_class', shape=(1,num_classes,),maxshape=(None,num_classes,), dtype='int8')
    ds_test_file = hdf5_ds_meta.create_dataset('test_file', shape=(1, ), maxshape=(None, ), dtype=str_dt)

   
    mlb = MultiLabelBinarizer()
    mlb.fit(labels)
    trainset_ind = 0
    valset_ind = 0
    testset_ind = 0
    for emitter in sorted(np.unique(list_ds_emit)):
        # Get index range of emitter
        first_ind = list_ds_emit.index(emitter)
        last_ind = len(list_ds_emit) - 1 - list_ds_emit[::-1].index(emitter)
        if (last_ind - first_ind + 2) < samps_per_class:
            raise ValueError("Emitter " + str(emitter) + " has too few samples")
        emitter_inds = set(range(first_ind, last_ind+1))
        if emitter not in test_specs:
            # Randomly sample for each set's indices
            train_inds = sorted(random.Random(seed).sample(emitter_inds, train_val_test[0]))
            val_inds = sorted(random.Random(seed).sample(emitter_inds-set(train_inds), train_val_test[1]))
            test_inds = sorted(random.Random(seed).sample(emitter_inds-set(train_inds)-set(val_inds), train_val_test[2]))
            emitter = emitter.split("_")
            emitter = (emitter,)
            cls_label = mlb.transform(emitter);
            cls_label = cls_label[0]
            # Add emitter samples to each set
            ds_train_sg.resize(trainset_ind + train_val_test[0], axis=0)
            ds_train_emit.resize(trainset_ind + train_val_test[0], axis=0)
            ds_train_cls.resize(trainset_ind + train_val_test[0], axis=0)
            ds_train_file.resize(trainset_ind + train_val_test[0], axis=0)
        
            ds_train_sg[trainset_ind : trainset_ind + train_val_test[0]] = dataset_sg[train_inds]
            ds_train_emit[trainset_ind : trainset_ind + train_val_test[0]] = dataset_emitter[train_inds]
            ds_train_cls[trainset_ind : trainset_ind + train_val_test[0]] = np.int8(cls_label)
            ds_train_file[trainset_ind : trainset_ind + train_val_test[0]] = dataset_file[train_inds]
            trainset_ind += train_val_test[0]
        
        
            ds_val_sg.resize(valset_ind + train_val_test[1], axis=0)
            ds_val_emit.resize(valset_ind + train_val_test[1], axis=0)
            ds_val_cls.resize(valset_ind + train_val_test[1], axis=0)
            ds_val_file.resize(valset_ind+ train_val_test[1], axis=0)
        
            ds_val_sg[valset_ind : valset_ind + train_val_test[1]] = dataset_sg[val_inds]
            ds_val_emit[valset_ind : valset_ind + train_val_test[1]] = dataset_emitter[val_inds]
            ds_val_cls[valset_ind : valset_ind + train_val_test[1]] = np.int8(cls_label)
            ds_val_file[valset_ind : valset_ind + train_val_test[1]] = dataset_file[val_inds]
            valset_ind += train_val_test[1]
        
        
            ds_test_sg.resize(testset_ind + train_val_test[2], axis=0)
            ds_test_emit.resize(testset_ind + train_val_test[2], axis=0)
            ds_test_cls.resize(testset_ind + train_val_test[2], axis=0)
            ds_test_file.resize(testset_ind + train_val_test[2], axis=0)
        
            ds_test_sg[testset_ind : testset_ind + train_val_test[2]] = dataset_sg[test_inds]
            ds_test_emit[testset_ind : testset_ind + train_val_test[2]] = dataset_emitter[test_inds]
            ds_test_cls[testset_ind : testset_ind + train_val_test[2]] = np.int8(cls_label)
            ds_test_file[testset_ind : testset_ind + train_val_test[2]] = dataset_file[test_inds]
            testset_ind += train_val_test[2]
        else:
            print("Emitter combination %s not used" % (emitter))
      
        

    if shuffle:
        # Rely on seed to maintain correct row integrity
        random.Random(seed).shuffle(ds_train_sg)
        random.Random(seed).shuffle(ds_train_emit)
        random.Random(seed).shuffle(ds_train_cls)
        random.Random(seed).shuffle(ds_train_file)

        random.Random(seed).shuffle(ds_val_sg)
        random.Random(seed).shuffle(ds_val_emit)
        random.Random(seed).shuffle(ds_val_cls)
        random.Random(seed).shuffle(ds_val_file)

        random.Random(seed).shuffle(ds_test_sg)
        random.Random(seed).shuffle(ds_test_emit)
        random.Random(seed).shuffle(ds_test_cls)
        random.Random(seed).shuffle(ds_test_file)
    
    hdf5_ds_sg.close()
    hdf5_ds_meta.close()

    return path_datasets+fname_ds_sg, path_datasets+fname_ds_meta