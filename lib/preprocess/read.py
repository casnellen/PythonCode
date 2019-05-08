"""
Functions for reading data and organizing them into HDF5 files
"""

__author__ = 'clee'
# Python 3.6.1

# Standard lib
import os
import sys
import datetime

# Additional lib
import h5py
import scipy.io as sio
import numpy as np


def read_mat(path_mat_data, path_data_store, func_preprocess,
             mat_key = 'pooledData',
             exclude_emits=[],
             data_notes="",
             fname_sg='/spectrograms.h5',
             fname_meta='/metadata.h5'):
    """
    Read spectrograms from .mat files and stores them into two HDF5 files
    Expects directory structure of [path_mat_data]/[Emitter Number]/[.mat file]
    Expects input spectrograms to have time axis oriented horizontally (output will be time-vertical)

    :param path_mat_data:           str                     Path to .mat spectrograms
    :param path_data_store:         str                     Path to output folder
    :param func_preprocess:         function                Takes in spectrogram, outputs same size spectrogram
    :param mat_key                  str                     Key of spectrogram variable in .mat file
    :param exclude_emits:           list of strings         List of emitter folders to ignore
    :param data_notes:              str                     Notes about data being generated
    :param fname_sg:                str                     Name of spectrograms file
    :param fname_meta:              str                     Name of metadata file

    :return:                        2-tuple of strings      File paths of spectrogram HDF5, metadata HDF5, and parent directory
    """

    # Create output folders and print notes to text file
    rundate = datetime.datetime.now().strftime('%b%d%y-%H%M%S')
    os.mkdir(path_data_store + '/' + rundate)
    os.mkdir(path_data_store + '/' + rundate + '/datasets')
    orig_stdout = sys.stdout
    data_notes_file = open(path_data_store + '/' + rundate + '/_data_notes.txt', 'w')
    sys.stdout = data_notes_file
    print(data_notes)
    sys.stdout = orig_stdout
    data_notes_file.close()
    path_hdf5 = path_data_store + '/' + rundate

    # Find (time-vertical) spectrogram shape
    shape_sg = None
    for root, subdirs, files in os.walk(path_mat_data):
        if (root != path_mat_data and files != []):
            for file in files:
                path_file = os.path.join(root, file)
                if file.split('.')[-1] == 'mat':
                    shape_sg = sio.loadmat(path_file)[mat_key].T.shape
                    break
            break
    if shape_sg == None:
        raise(FileNotFoundError, "Could not find .mat file")

    # Set up hdf5 files
    if os.path.isfile(path_hdf5 + fname_sg):
        os.remove(path_hdf5 + fname_sg)
    if os.path.isfile(path_hdf5 + fname_meta):
        os.remove(path_hdf5 + fname_meta)
    hdf5_sg = h5py.File(path_hdf5 + fname_sg, 'w')
    hdf5_meta = h5py.File(path_hdf5 + fname_meta, 'w')
    dataset_sg = hdf5_sg.create_dataset('sg', shape=(1, shape_sg[0], shape_sg[1]), maxshape=(None, shape_sg[0], shape_sg[1]))
    dataset_emit = hdf5_meta.create_dataset('emitter', shape=(1, ), maxshape=(None, ), dtype='int16')
    str_dt = h5py.special_dtype(vlen=str)
    dataset_file = hdf5_meta.create_dataset('file', shape=(1, ), maxshape=(None, ), dtype=str_dt)

    # Process .mat files
    samp_ind = 0
    for root, subdirs, files in os.walk(path_mat_data):
        subdirs.sort()
        files.sort()

        # Check for the right directory
        if (root != path_mat_data and
            root.split('/')[-1] not in exclude_emits and
            files != []):

            for file in files:
                path_file = os.path.join(root, file)
                if file.split('.')[-1] == 'mat':
                    dataset_sg.resize(samp_ind+1, axis=0)
                    dataset_emit.resize(samp_ind+1, axis=0)
                    dataset_file.resize(samp_ind+1, axis=0)

                    # Store data
                    dataset_sg[samp_ind] = func_preprocess(sio.loadmat(path_file)[mat_key].T)
                    dataset_emit[samp_ind] = np.int16(root.split('/')[-1])
                    dataset_file[samp_ind] = file.split('.')[0]
                    samp_ind += 1
    hdf5_sg.close()
    hdf5_meta.close()

    return path_hdf5+fname_sg, path_hdf5+fname_meta, path_hdf5

def read_mat_multi(path_mat_data, path_data_store, func_preprocess,
             mat_key = 'pooledData',
             exclude_emits=[],
             data_notes="",
             fname_sg='/spectrograms.h5',
             fname_meta='/metadata.h5'):
    """
    Read spectrograms from .mat files and stores them into two HDF5 files
    Expects directory structure of [path_mat_data]/[Emitter Number]/[.mat file]
    Expects input spectrograms to have time axis oriented horizontally (output will be time-vertical)

    :param path_mat_data:           str                     Path to .mat spectrograms
    :param path_data_store:         str                     Path to output folder
    :param func_preprocess:         function                Takes in spectrogram, outputs same size spectrogram
    :param mat_key                  str                     Key of spectrogram variable in .mat file
    :param exclude_emits:           list of strings         List of emitter folders to ignore
    :param data_notes:              str                     Notes about data being generated
    :param fname_sg:                str                     Name of spectrograms file
    :param fname_meta:              str                     Name of metadata file

    :return:                        2-tuple of strings      File paths of spectrogram HDF5, metadata HDF5, and parent directory
    """

    # Create output folders and print notes to text file
    rundate = datetime.datetime.now().strftime('%b%d%y-%H%M%S')
    os.mkdir(path_data_store + '/' + rundate)
    os.mkdir(path_data_store + '/' + rundate + '/datasets')
    orig_stdout = sys.stdout
    data_notes_file = open(path_data_store + '/' + rundate + '/_data_notes.txt', 'w')
    sys.stdout = data_notes_file
    print(data_notes)
    sys.stdout = orig_stdout
    data_notes_file.close()
    path_hdf5 = path_data_store + '/' + rundate

    # Find (time-vertical) spectrogram shape
    shape_sg = None
    for root, subdirs, files in os.walk(path_mat_data):
        if (root != path_mat_data and files != []):
            for file in files:
                path_file = os.path.join(root, file)
                if file.split('.')[-1] == 'mat':
                    shape_sg = sio.loadmat(path_file)[mat_key].T.shape
                    break
            break
    if shape_sg == None:
        raise(FileNotFoundError, "Could not find .mat file")

    # Set up hdf5 files
    if os.path.isfile(path_hdf5 + fname_sg):
        os.remove(path_hdf5 + fname_sg)
    if os.path.isfile(path_hdf5 + fname_meta):
        os.remove(path_hdf5 + fname_meta)
    hdf5_sg = h5py.File(path_hdf5 + fname_sg, 'w')
    hdf5_meta = h5py.File(path_hdf5 + fname_meta, 'w')
    dataset_sg = hdf5_sg.create_dataset('sg', shape=(1, shape_sg[0], shape_sg[1]), maxshape=(None, shape_sg[0], shape_sg[1]))
    dataset_emit = hdf5_meta.create_dataset('emitter', shape=(1, ), maxshape=(None, ), dtype='int16')
    str_dt = h5py.special_dtype(vlen=str)
    dataset_file = hdf5_meta.create_dataset('file', shape=(1, ), maxshape=(None, ), dtype=str_dt)

    # Process .mat files
    samp_ind = 0
    for root, subdirs, files in os.walk(path_mat_data):
        subdirs.sort()
        files.sort()

        # Check for the right directory
        if (root != path_mat_data and
            root.split('/')[-1] not in exclude_emits and
            files != []):

            for file in files:
                path_file = os.path.join(root, file)
                if file.split('.')[-1] == 'mat':
                    dataset_sg.resize(samp_ind+1, axis=0)
                    dataset_emit.resize(samp_ind+1, axis=0)
                    dataset_file.resize(samp_ind+1, axis=0)

                    # Store data
                    dataset_sg[samp_ind] = func_preprocess(sio.loadmat(path_file)[mat_key].T)
                    dataset_emit[samp_ind] = np.int16(root.split('/')[-1])
                    dataset_file[samp_ind] = file.split('.')[0]
                    samp_ind += 1
    hdf5_sg.close()
    hdf5_meta.close()

    return path_hdf5+fname_sg, path_hdf5+fname_meta, path_hdf5

def read_mat_binary_relevance(path_mat_data, path_data_store, func_preprocess,
             mat_key = 'pooledData',
             exclude_emits=[],
             data_notes="",
             fname_sg='/spectrograms.h5',
             fname_meta='/metadata.h5'):
    """
    Read spectrograms from .mat files and stores them into two HDF5 files
    Expects directory structure of [path_mat_data]/[Emitter Number]/[.mat file]
    Expects input spectrograms to have time axis oriented horizontally (output will be time-vertical)

    :param path_mat_data:           str                     Path to .mat spectrograms
    :param path_data_store:         str                     Path to output folder
    :param func_preprocess:         function                Takes in spectrogram, outputs same size spectrogram
    :param mat_key                  str                     Key of spectrogram variable in .mat file
    :param exclude_emits:           list of strings         List of emitter folders to ignore
    :param data_notes:              str                     Notes about data being generated
    :param fname_sg:                str                     Name of spectrograms file
    :param fname_meta:              str                     Name of metadata file

    :return:                        2-tuple of strings      File paths of spectrogram HDF5, metadata HDF5, and parent directory
    """

    # Create output folders and print notes to text file
    rundate = datetime.datetime.now().strftime('%b%d%y-%H%M%S')
    os.mkdir(path_data_store + '/' + rundate)
    os.mkdir(path_data_store + '/' + rundate + '/datasets')
    orig_stdout = sys.stdout
    data_notes_file = open(path_data_store + '/' + rundate + '/_data_notes.txt', 'w')
    sys.stdout = data_notes_file
    print(data_notes)
    sys.stdout = orig_stdout
    data_notes_file.close()
    path_hdf5 = path_data_store + '/' + rundate

    # Find (time-vertical) spectrogram shape
    shape_sg = None
    for root, subdirs, files in os.walk(path_mat_data):
        if (root != path_mat_data and files != []):
            for file in files:
                path_file = os.path.join(root, file)
                if file.split('.')[-1] == 'mat':
                    shape_sg = sio.loadmat(path_file)[mat_key].T.shape
                    break
            break
    if shape_sg == None:
        raise(FileNotFoundError, "Could not find .mat file")

    # Set up hdf5 files
    if os.path.isfile(path_hdf5 + fname_sg):
        os.remove(path_hdf5 + fname_sg)
    if os.path.isfile(path_hdf5 + fname_meta):
        os.remove(path_hdf5 + fname_meta)
    hdf5_sg = h5py.File(path_hdf5 + fname_sg, 'w')
    hdf5_meta = h5py.File(path_hdf5 + fname_meta, 'w')
    dataset_sg = hdf5_sg.create_dataset('sg', shape=(1, shape_sg[0], shape_sg[1]), maxshape=(None, shape_sg[0], shape_sg[1]))
    str_dt = h5py.special_dtype(vlen=str)
    dataset_emit = hdf5_meta.create_dataset('emitter', shape=(1, ), maxshape=(None, ), dtype=str_dt)
    dataset_file = hdf5_meta.create_dataset('file', shape=(1, ), maxshape=(None, ), dtype=str_dt)

    # Process .mat files
    samp_ind = 0
    for root, subdirs, files in os.walk(path_mat_data):
        subdirs.sort()
        files.sort()

        # Check for the right directory
        if (root != path_mat_data and
            root.split('/')[-1] not in exclude_emits and
            files != []):

            for file in files:
                path_file = os.path.join(root, file)
                if file.split('.')[-1] == 'mat':
                    dataset_sg.resize(samp_ind+1, axis=0)
                    dataset_emit.resize(samp_ind+1, axis=0)
                    dataset_file.resize(samp_ind+1, axis=0)

                    # Store data
                    dataset_sg[samp_ind] = func_preprocess(sio.loadmat(path_file)[mat_key].T)
                    dataset_emit[samp_ind] = root.split('/')[-1]
                    dataset_file[samp_ind] = file.split('.')[0]
                    samp_ind += 1
    hdf5_sg.close()
    hdf5_meta.close()

    return path_hdf5+fname_sg, path_hdf5+fname_meta, path_hdf5

def read_mat_multi_label(path_mat_data, path_data_store, func_preprocess,
             mat_key = 'pooledData',
             exclude_emits=[],
             data_notes="",
             fname_sg='/spectrograms.h5',
             fname_meta='/metadata.h5'):
    """
    Read spectrograms from .mat files and stores them into two HDF5 files
    Expects directory structure of [path_mat_data]/[Emitter Number]/[.mat file]
    Expects input spectrograms to have time axis oriented horizontally (output will be time-vertical)

    :param path_mat_data:           str                     Path to .mat spectrograms
    :param path_data_store:         str                     Path to output folder
    :param func_preprocess:         function                Takes in spectrogram, outputs same size spectrogram
    :param mat_key                  str                     Key of spectrogram variable in .mat file
    :param exclude_emits:           list of strings         List of emitter folders to ignore
    :param data_notes:              str                     Notes about data being generated
    :param fname_sg:                str                     Name of spectrograms file
    :param fname_meta:              str                     Name of metadata file

    :return:                        2-tuple of strings      File paths of spectrogram HDF5, metadata HDF5, and parent directory
    """

    # Create output folders and print notes to text file
    rundate = datetime.datetime.now().strftime('%b%d%y-%H%M%S')
    os.mkdir(path_data_store + '/' + rundate)
    os.mkdir(path_data_store + '/' + rundate + '/datasets')
    orig_stdout = sys.stdout
    data_notes_file = open(path_data_store + '/' + rundate + '/_data_notes.txt', 'w')
    sys.stdout = data_notes_file
    print(data_notes)
    sys.stdout = orig_stdout
    data_notes_file.close()
    path_hdf5 = path_data_store + '/' + rundate

    # Find (time-vertical) spectrogram shape
    shape_sg = None
    for root, subdirs, files in os.walk(path_mat_data):
        if (root != path_mat_data and files != []):
            for file in files:
                path_file = os.path.join(root, file)
                if file.split('.')[-1] == 'mat':
                    shape_sg = sio.loadmat(path_file)[mat_key].T.shape
                    break
            break
    if shape_sg == None:
        raise(FileNotFoundError, "Could not find .mat file")

    # Set up hdf5 files
    if os.path.isfile(path_hdf5 + fname_sg):
        os.remove(path_hdf5 + fname_sg)
    if os.path.isfile(path_hdf5 + fname_meta):
        os.remove(path_hdf5 + fname_meta)
    hdf5_sg = h5py.File(path_hdf5 + fname_sg, 'w')
    hdf5_meta = h5py.File(path_hdf5 + fname_meta, 'w')
    dataset_sg = hdf5_sg.create_dataset('sg', shape=(1, shape_sg[0], shape_sg[1]), maxshape=(None, shape_sg[0], shape_sg[1]))
    str_dt = h5py.special_dtype(vlen=str)
    dataset_emit = hdf5_meta.create_dataset('emitter', shape=(1, ), maxshape=(None, ), dtype=str_dt)
    dataset_file = hdf5_meta.create_dataset('file', shape=(1, ), maxshape=(None, ), dtype=str_dt)

    # Process .mat files
    samp_ind = 0
    for root, subdirs, files in os.walk(path_mat_data):
        subdirs.sort()
        files.sort()

        # Check for the right directory
        if (root != path_mat_data and
            root.split('/')[-1] not in exclude_emits and
            files != []):

            for file in files:
                path_file = os.path.join(root, file)
                if file.split('.')[-1] == 'mat':
                    dataset_sg.resize(samp_ind+1, axis=0)
                    dataset_emit.resize(samp_ind+1, axis=0)
                    dataset_file.resize(samp_ind+1, axis=0)

                    # Store data
                    dataset_sg[samp_ind] = func_preprocess(sio.loadmat(path_file)[mat_key].T)
                    fullLabel = root.split('/')[-1]
                    dataset_emit[samp_ind] = fullLabel.split("_")
                    dataset_file[samp_ind] = file.split('.')[0]
                    samp_ind += 1
    hdf5_sg.close()
    hdf5_meta.close()

    return path_hdf5+fname_sg, path_hdf5+fname_meta, path_hdf5