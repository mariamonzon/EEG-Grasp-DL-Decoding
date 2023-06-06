"""
Cropped Decoding on Custom dataset

======================================

"""

from braindecode.datasets.moabb import MOABBDataset
import mne
from braindecode.datautil.preprocess import exponential_moving_standardize
from braindecode.datautil.preprocess import MNEPreproc, NumpyPreproc, preprocess
import torch
from braindecode.util import set_random_seeds
from braindecode.models import ShallowFBCSPNet, EEGNetv4
from braindecode.models.util import to_dense_prediction_model, get_output_shape
from skorch.callbacks import LRScheduler
from skorch.helper import predefined_split
from braindecode import EEGClassifier
from braindecode.training.losses import CroppedLoss

import argparse

######################################################################
# # In Braindecode, there are two supported configurations created for
# # training models: trialwise decoding and cropped decoding. We will
# # explain this visually by comparing trialwise to cropped decoding.
# #     -  The network architecture implicitly defines the crop size (it is the
# #        receptive field size, i.e., the number of timesteps the network uses
# #        to make a single prediction)
# #     -  The window size is a user-defined hyperparameter, called
# #        ``input_window_samples`` in Braindecode. It mostly affects runtime
# #        (larger window sizes should be faster). As a rule of thumb, you can
# #        set it to two times the crop size.
# #     -  Crop size and window size together define how many predictions the
# #        network makes per window: ``#window−#crop+1=#predictions``

######################################################################
# Loading and preprocessing the dataset
# -------------------------------------

######################################################################
# Datasets

#
# dataset = SleepPhysionet(
#     subject_ids=[0, 1], recording_ids=[1], crop_wake_mins=30)
#


if __name__=='__main__':

    # create inline argumends
    parser = argparse.ArgumentParser(description="EEG decoding with Deep Learning")
    parser.add_argument('-lf', '--low_freq_cut',    type=float, default=4.0,    help='low cut frequency for filtering' )
    parser.add_argument('-hf', '--high_freq_cut',   type=float, default=38.0,   help='high cut frequency for filtering')
    parser.add_argument('-dev', '--device',         type=str,   default='gpu',  help='Select training device: cpu or gpu',
                        choices = ['gpu', 'cpu'])

    args = parser.parse_args()
    low_cut_hz = args.low_freq_cut  # low cut frequency for filtering
    high_cut_hz = args.high_freq_cut        # high cut frequency for filtering

    # Parameters for exponential moving standardization
    factor_new = 1e-3
    init_block_size = 1000
    subject_id = [1,2,3,4,5,6,7]
    # dataset = MOABBDataset(dataset_name="BNCI2014001", subject_ids=[subject_id])
    #  Schirrmeister2017
    dataset = MOABBDataset(dataset_name="BNCI2014001", subject_ids=subject_id)
    # dataset_hgd = MOABBDataset(dataset_name="Schirrmeister2017", subject_ids=subject_id)

    preprocessors = [
        # keep only EEG sensors
        MNEPreproc(fn='pick_types', eeg=True, meg=False, stim=False),
        # convert from volt to microvolt, directly modifying the numpy array
        NumpyPreproc(fn=lambda x: x * 1e6),
        # bandpass filter
        MNEPreproc(fn='filter', l_freq=low_cut_hz, h_freq=high_cut_hz),
        # exponential moving standardization
        NumpyPreproc(fn=exponential_moving_standardize, factor_new=factor_new,
            init_block_size=init_block_size)
    ]
    model = EEGNetv4(
        in_chans =22,
        n_classes=4,
        input_window_samples=1000,
        final_conv_length=30,
    )
    print(model)
    # Transform the data
    preprocess(dataset, preprocessors)


    ######################################################################
    # Create model and compute windowing parameters
    # ---------------------------------------------
    ######################################################################
    # In contrast to trialwise decoding, we first have to create the model
    # before we can cut the dataset into windows.
    #
    # This is because we need to # know the receptive field of the network to know how large the window
    # stride should be.



    ######################################################################
    # We first choose the compute/input window size that will be fed to the
    # network during training This has to be larger than the networks
    # receptive field size and can otherwise be chosen for computational
    # efficiency (see explanations in the beginning of this tutorial). Here we
    # choose 1000 samples, which are 4 seconds for the 250 Hz sampling rate.

    input_window_samples = 1000


    ######################################################################
    # Now we create the model. To enable it to be used in cropped decoding
    # efficiently, we manually set the length of the final convolution layer
    # to some length that makes the receptive field of the ConvNet smaller
    # than ``input_window_samples`` (see ``final_conv_length=30`` in the model
    # definition).
    #



    n_classes=4
    # Extract number of chans from dataset

    n_chans = dataset[0][0].shape[0]

    model = ShallowFBCSPNet(
        n_chans,
        n_classes,
        input_window_samples=input_window_samples,
        final_conv_length=30,
    )


    # Send model to GPU
    if args.device == 'gpu':
        model.cuda()



    ######################################################################
    # And now we transform model with strides to a model that outputs dense
    # prediction, so we can use it to obtain predictions for all
    # crops.
    #

    to_dense_prediction_model(model)


    ######################################################################
    # To know the models’ receptive field, we calculate the shape of model
    # output for a dummy input.
    #

    n_preds_per_input = get_output_shape(model, n_chans, input_window_samples)[2]


    ######################################################################
    # Cut the data into windows
    # -------------------------
    #


    ######################################################################
    # In contrast to trialwise decoding, we have to supply an explicit window size and window stride to the
    # ``create_windows_from_events`` function.
    #

    import numpy as np
    from braindecode.datautil.windowers import create_windows_from_events

    trial_start_offset_seconds = -0.5
    # Extract sampling frequency, check that they are same in all datasets
    sfreq = dataset.datasets[0].raw.info['sfreq']
    assert all([ds.raw.info['sfreq'] == sfreq for ds in dataset.datasets])

    # Calculate the trial start offset in samples.
    trial_start_offset_samples = int(trial_start_offset_seconds * sfreq)

    # Create windows using braindecode function for this. It needs parameters to define how
    # trials should be used.
    windows_dataset = create_windows_from_events(
        dataset,
        trial_start_offset_samples=trial_start_offset_samples,
        trial_stop_offset_samples=0,
        window_size_samples=input_window_samples,
        window_stride_samples=n_preds_per_input,
        drop_last_window=False,
        preload=True,
    )


    ######################################################################
    # Split the dataset
    # -----------------
    #
    # This code is the same as in trialwise decoding.
    #

    splitted = windows_dataset.split('session')
    train_set = splitted['session_T']
    valid_set = splitted['session_E']


    ######################################################################
    # Training
    # --------
    #


    ######################################################################
    # In difference to trialwise decoding, we now should supply
    # ``cropped=True`` to the EEGClassifier, and ``CroppedLoss`` as the
    # criterion, as well as ``criterion__loss_function`` as the loss function
    # applied to the meaned predictions.
    #


    ######################################################################
    # .. note::
    #    In this tutorial, we use some default parameters that we
    #    have found to work well for motor decoding, however we strongly
    #    encourage you to perform your own hyperparameter optimization using
    #    cross validation on your training data.



    # These values we found good for shallow network:
    lr = 0.0625 * 0.01
    weight_decay = 0.00001

    # For deep4 they should be:
    # lr = 1 * 0.01
    # weight_decay = 0.5 * 0.001

    batch_size = 64
    n_epochs = 30

    clf = EEGClassifier(
        model,
        cropped=True,
        criterion=CroppedLoss,
        criterion__loss_function=torch.nn.functional.nll_loss,
        optimizer=torch.optim.AdamW,
        train_split=predefined_split(valid_set),
        optimizer__lr=lr,
        optimizer__weight_decay=weight_decay,
        iterator_train__shuffle=True,
        batch_size=batch_size,
        callbacks=[
            "accuracy", ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1)),
        ],
        device=device,
    )
    # Model training for a specified number of epochs. `y` is None as it is already supplied
    # in the dataset.
    clf.fit(train_set, y=None, epochs=n_epochs)
    clf.save_params(f_params='model.pkl',
                    f_optimizer='optimizer.pkl',
                    f_history='history.json')

    print("Model Trained")

    #

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import pandas as pd
    # Extract loss and accuracy values for plotting from history object
    results_columns = ['train_loss', 'valid_loss', 'train_accuracy', 'valid_accuracy']
    df = pd.DataFrame(clf.history[:, results_columns], columns=results_columns,
                      index=clf.history[:, 'epoch'])

    # get percent of misclass for better visual comparison to loss
    df = df.assign(train_misclass=100 - 100 * df.train_accuracy,
                   valid_misclass=100 - 100 * df.valid_accuracy)

    plt.style.use('seaborn')
    fig, ax1 = plt.subplots(figsize=(8, 3))
    df.loc[:, ['train_loss', 'valid_loss']].plot(
        ax=ax1, style=['-', ':'], marker='o', color='tab:blue', legend=False, fontsize=14)

    ax1.tick_params(axis='y', labelcolor='tab:blue', labelsize=14)
    ax1.set_ylabel("Loss", color='tab:blue', fontsize=14)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    df.loc[:, ['train_misclass', 'valid_misclass']].plot(
        ax=ax2, style=['-', ':'], marker='o', color='tab:red', legend=False)
    ax2.tick_params(axis='y', labelcolor='tab:red', labelsize=14)
    ax2.set_ylabel("Misclassification Rate [%]", color='tab:red', fontsize=14)
    ax2.set_ylim(ax2.get_ylim()[0], 85)  # make some room for legend
    ax1.set_xlabel("Epoch", fontsize=14)

    # where some data has already been plotted to ax
    handles = []
    handles.append(Line2D([0], [0], color='black', linewidth=1, linestyle='-', label='Train'))
    handles.append(Line2D([0], [0], color='black', linewidth=1, linestyle=':', label='Valid'))
    plt.legend(handles, [h.get_label() for h in handles], fontsize=14)
    plt.tight_layout()
    plt.show()




    from skorch.callbacks import EpochScoring
    # Neural Net wrapper
    from skorch import NeuralNetClassifier
    # PyTorch model
    # Pipeline
    from sklearn.pipeline import Pipeline
    # Scaler
    from sklearn.preprocessing import StandardScaler
    # Grid Search
    from sklearn.model_selection import GridSearchCV
    # The Neural Net is instantiated, none hyperparameter is provided
    nn = NeuralNetClassifier(model, verbose=0, train_split=False)
    # The pipeline is instantiated, it wraps scaling and training phase
    pipeline = Pipeline([('scale', StandardScaler()), ('nn', nn)])

    # The parameters for the grid search are defined
    # It must be used the prefix "nn__" when setting hyperparamters for the training phase
    # It must be used the prefix "nn__module__" when setting hyperparameters for the Neural Net
    params = {
        'nn__max_epochs':[10, 20],
        'nn__lr': [0.1, 0.01],
        'nn__module__num_units': [5, 10],
        'nn__module__dropout': [0.1, 0.5],
        # 'nn__optimizer': [optim.Adam, optim.SGD, optim.RMSprop]
        }

    # The grid search module is instantiated
    gs = GridSearchCV(pipeline, params, refit=False, cv=5, scoring='balanced_accuracy', verbose=1)
    # Initialize grid search
    gs.fit(train_set)





    ############ TEST ############################

    net = EEGClassifier(
        model,
        cropped=True,
        criterion=CroppedLoss,
        criterion__loss_function=torch.nn.functional.nll_loss,
        optimizer=torch.optim.AdamW,
        optimizer__lr=lr,
        optimizer__momentum=0.95,
        optimizer__weight_decay=weight_decay,
        iterator_train__shuffle=True,
        batch_size=batch_size,
        callbacks=[ "accuracy", ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1)),  ],
        device=device,
    ).initialize()
    net.load_params(f_params='model.pkl',
                    f_optimizer='optimizer.pkl',
                    f_history='history.json')

    net.predict(test_set)


