import h5py
import numpy as np
import pickle 
from  gwpy.timeseries import TimeSeries
import os

from train.model import DeepClean
from train.architectures import Autoencoder
from train.metrics import OnlinePsdRatio, PsdRatio
from lightning import pytorch as pl
import logging


from ml4gw.transforms import ChannelWiseScaler

from utils.filt import BandpassFilter
from clean.frames import LiveDataBuffer, FrameCrawler, Path
from clean.data import DeepCleanInferenceDataset
from clean.infer import OnlineInference

## all the paramters go here (will be taken from the config file)

path_to_model = "../test/test_run/lightning_logs/version_0/checkpoints/last.ckpt"

"""
LiveDataBuffer
        self.crawler = crawler
        self.channels = channels
        self.sample_rate: float = 4096
        self.edge_duration: float = 1
        self.target_duration: float = 1
        self.analysis_duration

DeepCleanInferenceDataset
        hoft_dir: str,
        witness_dir: str,
        channels: list[str],
        kernel_length: float,
        freq_low: list[float],
        freq_high: list[float],
        batch_size: int,
        clean_stride: float,
        sample_rate: float,
        inference_sampling_rate: float,
        start_offset: float = 0,
        filt_order: float = 8, 
        

"""



#print (help(DeepClean.load_from_checkpoint))

model = DeepClean.load_from_checkpoint(
    path_to_model, 
    arch=Autoencoder(num_witnesses=21, 
            hidden_channels=[8,16,32,64]),
    loss=PsdRatio(sample_rate = 4096,
            fftlength = 2,
            freq_low = [55],
            freq_high = [65]),
    metric=OnlinePsdRatio(inference_sampling_rate= 64,
            edge_pad= 0.25,
            filter_pad= 0.5,
            sample_rate= 4096,
            bandpass= BandpassFilter([55], [65], 4096, 8),
            y_scaler= ChannelWiseScaler()),
    patience=None,
    save_top_k_models=10,
)


hoft_dir = "/home/muhammed.saleem/deepClean/deepclean/O3_60Hz/H1/pseudo_replay/llhoft/"
witness_dir = "/home/muhammed.saleem/deepClean/deepclean/O3_60Hz/H1/pseudo_replay/lldetchar/"

channels = ["H1:GDS-CALIB_STRAIN_CLEAN",
            "H1:PEM-CS_MAINSMON_EBAY_1_DQ",
            "H1:ASC-INP1_P_INMON",
            "H1:ASC-INP1_Y_INMON",
            "H1:ASC-MICH_P_INMON",
           "H1:ASC-MICH_Y_INMON",
           "H1:ASC-PRC1_P_INMON",
           "H1:ASC-PRC1_Y_INMON",
           "H1:ASC-PRC2_P_INMON",
           "H1:ASC-PRC2_Y_INMON",
           "H1:ASC-SRC1_P_INMON",
           "H1:ASC-SRC1_Y_INMON",
           "H1:ASC-SRC2_P_INMON",
           "H1:ASC-SRC2_Y_INMON",
           "H1:ASC-DHARD_P_INMON",
           "H1:ASC-DHARD_Y_INMON",
           "H1:ASC-CHARD_P_INMON",
           "H1:ASC-CHARD_Y_INMON",
           "H1:ASC-DSOFT_P_INMON",
           "H1:ASC-DSOFT_Y_INMON",
           "H1:ASC-CSOFT_P_INMON",
           "H1:ASC-CSOFT_Y_INMON"]

inference_dataset = DeepCleanInferenceDataset(
        hoft_dir = hoft_dir,
        witness_dir = witness_dir,
        channels = channels,
        kernel_length = 3.0,
        freq_low = [55],
        freq_high = [65],
        batch_size = 1,
        clean_stride = 1,
        sample_rate = 4096.0,
        inference_sampling_rate = 1,
        start_offset = 0,
        filt_order = 8)

        

device = "cuda"
online_inference = OnlineInference(
        dataset = inference_dataset,
        model = model,
        outdir = './outdir/cleaned_frames',
        device = device)


for k in range(200):
    online_inference.predict_and_write()
    inference_dataset.update()
    





#hoft_buffer = LiveDataBuffer(crawler=crawler_hoft, channels=channels[:1])
#hoft_buffer.update()
#pickle.dump(hoft_buffer.data, open("hoft_buffer.pickle", "wb"))

#detchar_buffer = LiveDataBuffer(crawler=crawler_detchar, channels=channels[1:])
#detchar_buffer.update()
#pickle.dump(detchar_buffer.data, open("detchar_buffer.pickle", "wb"))


