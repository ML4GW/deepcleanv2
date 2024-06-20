import os
import torch
from gwpy.timeseries import TimeSeries
import logging

from train.model import DeepClean
from train.architectures import Autoencoder
from train.metrics import OnlinePsdRatio
from ml4gw.transforms import ChannelWiseScaler
from utils.filt import BandpassFilter
from clean.data import DeepCleanInferenceDataset


class OnlineInference:
    """
    The top-level inference class
        - uses frame directories, pre-loaded models, 
        and output directories as inputs
        
        - get pre-processed data to feed to the model
        - produces predictions 
        - postprocess the prediction
        - writes the cleaned frames   
    """

    def __init__(self, dataset, model, outdir, device):
        self.__logger = logging.getLogger("DeepClean Online Inference")
        self.model = model
        self.device = device
        self.dataset = dataset
        self.outdir = outdir
        if not os.path.exists(outdir):
            os.makedirs(self.outdir)
        
    def predict(self):
        witness = list(self.dataset.X_inference)[0]
        self.pred = self.model(witness.to(self.device))

    def postprocess(self):
        self.noise = self.model.metric.y_scaler(self.pred, reverse=True)
        self.noise = self.model.metric.bandpass(self.noise.cpu().detach().numpy())
        self.noise = torch.tensor(self.noise, device=self.device).flatten()
        start = self.model.metric.filter_pad
        stop = -self.model.metric.filter_pad
        self.noise = self.noise[start:stop]
        self.raw = list(self.dataset.y_inference)[0].to(self.device).flatten()
        self.raw = self.model.metric.y_scaler(self.raw, reverse=True)
        self.raw = self.raw[start:stop]

    def write(self):
        cleaned = self.raw - self.noise
        cleaned = cleaned.cpu().numpy()
        gpstime = self.dataset.hoft_crawler.t0 - 1
        ts = TimeSeries(data=cleaned, t0=gpstime, sample_rate=self.dataset.hparams.sample_rate,
                channel=self.dataset.strain_channel+"_DC", unit='seconds')
        fname = self.dataset.hoft_crawler.file_format.get_name(timestamp=gpstime, length=1)
        filepath = os.path.join(self.outdir, fname)
        ts.write(filepath)
        self.__logger.info(f"Data written to file: {filepath}")
        
        
    def predict_and_write(self):
        self.predict()
        self.postprocess()
        self.write()
