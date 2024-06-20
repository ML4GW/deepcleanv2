from collections.abc import Callable
from typing import Optional

import torch
from ml4gw.transforms import SpectralDensity
from torchmetrics import Metric


class PsdRatio(torch.nn.Module):
    def __init__(
        self,
        sample_rate: float,
        fftlength: float,
        freq_low: list[float],
        freq_high: list[float],
        overlap: Optional[float] = None,
        asd: bool = False,
    ) -> None:
        super().__init__()
        self.spectral_density = SpectralDensity(
            sample_rate,
            fftlength,
            overlap=overlap,
            average="median",
            fast=True,
        )
        self.asd = asd
        self.sample_rate = sample_rate

        N = int(fftlength * sample_rate / 2) + 1
        mask = torch.zeros((N,), dtype=torch.bool)
        for fl, fh in zip(freq_low, freq_high):
            low = int(fl * fftlength)
            high = int(fh * fftlength)
            mask[low : high + 1] = 1
        self.register_buffer("mask", mask)

    def forward(self, pred, strain):
        cleaned = strain - pred
        residual = self.spectral_density(cleaned.double())
        target = self.spectral_density(strain.double())

        ratio = residual / target
        ratio = ratio[:, self.mask]
        if self.asd:
            ratio = ratio**0.5
        loss = ratio.mean(dim=-1)
        return loss


class OnlinePsdRatio(Metric):
    def __init__(
        self,
        inference_sampling_rate: float,
        edge_pad: float,
        filter_pad: float,
        sample_rate: float,
        bandpass: Callable,
        y_scaler: torch.nn.Module,
    ) -> None:
        super().__init__()
        self.stride = int(sample_rate / inference_sampling_rate)
        self.filter_pad = int(filter_pad * sample_rate)
        self.edge_pad = int(edge_pad * sample_rate)
        self.sample_rate = sample_rate

        self.loss_fn = None
        self.bandpass = bandpass
        self.y_scaler = y_scaler

        self.add_state("predictions", default=[])
        self.add_state("strain", default=[])

    def update(self, y, kind):
        if self.loss_fn is None:
            raise ValueError("Must provide loss_fn before calling update")
        getattr(self, kind).append(y)

    def clean(self):
        # first build our overlapping predictions
        # into a single timeseries of noise predictions
        size = sum([i.numel() for i in self.strain])
        batch_size = len(self.predictions[0])
        device = self.predictions[0].device
        dtype = self.predictions[0].dtype

        y_pred = torch.zeros(
            (size - self.edge_pad,), device=device, dtype=dtype
        )

        # for each predicted window, only slice out
        # the single stride of new data from it that's
        # sufficiently far from the edge to be considered
        # "safe" and place it in the corresponding position
        # in the full timeseries. We can do this array-style
        # with some fancy indexing. We'll start by building
        # the array of indices we'll grab from the predicted batches
        get_idx = torch.arange(self.stride, device=device)
        offset = int(self.sample_rate) - self.edge_pad - self.stride
        get_idx += offset

        # then turn this into a matrix of indices where
        # each row of predictions will go in the timeseries
        set_idx = get_idx.view(1, -1).repeat(batch_size, 1)
        batch_offset = torch.arange(batch_size, device=device)
        set_idx += batch_offset[:, None] * self.stride

        for i, y in enumerate(self.predictions):
            sidx = set_idx[: len(y)]
            y_pred[sidx + i * batch_size * self.stride] = y[:, get_idx]

            # for the very first frame, we have no choice
            # but to fill the left side with our predictions.
            # This won't really matter since we don't end up
            # measuring ourselves on this frame, but we'll need
            # it for providing filter padding.
            if not i:
                y_pred[:offset] = y[0, :offset]

        # now clean the target timeseries in the
        # online fashion, one frame at a time, plus
        # some filter settle-in padding on each side.
        # Ignore the first and last frames to account
        # for this filter settle-in.
        num_frames = int((len(y_pred) - self.filter_pad) // self.sample_rate)
        noise = []
        for i in range(1, num_frames - 1):
            start = int(i * self.sample_rate) - self.filter_pad
            stop = int((i + 1) * self.sample_rate) + self.filter_pad
            noise.append(y_pred[start:stop])

        # postprocess, doing the bandpass filtering back
        # in numpy because torchaudio won't work
        noise = torch.stack(noise).double()
        noise = self.y_scaler(noise, reverse=True)
        noise = self.bandpass(noise.cpu().numpy())
        noise = torch.tensor(noise, device=device)

        # slice out the filter padding so that the
        # frames in each row are no longer overlapping,
        # then reshape them to a proper timeseries
        noise = noise[:, self.filter_pad : -self.filter_pad]
        noise = noise.reshape(1, -1)

        # reshape our raw strain into a timeseries
        raw = torch.cat(self.strain, dim=0)[1 : num_frames - 1]
        raw = raw.view(1, -1)
        return noise, raw.double()

    def compute(self, reduce: bool = True):
        noise, raw = self.clean()
        if reduce:
            return self.loss_fn(noise, raw).mean()
        return noise, raw
