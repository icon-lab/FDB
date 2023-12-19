"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py
"""

import enum
import math

import numpy as np
import torch as th
import matplotlib.pyplot as plt
import time
import os
from .nn import mean_flat

class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB


class DiffusionBridge:
    """
    Utilities for training and sampling FDB.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param undersampling_rate: acceleration rate determining intensity of k-space removal.
    :param image_size: the dimensions of the image used by the model.
    :param data_type: a string to differntiate singlecoil and multicoil.
    """

    def __init__(
        self,
        *,
        steps,
        undersampling_rate,
        image_size,
        data_type,
    ):

        self.undersampling_rate = undersampling_rate
        self.image_size = image_size
        self.data_type = data_type

        self.num_timesteps = steps

        if os.path.isfile("w.npy"):
            self.w = np.load("w.npy")
            self.w = np.interp(np.linspace(0, 1, self.num_timesteps), np.linspace(0, 1, self.w.shape[0]), self.w)
            self.n = np.ones(self.w.shape[0])
        else:
            self.w = np.zeros(self.num_timesteps)
            self.n = np.zeros(self.num_timesteps)

    def fft2c(self, x, dim=((-2,-1)), img_shape=None):
        """ 2 dimensional Fast Fourier Transform """
        return th.fft.fftshift(th.fft.fft2(th.fft.ifftshift(x, dim=dim), s=img_shape, dim=dim), dim = dim)

    def ifft2c(self, x, dim=((-2,-1)), img_shape=None):
        """ 2 dimensional Inverse Fast Fourier Transform """
        return th.fft.fftshift(th.fft.ifft2(th.fft.ifftshift(x, dim=dim), s=img_shape, dim=dim), dim = dim)

    def q_sample(self, x_0, t):
        """
        Remove k-space points from the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_0: the initial fully sampled image.
        :param t: the number of forward steps
        :return x_t: the undersampled image after removing k-space points from x_0
        """

        dim = self.image_size
        R = self.undersampling_rate
        t = int(t[0])
        T = self.num_timesteps
        N = int(self.image_size * self.image_size * (1 - 1 / R))
        n = int(t * N / T)
        r_start = dim / 2
        if self.data_type == "singlecoil":
            r_end = 3
        elif self.data_type == "multicoil":
            r_end = 4
        r = r_start - t * (r_start - r_end) / T

        img = x_0[:, [0]] + x_0[:, [1]] * 1j
        img = self.fft2c(img)
        img_0 = th.clone(img)

        for i in range(n):
            x = np.random.randint(dim)
            y = np.random.randint(dim)
            while th.all(img[:, :, x, y] == 0) or (x - dim / 2) ** 2 + (y - dim / 2) ** 2 < r ** 2:
                x = np.random.randint(dim)
                y = np.random.randint(dim)
            img[:, :, x, y] = 0

            if i == n - int(N / T):
                img_t_minus_1 = th.clone(img)

        self.w[t] = (self.n[t] / (self.n[t] + 1)) * self.w[t] + (1 / (self.n[t] + 1)) * ((th.mean(th.abs(img_t_minus_1)) - th.mean(th.abs(img))) / (th.mean(th.abs(img_0)) - th.mean(th.abs(img)))).cpu().detach().numpy()
        self.n[t] += 1
        np.save("w.npy", self.w)

        img = self.ifft2c(img)
        x_t = th.cat([img.real, img.imag], 1)

        return x_t       

    def p_sample(self, model, x_t, M, t):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x_t: the current tensor at timestep t.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :return x_t_minus_1: the tensor at timestep t-1
        """

        model_output = model(x_t, t)

        pred_x0 = model_output.clamp(-1, 1)

        x_t_minus_1 = self.ifft2c(M[t] * self.fft2c(pred_x0)) + _extract_into_tensor(1 - self.w, t, x_t.shape) * self.ifft2c(M[t+1] * self.fft2c(x_t - pred_x0))

        return x_t_minus_1

    def p_sample_loop_condition(
        self,
        model,
        shape,
        kspace,
        mask,
        coil_map=None,
        model_kwargs=None,
        device=None
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param kspace: the kspace of the undersampled image
        :param mask: the undersampling mask
        :param coil_map: the coil sensitivities for multicoil data, (None for singlecoil)
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :return: a non-differentiable batch of samples.
        """

        final = []
        for sample in self.p_sample_loop_condition_progressive(
            model,
            shape,
            kspace,
            mask,
            coil_map,
            model_kwargs=model_kwargs,
            device=device
        ):
            final.append(sample)
        return final

    def create_mask(self, shape, m_og):
        if self.data_type == "singlecoil":
            m_og = m_og[0,0].cpu().detach().numpy().astype('i1')
        elif self.data_type == "multicoil":
            m_og = m_og[0,0,0].cpu().detach().numpy().astype('i1')

        rows = m_og.shape[0]
        cols = m_og.shape[1]

        num_points = rows * cols - np.sum(m_og)

        mask = np.ones((rows, cols))
        M = np.ones((self.num_timesteps + 1, rows, cols))

        r_start = np.max((rows, cols)) / 2
        if self.data_type == "singlecoil":
            r_end = 3
        elif self.data_type == "multicoil":
            r_end = 4

        T = self.num_timesteps
        for t in range(self.num_timesteps):
            r = r_start - t * (r_start - r_end) / self.num_timesteps
            p = int(num_points / T)
            for i in range(p):
                x = np.random.randint(rows)
                y = np.random.randint(cols)
                while mask[x,y] == 0 or m_og[x,y] == 1 or (x-rows/2)**2 + (y-cols/2)**2 < r**2:
                    x = np.random.randint(rows)
                    y = np.random.randint(cols)

                mask[x,y] = 0

            M[t+1] = mask
            num_points -= p
            T -= 1

        return M

    def p_sample_loop_condition_progressive(
        self,
        model,
        shape,
        kspace,
        mask,
        coil_map=None,
        model_kwargs=None,
        device=None
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))

        if self.data_type == "singlecoil":
            img1 = kspace[:, [0]] + kspace[:, [1]] * 1j
            img1 = self.ifft2c(img1)
        elif self.data_type == "multicoil":
            img1 = self.ifft2c(kspace)
            img1 = th.unsqueeze(th.mean(img1, dim=1), dim=1)

        img = th.cat([img1.real, img1.imag], 1).to(th.float32)

        indices = list(range(self.num_timesteps))[::-1]

        M = th.from_numpy(self.create_mask(shape, mask)).to(device)

        for i in indices:
            if i % 100 == 0:
                print('ITER:', i)

            t = th.tensor([i] * shape[0], device=device)

            with th.no_grad():
                out = self.p_sample(model, img, M, t)
                img = self.data_consistency(out, kspace, mask, coil_map)
                yield img

    def data_consistency(self, img, kspace, mask, coil_map=None):
        """
        Applies data consistency between the output image and the image prior.
        Returns mask * kspace + (1 - mask) * fft(img)
        """
        if coil_map != None:
            num_coils = coil_map.shape[1]

        padx = (kspace.shape[-2] - img.shape[-2]) // 2
        pady = (kspace.shape[-1] - img.shape[-1]) // 2

        img1 = img[:, [0]] + img[:, [1]] * 1j

        if self.data_type == "multicoil":
            img1 = th.tile(img1[:, :, padx:self.image_size-padx, pady:self.image_size-pady], dims=[1, num_coils, 1, 1])
            img1 *= coil_map
            mask = th.tile(mask[0, 0:1, :, :, :], dims=[1, num_coils, 1, 1])
        else:
            img1 = img1[:, :, padx:self.image_size-padx, pady:self.image_size-pady]
            kspace1 = kspace[:, [0]] + kspace[:, [1]] * 1j

        img1_kspace = self.fft2c(img1)

        if self.data_type == "singlecoil":
            out1 = mask[:, [0]] * kspace1 + (1 - mask[:, [0]]) * img1_kspace
            out1 = self.ifft2c(out1)
        elif self.data_type == "multicoil":
            out1 = mask * kspace + (1 - mask) * img1_kspace
            out1 = self.ifft2c(out1)
            out1 = th.sum(th.conj(coil_map) * out1, dim=1)
            out1.unsqueeze_(1)

        out = th.cat([out1.real, out1.imag], 1).to(th.float32)
        return out

    def training_losses(self, model, x_0, t, save_dir, model_kwargs=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_0: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """

        x_t = self.q_sample(x_0, t)
        model_output = model(x_t, t, **model_kwargs)
        assert model_output.shape == x_0.shape

        terms = {}
        terms["loss"] = mean_flat((x_0 - model_output) ** 2)

        return terms


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    # print(arr,timesteps)
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
