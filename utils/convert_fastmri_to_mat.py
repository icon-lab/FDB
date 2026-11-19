"""Convert fastMRI brain multicoil .h5 files to <C>_under_sampled_<R>x_multicoil_<split>[_variant].mat

Each output holds (h5py axis order, complex stored as MATLAB v7.3 'single' structs):
  images_fs (N,H,W)      coil-combined fully sampled image  sum_c conj(S_c) x_c
  images_us (Nc,N,H,W)   under-sampled coil images          IFFT(M * FFT(x_c))
  coil_maps (Nc,N,H,W)   ESPIRiT maps, sum_c |S_c|^2 = 1 inside support, 0 outside
  map       (N,H,W)      centred binary k-space mask, one mask per subject

Example:
  python convert_fastmri_to_mat.py /data/multicoil_val out/ --contrast T2 --split val --nsub 10 --accel 4
"""
import argparse, glob, os, time
import numpy as np
import h5py

CONTRASTS = {'T1': ('AXT1', (320, 256)), 'T1POST': ('AXT1POST', (320, 256)),
             'T2': ('AXT2', (384, 288)), 'FLAIR': ('AXFLAIR', (320, 256))}
MASK_SUFFIX = {'random': '', 'poisson': '_poisson', 'uniform': '_uniform', 'gaussian1d': '_1DGaussian16'}
ACS = 40                                    # fully sampled centre (lines / square side)
CDT = np.dtype([('real', '<f4'), ('imag', '<f4')])

fft2c = lambda x: np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x, axes=(-2, -1)), norm='ortho'), axes=(-2, -1))
ifft2c = lambda x: np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x, axes=(-2, -1)), norm='ortho'), axes=(-2, -1))


def crop(x, H, W):
    h, w = x.shape[-2:]
    return x[..., (h - H) // 2:(h - H) // 2 + H, (w - W) // 2:(w - W) // 2 + W]


def compress(x, nc):
    """SVD coil compression. x: (Nc_in,H,W) -> (nc,H,W)."""
    X = x.reshape(len(x), -1)
    _, U = np.linalg.eigh(X @ X.conj().T)
    return (U[:, ::-1][:, :nc].conj().T @ X).reshape(nc, *x.shape[1:])


def espirit(x, calib=24, k=6, thresh=0.02, crop_thr=0.95):
    """ESPIRiT sensitivity maps (first eigenmap). x: (Nc,H,W) coil images."""
    nc, H, W = x.shape
    C = crop(fft2c(x), calib, calib)
    A = np.lib.stride_tricks.sliding_window_view(C, (k, k), axis=(1, 2))      # (nc, n, n, k, k)
    A = A.transpose(1, 2, 0, 3, 4).reshape(-1, nc * k * k)
    _, s, Vh = np.linalg.svd(A, full_matrices=False)
    V = Vh.conj().T[:, s >= thresh * s[0]]                                     # (nc*k*k, n)
    ker = np.zeros((nc, V.shape[1], H, W), np.complex64)
    ker[:, :, H // 2 - k // 2:H // 2 + k // 2, W // 2 - k // 2:W // 2 + k // 2] = \
        V.reshape(nc, k, k, -1).transpose(0, 3, 1, 2)
    G = fft2c(ker[..., ::-1, ::-1].conj()) * np.sqrt(H * W) / k                # (nc, n, H, W)
    G = G.transpose(2, 3, 0, 1)                                                # (H, W, nc, n)
    w, u = np.linalg.eigh(G @ G.conj().swapaxes(-1, -2))
    maps = u[..., -1] * (w[..., -1:] > crop_thr)                               # (H, W, nc)
    maps = maps * np.exp(-1j * np.angle(maps[..., :1]))                        # phase relative to coil 0
    return maps.transpose(2, 0, 1).astype(np.complex64)


def solve(f, target, lo, hi, it=40):
    """Bisection for f(p) == target, f increasing in p."""
    for _ in range(it):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) < target else (lo, mid)
    return (lo + hi) / 2


def make_mask(kind, R, H, W, rng):
    y, x = np.mgrid[:H, :W]
    r = np.hypot((y - H // 2) / (H / 2), (x - W // 2) / (W / 2))               # elliptical radius, 1 at edges
    acs = (abs(y - H // 2 + 0.5) < ACS / 2) & (abs(x - W // 2 + 0.5) < ACS / 2)
    if kind == 'random':                                                       # 2D Gaussian variable density
        sig = solve(lambda s: np.maximum(np.exp(-r ** 2 / (2 * s ** 2)), acs).sum(), H * W / R, 1e-3, 10)
        m = rng.random((H, W)) < np.exp(-r ** 2 / (2 * sig ** 2))
    elif kind == 'gaussian1d':                                                 # 1D lines along W, 16 centre lines
        d = np.arange(W) - W // 2
        c = abs(d + 0.5) < 8
        sig = solve(lambda s: np.maximum(np.exp(-d ** 2 / (2 * s ** 2)), c).sum(), W / R, 1e-3, 10 * W)
        while True:                                                            # redraw until exactly W/R lines
            lines = (rng.random(W) < np.exp(-d ** 2 / (2 * sig ** 2))) | c
            if lines.sum() == round(W / R):
                return np.broadcast_to(lines, (H, W)).astype(np.float32)
    elif kind == 'uniform':                                                    # regular grid, steps a*b = R
        a = max(i for i in range(1, int(np.sqrt(R)) + 1) if R % i == 0)
        m = (y % a == 0) & (x % (R // a) == 0)
    elif kind == 'poisson':                                                    # variable-density Poisson disc
        order = rng.permutation(H * W)

        def draw(scale):
            g = np.zeros((H, W), bool)
            rad = scale * r
            for i in order:
                py, px = divmod(i, W)
                q = int(np.ceil(rad[py, px]))
                ys, xs = slice(max(py - q + 1, 0), py + q), slice(max(px - q + 1, 0), px + q)
                if q <= 1 or not (g[ys, xs] & ((y[ys, xs] - py) ** 2 + (x[ys, xs] - px) ** 2 < rad[py, px] ** 2)).any():
                    g[py, px] = True
            return g

        scale = solve(lambda s: -draw(s).sum(), -H * W / R, 0, 20, it=12)
        m = draw(scale)
    return (m | acs).astype(np.float32)


def write_mat_header(path):
    txt = 'MATLAB 7.3 MAT-file, Platform: GLNXA64, Created on: %s HDF5 schema 1.00 .' % time.strftime('%a %b %d %H:%M:%S %Y')
    with open(path, 'r+b') as f:
        f.write(txt.ljust(116).encode() + b'\x00' * 8 + b'\x00\x02IM')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('in_dir', help='folder with fastMRI brain multicoil .h5 files')
    p.add_argument('out_dir')
    p.add_argument('--contrast', choices=CONTRASTS, required=True)
    p.add_argument('--split', required=True, help='train / val / test')
    p.add_argument('--nsub', type=int, required=True, help='number of subjects (dataset uses 100/10/40)')
    p.add_argument('--accel', type=int, required=True)
    p.add_argument('--mask', choices=MASK_SUFFIX, default='random')
    p.add_argument('--ncoils', type=int, default=5)
    p.add_argument('--nslices', type=int, default=10, help='central slices kept per subject')
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()

    acq, (H, W) = CONTRASTS[a.contrast]
    files = []
    for f in sorted(glob.glob(os.path.join(a.in_dir, '*.h5'))):
        with h5py.File(f, 'r') as h:
            S, nc_in, kx, ky = h['kspace'].shape
            if h.attrs.get('acquisition') == acq and S >= a.nslices and nc_in >= a.ncoils and kx >= H and ky >= W:
                files.append(f)
        if len(files) == a.nsub:
            break
    assert len(files) == a.nsub, f'only {len(files)} usable {acq} volumes in {a.in_dir}'

    suffix = MASK_SUFFIX[a.mask] + (f'_{a.ncoils}_coils' if a.ncoils != 5 else '')
    out = os.path.join(a.out_dir, f'{a.contrast}_under_sampled_{a.accel}x_multicoil_{a.split}{suffix}.mat')
    os.makedirs(a.out_dir, exist_ok=True)
    N, nc, rng = a.nsub * a.nslices, a.ncoils, np.random.default_rng(a.seed)

    with h5py.File(out, 'w', userblock_size=512) as o:
        mk = lambda name, shape, dt: o.create_dataset(name, shape, dt, compression='gzip')
        d = {'images_fs': mk('images_fs', (N, H, W), CDT), 'images_us': mk('images_us', (nc, N, H, W), CDT),
             'coil_maps': mk('coil_maps', (nc, N, H, W), CDT), 'map': mk('map', (N, H, W), np.float32)}
        for v in d.values():
            v.attrs['MATLAB_class'] = np.bytes_('single')
        cplx = lambda z: np.ascontiguousarray(z, np.complex64).view(CDT)

        for i, f in enumerate(files):
            with h5py.File(f, 'r') as h:
                s0 = h['kspace'].shape[0] // 2 - a.nslices // 2
                ksp = h['kspace'][s0:s0 + a.nslices]                            # (nslices, Nc_in, kx, ky)
            mask = make_mask(a.mask, a.accel, H, W, rng)
            img = crop(ifft2c(ksp), H, W)
            for j in range(a.nslices):
                x = compress(img[j], nc)
                S_ = espirit(x)
                n = i * a.nslices + j
                d['images_fs'][n] = cplx((S_.conj() * x).sum(0))
                d['images_us'][:, n] = cplx(ifft2c(fft2c(x) * mask))
                d['coil_maps'][:, n] = cplx(S_)
                d['map'][n] = mask
            print(f'[{i + 1}/{len(files)}] {os.path.basename(f)}  R={mask.size / mask.sum():.2f}', flush=True)
    write_mat_header(out)
    print('wrote', out)


if __name__ == '__main__':
    main()