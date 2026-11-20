"""Convert IXI brain .nii.gz volumes to <C>_<R>_multi_synth_recon_<split>.mat

IXI provides magnitude images only, so k-space is simulated from the image (single coil, no coil maps).
Each output holds (h5py axis order, complex stored as MATLAB v7.3 'single' structs):
  images_fs (N,H,W)   fully sampled image x, max-normalised per subject (real, stored complex)
  images_us (N,H,W)   under-sampled image                IFFT(M * FFT(x))
  map       (N,H,W)   centred binary k-space mask, one mask per subject
A <name>_subjects.txt sidecar lists the IXI subject IDs in the order they appear in the file.

Subjects are partitioned into train/val/test once (fixed --split_seed) over the subjects that have every
contrast in --require, so the same subject IDs fall in the same split for T1, T2 and PD.
Slices are axial: volumes are reoriented to RAS, in-plane pixels made square, anterior up.

Example:
  python convert_ixi_to_mat.py /data/IXI out/ --contrast T2 --split train --accel 4
"""
import argparse, glob, os, re, time
import numpy as np
import h5py
import nibabel as nib
from scipy.ndimage import zoom

CONTRASTS = {'T1': (256, 256), 'T2': (256, 256), 'PD': (256, 256)}
MASKS = ('random', 'poisson', 'uniform', 'gaussian1d')
ACS = 40                                    # fully sampled centre (lines / square side)
CDT = np.dtype([('real', '<f4'), ('imag', '<f4')])
NAME = re.compile(r'(IXI\d+-[A-Za-z]+-\d+)-(T1|T2|PD)\.nii(\.gz)?$')

fft2c = lambda x: np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x, axes=(-2, -1)), norm='ortho'), axes=(-2, -1))
ifft2c = lambda x: np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x, axes=(-2, -1)), norm='ortho'), axes=(-2, -1))


def fit(x, H, W):
    """Centre zero-pad, then centre-crop, the last two axes to (H, W)."""
    ph, pw = max(H - x.shape[-2], 0), max(W - x.shape[-1], 0)
    x = np.pad(x, [(0, 0)] * (x.ndim - 2) + [(ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2)])
    h, w = x.shape[-2:]
    return x[..., (h - H) // 2:(h - H) // 2 + H, (w - W) // 2:(w - W) // 2 + W]


def n_axial(path):
    """Number of axial slices, read from the header only."""
    img = nib.load(path)
    ornt = nib.orientations.io_orientation(img.affine)
    return img.shape[int(np.flatnonzero(ornt[:, 0] == 2)[0])]


def load_axial(path, nslices):
    """Central axial slices as (nslices, Y, X): RAS-reoriented, square in-plane pixels, anterior up."""
    img = nib.as_closest_canonical(nib.load(path))
    sx, sy = img.header.get_zooms()[:2]
    z0 = img.shape[2] // 2 - nslices // 2
    vol = img.get_fdata(dtype=np.float32)[:, :, z0:z0 + nslices]               # (X, Y, nslices)
    s = min(sx, sy)
    if abs(sx - sy) > 1e-3:                                                    # e.g. IXI T1 (sagittal acq.)
        vol = zoom(vol, (sx / s, sy / s, 1), order=1)
    return np.ascontiguousarray(np.rot90(vol, axes=(0, 1)).transpose(2, 0, 1))


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
    p.add_argument('in_dir', help='folder (searched recursively) with IXI*-T1/T2/PD.nii.gz files')
    p.add_argument('out_dir')
    p.add_argument('--contrast', choices=CONTRASTS, required=True)
    p.add_argument('--split', choices=('train', 'val', 'test'), required=True)
    p.add_argument('--nsub', type=int, nargs=3, default=(25, 5, 10), metavar=('TRAIN', 'VAL', 'TEST'),
                   help='subjects per split')
    p.add_argument('--accel', type=int, required=True)
    p.add_argument('--mask', choices=MASKS, default='random')
    p.add_argument('--nslices', type=int, default=100, help='central axial slices kept per subject')
    p.add_argument('--require', nargs='+', choices=CONTRASTS, default=list(CONTRASTS),
                   help='use only subjects having all these contrasts (keeps splits identical across contrasts)')
    p.add_argument('--split_seed', type=int, default=0, help='seed for the train/val/test subject partition')
    p.add_argument('--seed', type=int, default=0, help='seed for the masks')
    a = p.parse_args()

    H, W = CONTRASTS[a.contrast]
    require = sorted(set(a.require) | {a.contrast})
    paths = {}
    for f in glob.glob(os.path.join(a.in_dir, '**', 'IXI*.nii*'), recursive=True):
        m = NAME.match(os.path.basename(f))
        if m:
            paths.setdefault(m[1], {})[m[2]] = f
    subjects = [s for s in sorted(paths)
                if all(c in paths[s] and n_axial(paths[s][c]) >= a.nslices for c in require)]

    ntr, nva, nte = a.nsub
    assert ntr + nva + nte <= len(subjects), \
        f'only {len(subjects)} usable subjects with {"/".join(require)} in {a.in_dir}'
    lo, hi = {'train': (0, ntr), 'val': (ntr, ntr + nva), 'test': (ntr + nva, ntr + nva + nte)}[a.split]
    order = np.random.default_rng(a.split_seed).permutation(len(subjects))
    subs = [subjects[k] for k in sorted(order[lo:hi])]

    out = os.path.join(a.out_dir, f'{a.contrast}_{a.accel}_multi_synth_recon_{a.split}.mat')
    os.makedirs(a.out_dir, exist_ok=True)
    N, ns, rng = len(subs) * a.nslices, a.nslices, np.random.default_rng(a.seed)

    with h5py.File(out, 'w', userblock_size=512) as o:
        mk = lambda name, dt: o.create_dataset(name, (N, H, W), dt, compression='gzip')
        d = {'images_fs': mk('images_fs', CDT), 'images_us': mk('images_us', CDT), 'map': mk('map', np.float32)}
        for v in d.values():
            v.attrs['MATLAB_class'] = np.bytes_('single')
        cplx = lambda z: np.ascontiguousarray(z, np.complex64).view(CDT)

        for i, s in enumerate(subs):
            x = fit(load_axial(paths[s][a.contrast], ns), H, W)                 # (ns, H, W)
            x = x / max(float(x.max()), 1e-8)
            mask = make_mask(a.mask, a.accel, H, W, rng)
            sl = slice(i * ns, (i + 1) * ns)
            d['images_fs'][sl] = cplx(x)
            d['images_us'][sl] = cplx(ifft2c(fft2c(x) * mask))
            d['map'][sl] = np.broadcast_to(mask, (ns, H, W))
            print(f'[{i + 1}/{len(subs)}] {s}  R={mask.size / mask.sum():.2f}', flush=True)
    write_mat_header(out)
    with open(out[:-4] + '_subjects.txt', 'w') as f:
        f.write('\n'.join(subs) + '\n')
    print('wrote', out)


if __name__ == '__main__':
    main()
