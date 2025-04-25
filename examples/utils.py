import numpy as np
from context import mvtomo


def random_box(shape, seed=234597):
    """Fills a volume dataset with a hollow box phantom

    :param vd: `ts.Data`
        A volume dataset.
    :returns:
    :rtype: ts.Data

    """
    retval = np.zeros(shape, dtype="float32")

    s20 = [s * 20 // 100 for s in shape]
    s40 = [s * 40 // 100 for s in shape]

    bbox_slices = tuple(slice(a, l - a) for (a, l) in zip(s20, shape))
    sbox_slices = tuple(slice(a, l - a) for (a, l) in zip(s40, shape))

    retval[bbox_slices] = np.random.random(size=retval[bbox_slices].shape)
    retval[sbox_slices] = np.random.random(size=retval[sbox_slices].shape) * 0.1

    return retval


def bin_ndarray(ndarray, new_shape, operation="sum"):
    """
    Bins an ndarray in all axes based on the target shape, by summing or
        averaging.

    Number of output dimensions must match number of input dimensions and
        new axes must divide old ones.

    Example
    -------
    >>> m = np.arange(0,100,1).reshape((10,10))
    >>> n = bin_ndarray(m, new_shape=(5,5), operation='sum')
    >>> print(n)

    [[ 22  30  38  46  54]
     [102 110 118 126 134]
     [182 190 198 206 214]
     [262 270 278 286 294]
     [342 350 358 366 374]]

    """
    operation = operation.lower()
    if not operation in ["sum", "mean", "max"]:
        raise ValueError("Operation not supported.")
    if ndarray.ndim != len(new_shape):
        raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape, new_shape))
    compression_pairs = [(d, c // d) for d, c in zip(new_shape, ndarray.shape)]
    flattened = [l for p in compression_pairs for l in p]
    ndarray = ndarray.reshape(flattened)
    for i in range(len(new_shape)):
        op = getattr(ndarray, operation)
        ndarray = op(-1 * (i + 1))
    return ndarray


def getphantom(L=128, upscale=1, n_angles=90, z_order=2):
    angles = np.linspace(0, np.pi, n_angles)
    N = upscale * L

    vol_shape = (N, N, N)
    height = N
    det_width = int(1.5 * N)
    y_shape = {
        0: (len(angles), det_width, height),
        2: (height, len(angles), det_width),
    }

    phantom = np.log(2) / L * random_box(vol_shape)
    sino = np.zeros(y_shape[z_order])

    A = mvtomo.get_operator(phantom, sino, angles)
    sino = A(phantom, out=sino)

    # return to original scale
    phantom_shape = [l // upscale for l in phantom.shape]
    sino_shape = [l // upscale for l in sino.shape]
    sino_shape[1] = n_angles
    phantom = bin_ndarray(phantom, phantom_shape, operation="mean")
    sino = bin_ndarray(sino, sino_shape, operation="mean") / upscale

    return phantom, sino, angles


def getphantom(L=128, upscale=1, n_angles=90, z_order=2):
    angles = np.linspace(0, np.pi, n_angles)
    N = upscale * L

    vol_shape = (N, N, N)
    height = N
    det_width = int(1.5 * N)
    y_shape = {
        0: (len(angles), det_width, height),
        2: (height, len(angles), det_width),
    }

    phantom = np.log(2) / L * random_box(vol_shape)
    sino = np.zeros(y_shape[z_order])

    A = mvtomo.get_operator(phantom, sino, angles)
    sino = A(phantom, out=sino)

    # return to original scale
    phantom_shape = [l // upscale for l in phantom.shape]
    sino_shape = [l // upscale for l in sino.shape]
    sino_shape[1] = n_angles
    phantom = bin_ndarray(phantom, phantom_shape, operation="mean")
    sino = bin_ndarray(sino, sino_shape, operation="mean") / upscale

    return phantom, sino, angles


def getphantom_mv(L=128, upscale=2, n_angles_scout=10, n_angles_interior=90, z_order=2):
    N = upscale * L
    angles_scout = np.linspace(0, np.pi, n_angles_scout)
    angles_interior = np.linspace(0, np.pi, n_angles_interior)

    vol_shape = (N, N, N)
    height = N
    full_width = int(1.5 * N)
    roi_width = int(0.5 * N)

    y_full_shape = {
        0: (len(angles_scout), full_width, height),
        2: (height, len(angles_scout), full_width),
    }
    y_roi_shape = {
        0: (len(angles_interior), roi_width, height),
        2: (height, len(angles_interior), roi_width),
    }

    phantom = np.log(2) / L * random_box(vol_shape)
    sino = np.zeros(y_full_shape[z_order])
    sino_roi = np.zeros(y_roi_shape[z_order])

    A = mvtomo.get_operator(phantom, sino, angles_scout)
    A_roi = mvtomo.get_operator(phantom, sino_roi, angles_interior)

    sino = A(phantom, out=sino)
    sino_roi = A_roi(phantom, out=sino_roi)
    print(sino.shape)
    print(sino_roi.shape)

    # return to original scale
    phantom_shape = [l // upscale for l in phantom.shape]
    sino_shape = [l // upscale for l in sino.shape]
    sino_roi_shape = [l // upscale for l in sino_roi.shape]
    sino_shape[1] = n_angles_scout
    sino_roi_shape[1] = n_angles_interior
    phantom = bin_ndarray(phantom, phantom_shape, operation="mean")
    sino = bin_ndarray(sino, sino_shape, operation="mean") / upscale
    sino_roi = bin_ndarray(sino_roi, sino_roi_shape, operation="mean") / upscale

    return phantom, sino, sino_roi, angles_scout, angles_interior


def test_adjoint(x, y, A):
    x = np.random.rand(*x.shape).astype(np.float32)
    y = np.random.rand(*y.shape).astype(np.float32)
    Ax = A(x, out=np.zeros_like(y))
    ATy = A.T(y, out=np.zeros_like(x))

    lhs = np.vdot(Ax, y)
    rhs = np.vdot(x, ATy)

    # Calculate relative difference
    avg = (abs(lhs) + abs(rhs)) / 2
    rel_diff = abs(lhs - rhs) / avg

    print(f"    <Ax, y> = {lhs:.4e}")
    print(f"    <x, A*y> = {rhs:.4e}")
    print(f"    Relative difference: {rel_diff:.4e}")
