import numpy as np


def bin_idx(*dims, bin_size: float = 1) -> tuple:
    """
    Parameters
    ----------
    *dims : list or tuple
        The values in each dimension.
    bin_size : float
        The size of the hyperbin.

    Returns
    -------
    binned : tuple
        A tuple containing the integer bin for each dimension.
    """
    return tuple([int(np.floor((1.0 / bin_size) * x)) for x in dims])


def flow_field(
    embedding: np.ndarray, seq_shapes: list, n_bins: int = 8, normalize: bool = True
) -> np.ndarray:
    """Calculate a flow field from the trajectories in the embedding.

    Parameters
    ----------
    embedding : np.ndarray (N, 2)
        The two-dimensional embedding.
    seq_shapes : list
        The shapes of each sequence in the embedding.
    n_bins : int
        The number of bins per unit of the embedding.
    normalize : bool
        A flag to normalize vector lengths to 1.

    Returns
    -------
    xyuvs : np.ndarray (N, 5)
        An array to construct a quiver plot. xy are the centres of each vector.
        uv are the directions of each vector. s is the number of individual
        vectors in the bin.


    Notes
    -----
    The trajectory interpolation to find the bins crossed is very naive. This
    could be a much smarter algorithm (e.g. Bresenham's line algorithm) or a
    line-box intersection algorithm as used in ray tracing.
    """

    if embedding.shape[-1] != 2:
        raise ValueError("Only 2D embeddings are supported.")

    bin_size = 1.0 / n_bins

    vectors = {}

    for i in range(len(seq_shapes)):

        s = slice(sum(seq_shapes[:i]), sum(seq_shapes[: i + 1]), 1)
        xy = embedding[s, :].reshape(-1, 1, 2)
        segments = np.hstack([xy[:-1], xy[1:]])

        # follow the segment and update bins that are crossed
        for s in segments:

            # NOTE(arl) this is a really crude way to do this
            ix = np.linspace(s[0, 0], s[1, 0], 100)
            iy = np.linspace(s[0, 1], s[1, 1], 100)

            dx = s[1, 0] - s[0, 0]
            dy = s[1, 1] - s[0, 1]

            bins_crossed = [bin_idx(i, j, bin_size=bin_size) for i, j in zip(ix, iy)]
            bins_crossed = list(set(bins_crossed))

            for b in bins_crossed:
                if b not in vectors.keys():
                    vectors[b] = [[dx, dy]]
                else:
                    vectors[b].append([dx, dy])

    xyuvs = []

    for k, v in vectors.items():
        xy = np.array([ki * bin_size for ki in k])
        uv = np.sum(np.stack(v, axis=0), axis=0)
        if normalize:
            uv = uv / np.linalg.norm(uv, ord=1)
        s = np.array([len(v)])
        xyuvs.append(np.concatenate([xy, uv, s]))

    return np.stack(xyuvs, axis=0)
