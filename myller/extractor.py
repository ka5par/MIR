import numpy as np
import os, sys, librosa, math
from matplotlib import pyplot as plt
from numba import jit

from .utils.plotting import FloatingBox

from .utils.c4 import compute_accumulated_score_matrix, \
    compute_optimal_path_family, \
    compute_induced_segment_family_coverage, \
    colormap_penalty, \
    plot_ssm_ann_optimal_path_family, \
    compute_fitness, \
    compute_tempo_rel_set, \
    compute_sm_from_filename


# //TODO get rid of overkill dependencies.
# //TODO optimize the scape plot

# Repetition based methods.
# Given a song, gives out it's most repetitive output.


def normalization_properties_ssm(S):
    """Normalizes self-similartiy matrix to fulfill S(n,n)=1
    Yields a warning if max(S)<=1 is not fulfilled

    Notebook: C4/C4S3_AudioThumbnailing.ipynb
    """
    N = S.shape[0]
    for n in range(N):
        S[n, n] = 1
        max_S = np.max(S)

    if max_S > 1:
        print('Normalization condition for SSM not fulfill (max > 1)')
    return S


@jit(nopython=True)
def compute_optimal_path_family(D):
    """Compute an optimal path family given an accumulated score matrix

    Notebook: C4/C4S3_AudioThumbnailing.ipynb

    Args:
        D: Accumulated score matrix

    Returns
        P: Optimal path family consisting of list of paths
           (each path being a list of index pairs)
    """
    # Initialization
    inf = math.inf
    N = int(D.shape[0])
    M = int(D.shape[1])

    path_family = []
    path = []

    n = N - 1
    if(D[n, M-1] < D[n, 0]):
        m = 0
    else:
        m = M-1
        path_point = (N-1, M-2)
        path.append(path_point)

    # Backtracking
    while n > 0 or m > 0:

        # obtaining the set of possible predecesors given our current position
        if(n <= 2 and m <= 2):
            predecessors = [(n-1, m-1)]
        elif(n <= 2 and m > 2):
            predecessors = [(n-1, m-1), (n-1, m-2)]
        elif(n > 2 and m <= 2):
            predecessors = [(n-1, m-1), (n-2, m-1)]
        else:
            predecessors = [(n-1, m-1), (n-2, m-1), (n-1, m-2)]

        # case for the first row. Only horizontal movements allowed
        if n == 0:
            cell = (0, m-1)
        # case for the elevator column: we can keep going down the column or jumping to the end of the next row
        elif m == 0:
            if(D[n-1, M-1] > D[n-1, 0]):
                cell = (n-1, M-1)
                path_point = (n-1, M-2)
                if(len(path) > 0):
                    path.reverse()
                    path_family.append(path)
                path = [path_point]
            else:
                cell = (n-1, 0)
        # case for m=1, only horizontal steps to the elevator column are allowed
        elif m == 1:
            cell = (n, 0)
        # regular case
        else:

            # obtaining the best of the possible predecesors
            max_val = -inf
            for i in range(len(predecessors)):
                if(max_val < D[predecessors[i][0], predecessors[i][1]]):
                    max_val = D[predecessors[i][0], predecessors[i][1]]
                    cell = predecessors[i]

            # saving the point in the current path
            path_point = (cell[0], cell[1]-1)
            path.append(path_point)

        (n, m) = cell

    # adding last path to the path family
    path.reverse()
    path_family.append(path)
    path_family.reverse()

    return path_family


def compute_induced_segment_family_coverage(path_family):
    """Compute induced segment family and coverage from path family

    Notebook: C4/C4S3_AudioThumbnailing.ipynb

    Args:
        path_family: Path family

    Returns
        segment_family: Induced segment family
        coverage: Coverage of path family
    """
    num_path = len(path_family)
    coverage = 0
    if num_path > 0:
        segment_family = np.zeros((num_path, 2), dtype=int)
        for n in range(num_path):
            segment_family[n, 0] = path_family[n][0][0]
            segment_family[n, 1] = path_family[n][-1][0]
            coverage = coverage + segment_family[n, 1] - segment_family[n, 0] + 1
    else:
        segment_family = np.empty

    return segment_family, coverage


@jit(forceobj=True)
def compute_fitness(path_family, score, N):
    """Compute fitness measure and other metrics from path family

    Notebook: C4/C4S3_AudioThumbnailing.ipynb

    Args:
        path_family: Path family
        score: Score of path family
        N: Length of feature sequence

    Returns
        fitness: Fitness
        score: Score
        score_n: Normalized score
        coverage: Coverage
        coverage_n: Normalized coverage
        path_family_length: Length of path family (total number of cells)
    """
    eps = 1e-16
    num_path = len(path_family)
    M = path_family[0][-1][1] + 1

    # Normalized score
    path_family_length = 0
    for n in range(num_path):
        path_family_length = path_family_length + len(path_family[n])
    score_n = (score - M) / (path_family_length + eps)

    # Normalized coverage
    segment_family, coverage = compute_induced_segment_family_coverage(path_family)
    coverage_n = (coverage - M) / (N + eps)

    # Fitness measure
    fitness = 2 * score_n * coverage_n / (score_n + coverage_n + eps)

    return fitness, score, score_n, coverage, coverage_n, path_family_length


def visualize_scape_plot(SP, Fs=1, ax=None, figsize=(4, 3), title='',
                         xlabel='Center (seconds)', ylabel='Length (seconds)'):
    """Visualize scape plot

    Notebook: C4/C4S3_ScapePlot.ipynb

    Args:
        SP: Scape plot data (encodes as start-duration matrix)
        Fs: Sampling rate
        ax, figsize, title, xlabel, ylabel: Standard parameters for plotting

    Returns:
        fig, ax, im
    """
    fig = None
    if(ax is None):
        fig = plt.figure(figsize=figsize)
        ax = plt.gca()
    N = SP.shape[0]
    SP_vis = np.zeros((N, N))
    for length_minus_one in range(N):
        for start in range(N-length_minus_one):
            center = start + length_minus_one//2
            SP_vis[length_minus_one, center] = SP[length_minus_one, start]

    extent = np.array([-0.5, (N-1)+0.5, -0.5, (N-1)+0.5])/Fs
    im = plt.imshow(SP_vis, cmap='hot_r', aspect='auto', origin='lower', extent=extent)
    x = np.asarray(range(N))
    x_half_lower = x/2
    x_half_upper = x/2 + N/2 - 1/2
    plt.plot(x_half_lower/Fs, x/Fs, '-', linewidth=3, color='black')
    plt.plot(x_half_upper/Fs, np.flip(x, axis=0)/Fs, '-', linewidth=3, color='black')
    plt.plot(x/Fs, np.zeros(N)/Fs, '-', linewidth=3, color='black')
    plt.xlim([0, (N-1) / Fs])
    plt.ylim([0, (N-1) / Fs])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    plt.colorbar(im, ax=ax)
    return fig, ax, im


@jit(forceobj=True)
def compute_fitness_scape_plot(S):
    """Compute scape plot for fitness and other measures

    Notebook: /C4/C4S3_ScapePlot.ipynb

    Args:
        S: Self-similarity matrix

    Returns:
        SP_all: Vector containing five different scape plots for five measures
            (fitness, score, normalized score, coverage, normlized coverage)
    """
    N = S.shape[0]
    SP_fitness = np.zeros((N, N))
    SP_score = np.zeros((N, N))
    SP_score_n = np.zeros((N, N))
    SP_coverage = np.zeros((N, N))
    SP_coverage_n = np.zeros((N, N))

    for length_minus_one in range(N):
        for start in range(N-length_minus_one):
            S_seg = S[:, start:start+length_minus_one+1]
            D, score = compute_accumulated_score_matrix(S_seg)
            path_family = compute_optimal_path_family(D)
            fitness, score, score_n, coverage, coverage_n, path_family_length = compute_fitness(path_family, score, N)
            SP_fitness[length_minus_one, start] = fitness
            SP_score[length_minus_one, start] = score
            SP_score_n[length_minus_one, start] = score_n
            SP_coverage[length_minus_one, start] = coverage
            SP_coverage_n[length_minus_one, start] = coverage_n
    SP_all = [SP_fitness, SP_score, SP_score_n, SP_coverage, SP_coverage_n]
    return SP_all


def seg_max_SP(SP, length_of_seg=None):
    """Return segment with maximal value in SP

    Notebook: C4/C4S3_ScapePlot.ipynb

    Args:
        SP: Scape plot
        length_of_seg: Length of the segment that is searched for (seconds).

    Returns:
        seg: Segment [start_index:end_index]
    """
    N = SP.shape[0]

    if length_of_seg is None:
        arg_max = np.argmax(SP)
    else:
        arg_max = np.argmax(SP[length_of_seg, :]) + (length_of_seg)*SP.shape[1]

    ind_max = np.unravel_index(arg_max, [N, N])
    seg = [ind_max[1], ind_max[1] + ind_max[0]]

    return seg


def plot_seg_in_sp(ax, seg, S=None, Fs=1):
    """Plot segment and induced segements as points in SP visualization

    Notebook: C4/C4S3_ScapePlot.ipynb

    Args:
        ax: Axis for image
        seg: Segment [start_index:end_index]
        S: Self-similarity matrix
        Fs: Sampling rate
    """
    if S is not None:
        S_seg = S[:, seg[0]:seg[1] + 1]
        D, score = compute_accumulated_score_matrix(S_seg)
        path_family = compute_optimal_path_family(D)
        segment_family, coverage = compute_induced_segment_family_coverage(path_family)
        length = segment_family[:, 1] - segment_family[:, 0] + 1
        center = segment_family[:, 0] + length // 2
        ax.scatter(center / Fs, length / Fs, s=64, c='white', zorder=9999)
        ax.scatter(center / Fs, length / Fs, s=16, c='lime', zorder=9999)
    length = seg[1] - seg[0] + 1
    center = seg[0] + length // 2
    ax.scatter(center / Fs, length / Fs, s=64, c='white', zorder=9999)
    ax.scatter(center / Fs, length / Fs, s=16, c='blue', zorder=9999)


def plot_sp_ssm(SP, seg, S, ann, color_ann=[], title='', figsize=(5, 4)):
    """Visulization of SP and SSM
    Notebook: C4/C4S3_ScapePlot.ipynb"""
    float_box = FloatingBox()
    fig, ax, im = visualize_scape_plot(SP, figsize=figsize, title=title,
                                       xlabel='Center (frames)', ylabel='Length (frames)')
    plot_seg_in_sp(ax, seg, S)
    float_box.add_fig(fig)

    penalty = np.min(S)
    cmap_penalty = colormap_penalty(penalty=penalty)
    fig, ax, im = plot_ssm_ann_optimal_path_family(
        S, ann, seg, color_ann=color_ann, fontsize=8, cmap=cmap_penalty, figsize=(4, 4),
        ylabel='Time (frames)')
    float_box.add_fig(fig)
    float_box.show()


def check_segment(seg, S):
    """Prints properties of segments with regard to SSM S

    Notebook: C4/C4S3_ScapePlot.ipynb

    Args:
        seg: Segment [start_index:end_index]
        S: Self-similarity matrix

    Returns:
        path_family: Optimal path family
    """
    N = S.shape[0]
    S_seg = S[:, seg[0]:seg[1] + 1]
    D, score = compute_accumulated_score_matrix(S_seg)
    path_family = compute_optimal_path_family(D)
    fitness, score, score_n, coverage, coverage_n, path_family_length = compute_fitness(
        path_family, score, N)
    segment_family, coverage2 = compute_induced_segment_family_coverage(path_family)
    print('Segment (alpha):', seg)
    print('Length of segment:', seg[-1] - seg[0] + 1)
    print('Length of feature sequence:', N)
    print('Induced segment path family:\n', segment_family)
    print('Fitness: %0.10f' % fitness)
    print('Score: %0.10f' % score)
    print('Normalized score: %0.10f' % score_n)
    print('Coverage: %d, %d' % (coverage, coverage2))
    print('Normalized coverage: %0.10f' % coverage_n)
    print('Length of all paths of family: %d' % path_family_length)
    return path_family


def extract(fs, length=None, save_SSM=True, save_thumbnail=True, save_wav=True, save_SP=True, output_path='output/repetition/'):
    """Prints properties of segments with regard to SSM S

    Args:
        fs: filename of the song
        length: length of the output segment (if None, find best)
        save_SSM: save self similarity matrix (normalized)
        save_thumbnail: saves the range of the thumbnail.
        save_wav: saves the waveform audio file into output/repetition
        save_SP: save the spacial plot data.

    Returns:
        path_family: Optimal path family
        :param output_path:
    """

    for fn_wav in fs:
        name = os.path.split(fn_wav)[-1][:-4]

        tempo_rel_set = compute_tempo_rel_set(0.66, 1.5, 5)

        penalty = -2
        x, _, _, _, SSM, _ = compute_sm_from_filename(fn_wav,
                                                             L=21,
                                                             H=5,
                                                             L_smooth=12,
                                                             tempo_rel_set=tempo_rel_set,
                                                             penalty=penalty,
                                                             thresh=0.15)
        # Save not normalized SSM.
        if save_SSM:
            np.save(output_path+'{}_SSM.npy'.format(name), SSM)

        SSM = normalization_properties_ssm(SSM)

        SP_all = compute_fitness_scape_plot(SSM)
        SP = SP_all[0]

        seg = seg_max_SP(SP, length_of_seg=length)

        # path_family = check_segment(seg, S)
        # print(seg)

        if save_SSM:
            np.save(output_path+'{}_SSM_norm.npy'.format(name), SSM)

        if save_SP:
            np.save(output_path+'{}_SP.npy'.format(name), SP)

        if save_thumbnail:
            np.save(output_path+'{}_seg.npy'.format(name), seg)

        if save_wav:
            librosa.output.write_wav(output_path+'{}_audio.wav'.format(name),
                                        x[seg[0] * 22050:seg[1] * 22050], 22050)


if __name__ == '__main__':
    # fs = ["data/Pink Floyd - The Great Gig in The Sky.wav", "data/FMP_C4_Audio_Beatles_YouCantDoThat.wav"]
    fs = ['data/Pink Floyd - The Great Gig in The Sky.wav']  # list
    extract(fs, length=10, save_SSM=True, save_thumbnail=True, save_wav=True, save_SP=True)
