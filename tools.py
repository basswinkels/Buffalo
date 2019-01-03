from __future__ import division
import scipy.signal as sig
import numpy as np
from bisect import bisect_left, bisect_right
from gwpy.timeseries import TimeSeries
import gwpy.segments as seg
import time

import logging

logger = logging.getLogger(__name__)

# from PySpectral

factors = [10, 5, 2]
filt_order = 8


def factorize(n, factors):
    result = []
    rest = n
    for fact in factors:
        while rest % fact == 0:
            rest //= fact
            result.append(fact)
    if rest != 1:
        raise ValueError('Could not factorize %i using %s' % (n, factors))
    return result


class Decimator(object):
    """decimator that keeps state and handles large ratios with multiple passes"""

    # calculate filter coefs only once
    BA = dict((fact, sig.cheby1(filt_order, 0.05, 0.8 / fact))
              for fact in factors)

    def __init__(self, ratio):
        self.states = None
        self.ratios = factorize(ratio, factors)

    def calc(self, idata):
        """decimates one chunck of data"""
        if self.states is None:
            self.states = [idata[0] * sig.lfilter_zi(*self.BA[ratio])
                           for ratio in self.ratios]
        odata = idata
        for i, ratio in enumerate(self.ratios):
            b, a = self.BA[ratio]
            odata, self.states[i] = sig.lfilter(b, a, odata, zi=self.states[i])
            odata = odata[::ratio].copy()
            # force copy, otherwise a view keeps alive big original
        return odata

    def reset(self):
        """reset filters"""
        self.states = None


def expect_almost_int(x, error_msg="expected something close to integer"):
    """converts a float to an int and raises an error (with optional message) in case it is not close to integer"""
    rx = int(round(x))
    if not np.isclose(x, rx):
        raise ValueError(error_msg)
    return rx

# see bisect documentation
def next_smaller(a, x):
    """get first value in a smaller than x or the minimum"""
    i = bisect_left(a, x)
    if i:
        return a[i - 1]
    return a[0]


def next_bigger(a, x):
    """get first value in a larger than x or the maximum"""
    i = bisect_right(a, x)
    if i == len(a):
        return a[-1]
    return a[i]

def read_virgo_timeseries(source, channel, gstart, gstop_or_dur):
    """quick and dirty function to read virgo data as timeseries. this should one day be included in gwpy"""

    from virgotools import getChannel
    with getChannel(source, channel, gstart, gstop_or_dur) as data:
        return TimeSeries(data.data, unit=data.unit, t0=gstart, dt=data.dt, channel=channel)



def blockaver(x, n, axis=-1):
    """do n-sample averaging along given axis. If the length along the axis is not an exact multiple of n,
    the remaining values are ignored"""

    # shortcut for common case
    if x.ndim == 1:
        imx = n * (x.size // n)
        return x[:imx].reshape(-1, n).mean(axis=1)

    # move relevant axis to end
    x = np.swapaxes(x, axis, -1)

    # chop off extra bits
    imx = n * (x.shape[-1] // n)
    x = x[..., :imx]

    # add extra dimension to the end and take average along fast one
    x = x.reshape(x.shape[:-1] + (-1, n))
    x = x.mean(axis=-1)

    # move relevant axis back to original place
    x = np.swapaxes(x, axis, -1)

    return x


def sl2bitvect(known_seg, fs, active_segs):
    t0 = known_seg[0]
    n = expect_almost_int(abs(known_seg) * fs)
    bit_vect = np.zeros(n, dtype=bool)
    for active_seg in seg.SegmentList([known_seg]) & active_segs:  # & only works between two segmentlists
        istart = int(np.ceil((active_seg[0] - t0) * fs))
        istop = int(np.ceil((active_seg[1] - t0) * fs))
        bit_vect[istart:istop] = 1
    return bit_vect

def StateVect2ts(known_segs, fs, active_segs):
    assert len(known_segs) == 1  # for now
    bit_vects = []

    for known_seg in known_segs:
        t0 = known_seg[0]
        n = expect_almost_int(abs(known_seg) * fs)
        bit_vect = np.zeros(n, dtype=bool)
        for active_seg in seg.SegmentList([known_seg]) & active_segs:  # & only works between two segmentlists
            istart = int(np.ceil((active_seg[0] - t0) * fs))
            istop = int(np.ceil((active_seg[1] - t0) * fs))
            bit_vect[istart:istop] = 1
        bit_vects.append(bit_vect)
    return bit_vects


def fast_resample(x, ratio):
    """crude up/down sampling of signals without anti-alias

    ratio should be f_new / f_old, so if it is larger than 1 the signal will be upsampled,
    while smaller than 1 means downsampling. resampling is done without anti-alias filtering,
    which is fast and causes no transients, but might suffer from aliasing
    """

    if ratio > 1:  # upsample
        ratio = expect_almost_int(ratio, 'upsample ratio is not integer')
        x = np.repeat(x, ratio)

    elif ratio < 1:  # downsample
        ratio = expect_almost_int(1.0 / ratio, 'downsample ratio is not integer')
        x = blockaver(x, ratio)

    return x


def fast_resample_timeseries(ts, f_new):
    ratio = f_new / ts.sample_rate.value
    if np.isclose(ratio, 1):
        return ts  # shortcut

    data = fast_resample(ts.value, ratio)
    return TimeSeries(data, unit=ts.unit, t0=ts.t0, sample_rate=f_new,
                      channel=ts.channel, name=ts.name)


def poly2latex(p, var):
    assert len(p) >= 2
    rp = p[::-1]
    result = '$\mathtt{%.2e} + \mathtt{%.2e} * %s' % (rp[0], rp[1], var)
    result = ' + '.join([result] + ['\mathtt{%.2e} * %s^{%i}' % (ki, var, i) for i, ki in enumerate(rp[2:], 2)])
    result = result.replace('_', '\_')
    return result + '$'

"""

from argparse import ArgumentParser

def add_timeseries_arguments(parser, prefix=''):
    # type: (ArgumentParser, str) -> None

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--%ssource' % prefix)
    group.add_argument('--%schan' % prefix)


def get_timeseries(source, chan, start, stop):
    if source.endswith('.ffl'):
        assert chan is not None
        assert start is not None
        assert stop is not None
        return read_virgo_timeseries(source, chan, start, stop)
    else:
        return TimeSeries.read(source, xx)"""

NRETRY = 20
TSLEEP = 5
def retry_on_ioerror(fun):
    # does not work as decorator on multiprocessing worker function
    # see https://stackoverflow.com/a/8805244
    def wrapper(*args, **kwargs):
        for i in range(NRETRY):
            try:
                return fun(*args, **kwargs)
            except IOError as err:
                if i < NRETRY - 1:
                    logger.warning('caught an IOError for function %s, retrying after %.1f second',
                                   fun, TSLEEP)
                    time.sleep(TSLEEP)
                    continue
                else:
                    logger.warning('caught %d IOErrors for function %s, giving up',
                                   NRETRY, fun)
                    raise

    return wrapper