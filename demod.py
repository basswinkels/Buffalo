#!/usr/bin/env python

from __future__ import division

import argparse
import numpy as np

import logging
from gwpy.time import to_gps
from virgotools.frame_lib import expand_ffl


## parse command line


# assume for now that all frequencies and segment lengths are integers

parser = argparse.ArgumentParser(description='Tool to demodulate and decimate channels')
parser.add_argument('--ffl', type=expand_ffl, help="ffl file to read input from")
parser.add_argument('--chan', help="channel name, needed if source contains multiple channels")
parser.add_argument('--f_demod', required=True, type=int,  # todo: allow a list of multiple freqs
                    help="demodulation frequency, must be integer. When set to zero, the data is low-passed, not demodulated")
parser.add_argument('--f_out', required=True, type=int,
                    help="sample frequency of output signal, must be a sub-multiple of the one of the input signal")
parser.add_argument('--start', type=to_gps,
                    help='gps or utc time of start of period to process, accepts any valid input to gwpy.to_gps')
parser.add_argument('--end', type=to_gps,
                    help='gps or utc time of end of period to process, accepts any valid input to gwpy.to_gps')
# todo add duration instead of end?
# todo: add multiple channels, multiple f_demod?? YAGNI

args = parser.parse_args()

# slow imports only when parsing is successful

from gwpy.timeseries import TimeSeries
import h5py
from tools import Decimator, expect_almost_int


# hack to ensure 100 sec segments from frame
args.start = round(args.start, -2)  # todo: floor
args.end = round(args.end, -2)    # todo: ceil, then crop to span at the end


from virgotools import *


logger = logging.getLogger(__name__)
log_to_console('INFO')


# TODO: write command line, user, run time to h5 file

dur = args.end - args.start



# check that fdemod * seg_length is integer for all segsm

# check that fdemod > f_out/2, to avoid folded frequencies around DC

if args.f_demod:
    assert args.f_out < 2 * args.f_demod, \
        'output frequency must be smaller than twice the demodulation frequency'


# get a bit of data to determine f_in

with getChannel(args.ffl, args.chan, args.start, 1) as data:
    f_in = data.fsample


ratio = expect_almost_int(f_in / args.f_out)
logger.info('Decimating by a factor %i from %g to %g Hz', ratio, f_in, args.f_out)


decimator = Decimator(ratio)

tchunk = 100
lchunk = expect_almost_int(f_in * tchunk)
nchunk = int(dur // tchunk)

t = np.arange(lchunk) / f_in

if args.f_demod:
    lo = 2**.5 * np.exp(-2j * np.pi * args.f_demod * t) # local oscillator

out = []

timer = LoopTimer(nchunk, 'demodulating')
for ichunk in range(nchunk):
    with getChannel(args.chan, args.start + ichunk * tchunk, tchunk) as data:
        chunk = data.data
        if args.f_demod:
            chunk *= lo
        out.append(decimator.calc(chunk))
    timer.end(ichunk)
out = np.concatenate(out)

out_name = 'demod_%i_%s-%i-%i.h5' % (args.f_demod, args.chan, args.start, dur)
logger.info('Writing results to %s', out_name)

path = '%s_demod_%i' % (args.chan, args.f_demod)
out = TimeSeries(out, t0=args.start, dt=1.0 / args.f_out, channel=args.chan, name=path)
with h5py.File(out_name, 'w') as out_file:

    # for chan in chans: for f_demod in f_demods:


    out.write(out_file, path=path)
    dataset = out_file[path]
    # storing attrs on dateset itself breaks TimeSeries.read, use parent instead
    dataset.parent.attrs['f_demod'] = args.f_demod

logger.info('Finished!')