#!/usr/bin/env python

from __future__ import division

import argparse
import numpy as np

import logging
from gwpy.time import to_gps
from gwpy.timeseries import TimeSeries
import h5py
from virgotools.frame_lib import expand_ffl
from tools import Decimator, expect_almost_int


## parse command line


# assume for now that all frequencies and segment lengths are integers

parser = argparse.ArgumentParser(description='Tool to demodulate and decimate channels')
parser.add_argument('--ffl', type=expand_ffl, help="ffl file to read input from")
# source_group = parser.add_mutually_exclusive_group(required=True)
# source_groupp.add_argument(--timeseries)
parser.add_argument('--chan', help="channel name, needed if source contains multiple channels")
parser.add_argument('--f_demod', required=True, type=int,  # todo: allow a list of multiple freqs
                    help="demodulation frequency, must be integer")
parser.add_argument('--f_out', required=True, type=int,
                    help="sample frequency of output signal, must be a sub-multiple of the one of the input signal")
parser.add_argument('--start', type=to_gps,
                    help='gps or utc time of start of period to process, accepts any valid input to gwpy.to_gps')
parser.add_argument('--end', type=to_gps,
                    help='gps or utc time of end of period to process, accepts any valid input to gwpy.to_gps')
# todo add duration instead of end?
# todo: add multiple channels, multiple f_demod?? YAGNI
# todo: just decimate if f_demod=0?

args = parser.parse_args()

# slow imports only when parsing is succesful


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

lo = 2**.5 * np.exp(-2j * np.pi * args.f_demod * t) # local oscillator

out = []

timer = LoopTimer(nchunk, 'demodulating')
for ichunk in range(nchunk):
    with getChannel(args.chan, args.start + ichunk * tchunk, tchunk) as data:
        chunk = data.data
        # chunk = x[ichunk * lchunk:(ichunk+1) * lchunk]
        # todo: skip this to do simple decimation when fdemod == 0?
        out.append(decimator.calc(chunk * lo))
    timer.end(ichunk)
out = np.concatenate(out)

out_name = 'demod_%s_%i.h5' % (args.chan, args.f_demod)
logger.info('Writing results to %s', out_name)

path = '%s_demod_%i' % (args.chan, args.f_demod)
out = TimeSeries(out, t0=args.start, dt=1.0 / args.f_out, channel=args.chan, name=path)
with h5py.File(out_name, 'w') as out_file:

    # for chan in chans: for f_demod in f_demods:


    out.write(out_file, path=path)
    # out_file[path].attrs['f_demod'] = args.f_demod  # breaks TimeSeries.read, doesn't like unknown attrs
    # store it in the name for now


logger.info('Finished!')