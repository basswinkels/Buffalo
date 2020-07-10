#!/usr/bin/env python

from __future__ import division, print_function
import argparse
from gwpy.time import to_gps

parser = argparse.ArgumentParser(description='Tool to correlate channels')
parser.add_argument('--source', required=True,
                    help="source for channel(s) for which to find correlation")
parser.add_argument('--chans', nargs='+',
                    help="channel names (can be omitted if source constains a single channel)")
parser.add_argument('--f_target', required=True, type=float,
                    help="sample frequency at which correlation is performed, ideally same as aux data")
parser.add_argument('--aux_source', required=True,
                    help="source for auxiliary channels (name of ffl for now)")
parser.add_argument('--fit_order', type=int, default=1, help="order of polynomial fit (default: %(default)s)")
parser.add_argument('--ntop', type=int, default=200, help="number of winning channels to report (default: %(default)s)")
parser.add_argument('--start', type=to_gps, default=-float('inf'),
                    help='gps time of start of period to process, accepts any valid input to gwpy.to_gps.'
                         ' Can be omitted if source is a finite timeseries')
parser.add_argument('--end', type=to_gps, default=float('inf'),
                    help='gps time of end of period to process, accepts any valid input to gwpy.to_gps'
                         ' Can be omitted if source is a finite timeseries')
# todo: blacklistfile
args = parser.parse_args()

# hardcode for now
# bl_patterns = ['*max', '*min', 'V1:VAC*', 'V1:Daq*', '*rms']
bl_patterns = ['*max', '*min', '*BRMSMon_THR*', '*_Tolm_*']

# slow imports when parsing went ok

import numpy as np
from fnmatch import fnmatch
import multiprocessing as mp
import h5py
import warnings
import signal

from virgotools import * 
from virgotools.frame_lib import expand_ffl
from gwpy.timeseries import TimeSeries
import gwpy.segments as seg
from tools import expect_almost_int, fast_resample, fast_resample_timeseries,\
    retry_on_ioerror, read_virgo_timeseries, fill_gaps


# suppress warning of ill-conditioned polyfit
warnings.simplefilter('ignore', np.RankWarning)

# utilities


def match_patterns(s, patterns):
    return any(fnmatch(s, p) for p in patterns)


log_to_console('INFO')
log_to_file('correlator', 'DEBUG')


if args.chans is None:  # get all chans in source
    if not args.source.endswith(('.h5', '.hdf5')):
        raise NotImplementedError
    with h5py.File(args.source, 'r') as f:
        chans = f.keys()
        assert len(chans) <= 50, 'too many channels in source'
    args.chans = chans


# get all data and determine union of all spans
start = float(args.start) # get rid of LIGOTimeGPS
end = float(args.end)

span = seg.Segment(start, end)
ref_series = []
for chan in args.chans:
    logger.info('Reading target channel %s', chan)
    if args.source.endswith('.ffl'):
        data = read_virgo_timeseries(args.source, chan, start, end)
    else:
        # hack, read a bit extra if possible, gets cropped later
        data = TimeSeries.read(args.source, chan, start=start-100, end=end+100)
    logger.info('data.span: %s', data.span)
    span &= data.span
    ref_series.append(data)

# coerce into target sample rate
nref = len(ref_series)
nsamp = expect_almost_int(abs(span) * args.f_target)

# todo: abort if nsamp is prohibitively large

Dref = np.full((nref, nsamp), np.nan)
good = np.full(nsamp, True)

for iref, data in enumerate(ref_series):
    data = fast_resample_timeseries(data, args.f_target)
    data = data.crop(*span)
    assert len(data.value) == nsamp
    Dref[iref, :] = data.value
    good &= np.isfinite(data.value)

if not np.all(good):
    logger.info('Input data constains gaps!')
    Dref = Dref[:, good]


args.aux_source = expand_ffl(args.aux_source)

with FrameFile(args.aux_source) as ffl:
    logger.info('Getting all channel names from %s', args.aux_source)
    with ffl.get_frame(span.start) as frame:
        names = [str(adc.contents.name) for adc in frame.iter_adc()]


# remove blacklisted channels
naux = len(names)
logger.info('Found %i channels', naux)
names = [n for n in names if not match_patterns(n, bl_patterns)]
naux = len(names)
logger.info('%i channels left after blacklisting', naux)
   

# worker function

def process_chunk(chunk_names):
    chunk_residuals = np.full((len(chunk_names), nref), np.inf)  # channels that are skipped will rank lowest

    with retry_on_ioerror(FrameFile)(args.aux_source) as ffl:
        # note that opening an ffl fails on rare occasions, probably if it happens when file is updated
        # retry in a loop if this happens?
        for ichan, name in enumerate(chunk_names):
            try:
                with ffl.getChannel(name, *span) as aux:
                    auxdata = aux.data
                    try:
                        auxdata = fast_resample(auxdata, args.f_target / aux.fsample)
                    except ValueError:
                        logger.exception('Error while resampling channel %s, skipping', name)
                        continue
                    if len(auxdata) != nsamp:
                        print('Trouble with channel %s with fsample %g, it has length %i instead of %i' % (
                            name, aux.fsample, len(auxdata), nsamp))
                        if len(auxdata) > nsamp:  # ugly hack
                            auxdata = auxdata[:nsamp]
                            print('Fixed!')
                        else:
                            print('Skipping!')
                            continue
                    auxdata = auxdata[good]
            except Exception:
                log.exception('Caught error for channel %s, skipping ...', name)
                continue


            #if not np.isfinite(auxdata).all():
            #    #print('Channel %s contains non-finite values! You might see some Intel warning' % name)
            #    logger.info('Channel %s contains non-finite values! Skipping ...', name)
            #    continue
            fill_gaps(auxdata)

            if (auxdata == auxdata[0]).all():  # skip constant channel, leave residual as inf
                continue
            auxdata -= auxdata.mean()  # make polyfit better conditioned? should not matter for correlation
            for iref in range(nref):
                # if refchans[iref] == name:
                #     continue  # skip trivial correlation with target channel itself
                dref = Dref[iref, :]
                try:
                    p = np.polyfit(auxdata, dref, args.fit_order)
                    res = ((dref - np.polyval(p, auxdata))**2).sum()
                except:
                    # saw some weird errors with linalg stuff not converging
                    # probably ill-conditioned in pathetic cases for order=2
                    # when aux channel has only 2 values
                    logger.exception('Error while fitting channel %s:', name)
                    continue
                if np.isfinite(res):
                    chunk_residuals[ichan, iref] = res
    return chunk_residuals


# chop list of channel names in chunks
chunklen = 20
chunks = [names[i:i+chunklen] for i in range(0, naux, chunklen)]

# parallel processing of chunks

ncpu = max(mp.cpu_count() - 1, 1)
logger.info('Starting processing pool with %d cpu', ncpu)

# handle CTRL-C in multiprocessing pool: https://stackoverflow.com/a/35134329
original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
pool = mp.Pool(ncpu)
signal.signal(signal.SIGINT, original_sigint_handler)

try:
    residuals = []
    timer = LoopTimer(len(chunks), 'Processing chunks', tmin=2)
    for ichunk, res in enumerate(pool.imap(process_chunk, chunks)):
        residuals.append(res)
        timer.end(ichunk)

except KeyboardInterrupt:
    print("Loop cancelled by user, terminating worker processors")
    pool.terminate()
    pool.join()
    fatal_exit()

pool.close()
pool.join()


# squeeze results of chunks together
residuals = np.concatenate(residuals)
assert residuals.shape[0] == naux
# residuals = process_chunk(names)

# write results to file

assert not np.any(np.isnan(residuals))
# residuals[~np.isfinite(residuals)] = np.inf  # NaNs screw up the sorting

outname = 'correlate_%i.txt' % now_gps()
logger.info('Writing results to file %s', outname)

with open(outname, 'w') as outfile:
    # def p(*args, **kwargs):
    #    print(*args, **kwargs, file=outfile)
    def p(s):
        outfile.write(s + '\n')
    
    p('Brute-force search for correlation from %s to %s\nOrder of fit: %i' % (
        gps2str(span.start), gps2str(span.end), args.fit_order))
    
    for iref, refchan in enumerate(args.chans):
        p('\n\n*** Best correlation for channel %s ***\n' % refchan)
        p('rank  residual  channel')
        wrap = zip(residuals[:, iref], names)
        wrap.sort()
        # for i in xrange(args.ntop):
        #     p('%4i  %.2e  %s' % (i+1, wrap[i][0], wrap[i][1]))
        for rank, (res, chan) in enumerate(wrap[:args.ntop], 1):
            p('%4i  %.2e  %s' % (rank, res, chan))

logger.info('Finished!')
