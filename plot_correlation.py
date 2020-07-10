#!/usr/bin/env python

from __future__ import division, print_function
import argparse
from gwpy.time import to_gps, from_gps

parser = argparse.ArgumentParser(description='Tool to plot correlation')
parser.add_argument('--source', required=True,
                    help="source for channel for which to find correlation")
parser.add_argument('--chan', help="channel name (can be omitted if source constains single channel)")
parser.add_argument('--target_name', default='target',
                    help="name in plot for target channel")
parser.add_argument('--f_target', required=True, type=float,
                    help="sample frequency at which correlation is performed, ideally same as aux data")
parser.add_argument('--aux_source', required=True,
                    help="source for auxiliary channel (name of ffl for now)")
parser.add_argument('--aux_chan', required=True,
                    help="auxiliary channel")
parser.add_argument('--fit_order', type=int, default=1)
parser.add_argument('--start', type=to_gps, default=-float('inf'),
                    help='gps time of start of period to process, accepts any valid input to gwpy.to_gps.'
                         ' Can be omitted if source is a finite timeseries')
parser.add_argument('--end', type=to_gps, default=float('inf'),
                    help='gps time of end of period to process, accepts any valid input to gwpy.to_gps'
                         ' Can be omitted if source is a finite timeseries')

args = parser.parse_args()

import numpy as np
import h5py
import matplotlib.pyplot as plt

from virgotools import *
from gwpy.timeseries import TimeSeries
import gwpy.segments as seg
from tools import expect_almost_int, poly2latex, fast_resample


if args.chan is None:
    if not args.source.endswith(('.h5', '.hdf5')):
        raise NotImplementedError
    with h5py.File(args.source, 'r') as f:
        chans = f.keys()

    if len(chans) != 1:
        raise ValueError('Source does not contain exactly 1 channel, specify channel')
    args.chan = chans[0]


# get target data

span = seg.Segment(float(args.start), float(args.end))  # get rid of LIGOTimeGPS

data = TimeSeries.read(args.source, args.chan, start=float(args.start), end=float(args.end))
span &= data.span
data = data.crop(*span)
nsamp = expect_almost_int(abs(span) * args.f_target)

dref = data.value

dref = fast_resample(dref, args.f_target / data.sample_rate.value)
good = np.isfinite(dref)


# aux = read_virgo_timeseries(args.aux_source, args.aux_chan, *span)
aux = getChannel(args.aux_source, args.aux_chan, *span)
daux = aux.data
daux = fast_resample(daux, args.f_target / aux.fsample)


p = np.polyfit(daux[good], dref[good], args.fit_order)

t = np.arange(nsamp) / args.f_target

if False:
    plt.plot(daux, dref, '.b', daux, np.polyval(p, daux), 'r')
    #plt.xlabel(aux.name)
    #plt.ylabel(data.name)
    plt.show()

if False:
    plt.hexbin(daux, dref) #, cmap='hot')
    plt.gca().set_autoscale_on(False)  # freeze axis
    xgrid = np.linspace(aux.data.min(), aux.data.max(), 1000)
    plt.plot(xgrid, np.polyval(p, xgrid), 'r', lw=3)
    plt.xlabel(aux.name)
    plt.ylabel(data.name)
    plt.title('2D histogram and fit of order %i' % args.fit_order)
    plt.show()

if True:
    target_name =  'target'  # args.target_name[3:]  # 'line frequency'
    plt.plot(t, dref, '.', t, np.polyval(p, daux))
    plt.legend((target_name, args.aux_chan))
    plt.xlabel('Time (s) starting from gps = %i' % span[0])
    plt.ylabel('%s (%s)' % (target_name, data.unit))
    plt.title('%s = %s' % (target_name, poly2latex(p, args.aux_chan[3:])))
    plt.show()