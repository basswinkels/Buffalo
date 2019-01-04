#!/usr/bin/env python

# handling of keys and mouse: https://matplotlib.org/users/event_handling.html

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import argparse
import warnings

from scipy.interpolate import InterpolatedUnivariateSpline
from matplotlib.mlab import specgram
from tools import next_smaller, next_bigger, blockaver, sl2bitvect, expect_almost_int
from gwpy.timeseries import TimeSeries
import gwpy.segments as seg

from gwpy.time import from_gps

warnings.filterwarnings("ignore", category=RuntimeWarning)



class Points(object):
    def __init__(self, ax, color):
        self.ax = ax
        self.line, = ax.plot([], [], c=color)
        self.dots, = ax.plot([], [], marker='.', c=color, ls='none')
        self.x = []
        self.y = []
        self.order = 1
    
    def extrapolate(self, xint):
        # see https://stackoverflow.com/a/8166155
        if len(self.x) > 1:
            spline = InterpolatedUnivariateSpline(self.x, self.y, k=min(self.order, len(self.x) - 1))
            return spline(xint)
        elif len(self.x) == 1:
            return np.full_like(xint, self.y[0])
        else:
            return None  # or return nans?
    
    def sort(self):
        """sort points with increasing x"""
        i = np.argsort(self.x)
        self.x = list(np.array(self.x)[i])
        self.y = list(np.array(self.y)[i])
    
    def draw(self):
        # draw interpolated line
        ygrid = self.extrapolate(xgrid)
        if ygrid is None:
            self.line.set_data([], [])
        else:
            self.line.set_data(xgrid, ygrid)
        
        # draw dots
        self.dots.set_data(self.x, self.y)

    def clear(self):
        self.x = []
        self.y = []
        self.draw()
    
    def onclick(self, event):
        # only handle clicks in the relevant axis
        if event.inaxes is not self.ax:
            print 'outside axis'
            return
            
        # add point with left button
        if event.button == 1:
            self.x.append(event.xdata)
            self.y.append(event.ydata)
            self.sort()
            
        # delete point with right button
        if event.button == 3:
            if len(self.x) > 0:
                # event.xdata is a numpy float, so don't have to convert self.x to an array
                imn = np.argmin((event.xdata - self.x)**2 + (event.ydata - self.y)**2)
                del self.x[imn]
                del self.y[imn]
        
        self.draw()
        plt.draw()



class Veto(object):
    def __init__(self, ax):
        self.ax = ax
        self.segs = seg.SegmentList()
        self.first_x = None
        self.vline = ax.axvline(np.nan, c='k', lw=1)

    def onclick(self, event):
        # only handle clicks in the relevant axis
        if event.inaxes is not self.ax:
            # print 'outside axis'
            return

        if event.button == 1:
            if self.first_x is None:
                self.first_x = event.xdata
                self.vline.set_xdata(event.xdata)
            else:
                # todo: avoid segments with length zero, so when first_x = xdata? or handled by coalesce?
                self.segs.append(seg.Segment([self.first_x, event.xdata]))  # no need to sort points
                self.segs.coalesce()
                self.first_x = None
                self.vline.set_xdata(np.nan)
        elif event.button == 3:
            if self.first_x is None:
                click_seg = seg.Segment(event.xdata, event.xdata + data.dt.value)  # very small segment
                for iseg, segment in enumerate(self.segs):
                    if segment.intersects(click_seg):
                        del self.segs[iseg]
                        break
                    else:
                        pass #print 'Did not click any vertical bar'
            else:
                self.first_x = None
                self.vline.set_xdata(np.nan)

        update_specgram()




## parse command line

parser = argparse.ArgumentParser(description='GUI to plot spectrogram and track lines')
parser.add_argument('--input', help="input file for timeseries")

args = parser.parse_args()

# get input data

data = TimeSeries.read(args.input)  # assume single channel for now
# cannot store private attributes, get it from name
f_demod = int(data.name.split('_')[-1])
f_in = 1 / data.dt.value
twosided = np.iscomplexobj(data.value)

## do time math after f_in is known

# sample rate needs to be compatible with Virgo rates  # TODO 2**n for LIGO
allowed_f_out = [m * 10.**e for e in range(-4, 3) for m in (1, 2, 5)]

allowed_nfft = []
for f_out in allowed_f_out[::-1]:
    nfft = f_in * 2 / f_out  # factor 2 due to half-overlapping windows
    if nfft < 10:
        continue
    nfft = expect_almost_int(nfft)
    assert nfft % 2 == 0

    nwin = len(data.value) // (nfft // 2) - 1  # ignoring trimming
    if nwin < 10:
        continue
    allowed_nfft.append(nfft)

assert nfft, 'no valid nfft'

print 'allowed_nfft:', allowed_nfft

tmax = len(data) * data.dt.value
xgrid = np.linspace(0, tmax, 1000)  # for extrapolation


## prepare plots

fig, [[time_cut_ax, dummy_ax], [spec_ax, freq_cut_ax]] = plt.subplots(
    2, 2, gridspec_kw={'width_ratios': [20, 1], 'height_ratios': [1,10],
                       'hspace': 0, 'wspace': 0})
fig.tight_layout()


time_cut_ax.set_xticklabels([])
dummy_ax.axis('off')
freq_cut_ax.set_yticklabels([])

# make some dummy plots, in which the real data is later inserted
if twosided:
    extent = (0, tmax, f_demod - f_in / 2, f_demod + f_in / 2)
else:
    extent = (0, tmax, 0, f_in / 2)
spec_im = spec_ax.imshow(np.full((2, 2), np.nan), aspect='auto',
                         extent=extent, origin='lower')

spec_im.get_cmap().set_bad(color='grey')  # https://stackoverflow.com/a/46649061

time_cut, brms_dots = time_cut_ax.plot([], [], 'k', [], [], 'r')
freq_cut, = freq_cut_ax.plot([], [], 'k')
max_dots, = spec_ax.plot([], [], '.r')
hline = freq_cut_ax.axhline(np.nan, c='r', lw=1)
vline = time_cut_ax.axvline(np.nan, c='r', lw=1)



g0 = data.t0.value
spec_ax.set_xlabel('Time (s) after gps = {:d} ({:%Y-%m-%d %H:%M:%S} UTC)'.format(int(g0), from_gps(g0)))
spec_ax.set_ylabel('Frequency (Hz)')

veto = Veto(spec_ax)

points = [
    None,
    Points(spec_ax, color='r'),
    Points(spec_ax, color='lime'),
    Points(spec_ax, color='b'),
    None,
    None,
    None,
    None,
    None,
    veto,
]

Z = F = T = S = None

def update_specgram():
    global Z, F, T, S

    mask = sl2bitvect(seg.Segment(0, abs(data.span)), data.sample_rate.value, veto.segs)
    dd = data.copy()
    dd[mask] = np.nan

    S, F, T = specgram(x=dd, NFFT=nfft, Fs=f_in, noverlap=nfft // 2)

    F += f_demod
    
    # divide by median to make changes more pronounced
    # med = np.median(spec, axis=1)
    # spec /= med[:,np.newaxis]

    S = blockaver(S, navg)
    T = blockaver(T, navg)

    # sanity check
    assert np.isclose(F[1] - F[0], f_in / nfft)
    assert np.isclose(T[1] - T[0], nfft * navg / (f_in * 2))

    # prevent log(0)
    S[S <= 0] = np.nan  # this causes a warning, don't bother

    Z = np.log(S)
    # Z[np.isinf(Z)] = np.nan  # hide -inf due to log(0)

    spec_im.set_data(Z)
    spec_im.set_clim((np.nanmin(Z), np.nanmax(Z)))


# initial state
navg = 1
cid_click = None
state = 0
nfft = allowed_nfft[len(allowed_nfft) // 2]


def update_title():
    fig.suptitle('state = %i, nfft = %i, navg = %i, $\Delta t$ = %g, $\Delta f$ = %g'
                 % (state, nfft, navg, nfft * navg / (f_in * 2), f_in / nfft))


def on_key(event):
    global freq_out
    # early exit for standard shortcut keys

    if event.key in 'ophxysq':
        return

    global nfft, navg, cid_click, state
    print('you pressed', event.key)
    '''
    if state in (1,2,3):
        if event.key in '=+' and points[state].order < 5:
            points[state].order += 1
        if event.key in '-_' and points[state].order > 1:
            points[state].order -= 1
        points[state].draw()'''
    
    if event.key == ',':
        nfft = next_smaller(allowed_nfft, nfft)
        update_specgram()
    if event.key == '.':
        nfft = next_bigger(allowed_nfft, nfft)
        update_specgram()
    
    if event.key == '[' and navg > 1:
        navg -= 1
        update_specgram()
    if event.key == ']':
        navg += 1
        update_specgram()
    
    if event.key == 'd' and state:
        points[state].clear()
     
    if event.key in '01239':
        if cid_click is not None:
            fig.canvas.mpl_disconnect(cid_click)
            cid_click = None

        state = int(event.key)
        if state:
            cid_click = fig.canvas.mpl_connect('button_press_event', points[state].onclick)

    if event.key == 'm':
        if len(max_dots.get_xdata()):
            print 'removing maxima'
            max_dots.set_data([], [])
        else:
            fmin = points[2].extrapolate(T)
            if fmin is None:
                fmin = np.full_like(T, -np.inf)
            fmax = points[3].extrapolate(T)
            if fmax is None:
                fmax = np.full_like(T, np.inf)
            
            if not np.all(fmax > fmin):
                print 'line 3 must be above line 2'
                return
            
            ZZ = Z.copy()
            for itime in range(len(T)):
                ZZ[:, itime][(fmin[itime] > F) | (F > fmax[itime])] = -np.inf
            
            imx = ZZ.argmax(axis=0)
            fpeak = F[imx]
            fpeak[~np.all(np.isfinite(Z), axis=0)] = np.nan
            max_dots.set_data(T, fpeak)
    if event.key == 'M':
        if not len(max_dots.get_xdata()):
            print 'No maxima to save'
            return

        # recover data from plot
        tmax, fpeak = max_dots.get_data()
        # FIXME: shift by half delta t? specgram gives middle of bin, while frame convention os beginning of bin
        freq_out = TimeSeries(fpeak, unit='Hz', dt=tmax[1]-tmax[0], t0=data.t0.value + tmax[0])
        fname = 'spectool_freq.h5'
        print 'Saving frequency to file', fname
        freq_out.write(fname, path='peak_freq')

    
    if event.key == 'b':
        if len(brms_dots.get_xdata()):
            brms_dots.set_data([], [])
        else:
            # hide existing time cut
            time_cut.set_data([], [])
            vline.set_xdata(np.nan)

            # calculate brms
            fmin = points[2].extrapolate(T)
            if fmin is None:
                fmin = np.full_like(T, -np.inf)
            fmax = points[3].extrapolate(T)
            if fmax is None:
                    fmax = np.full_like(T, np.inf)

            if not np.all(fmax > fmin):
                print 'line 3 must be above line 2'
                return
            SS = S.copy()
            for itime in range(len(T)):
                SS[:,itime][(fmin[itime] > F) | (F > fmax[itime])] = 0

            brms = SS.sum(axis=0)**0.5

            brms = np.log(brms)  # just for plotting

            brms_dots.set_data(T, brms)
            time_cut_ax.axis(spec_ax.get_xlim() + (np.nanmin(brms), np.nanmax(brms)))

    if event.key == 'B':
        if not len(brms_dots.get_xdata()):
            print 'No brms to save'
            return

        # recover data from plot
        tbrms, brms = brms_dots.get_data()
        # FIXME: shift by half delta t? specgram gives middle of bin, while frame convention os beginning of bin
        brms_out = TimeSeries(brms, unit=data.unit, dt=tbrms[1]-tbrms[0], t0=data.t0.value + tbrms[0])
        fname = 'spectool_brms.h5'
        print 'Saving brms to file', fname
        brms_out.write(fname, path='brms')

        
    update_title()
    plt.draw()
    print 'state = %i, nfft = %i, order = %s, cid = %s' % (
        state, nfft, points[state].order if state in (1, 2, 3) else '-', cid_click)


cid_key = fig.canvas.mpl_connect('key_press_event', on_key)


def onmove(event):
    if event.inaxes is spec_ax and Z is not None:
        if len(brms_dots.get_xdata()):
            time_cut.set_data([], [])
            vline.set_xdata(np.nan)
        else:
            i_freq = np.argmin(np.abs(F - event.ydata))
            cut = Z[i_freq, :]
            time_cut.set_data(T, cut)
            time_cut_ax.axis(spec_ax.get_xlim() + (np.nanmin(Z), np.nanmax(Z)))  # fixme: use im.get_clim() of specgram
            vline.set_xdata(event.xdata)

        i_time = np.argmin(np.abs(T - event.xdata))
        cut = Z[:, i_time]
        freq_cut.set_data(cut, F)
        freq_cut_ax.axis((np.nanmin(Z), np.nanmax(Z)) + spec_ax.get_ylim())
        hline.set_ydata(event.ydata)

        plt.draw()


cid_move = fig.canvas.mpl_connect('motion_notify_event', onmove)

update_specgram()
update_title()
plt.show()

print 'Finished!'
