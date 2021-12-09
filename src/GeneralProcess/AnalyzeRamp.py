# tools for analyzing ramp-based protocols
# only does `ramp_dt` as of April 16, 2021

import math
import numpy as np
import pandas as pd
from collections import OrderedDict

from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline
from scipy.integrate import simps, romb, trapz

from matplotlib import rcParams
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.transforms import IdentityTransform, TransformedBbox, Bbox

cmap = plt.cm.get_cmap("gist_rainbow")


class AngleAnnotation(Arc):
    """
    https://matplotlib.org/stable/gallery/text_labels_and_annotations/angle_annotation.html
    Draws an arc between two vectors which appears circular in display space.
    """

    def __init__(self, xy, p1, p2, size=75, unit="points", ax=None,
                 text="", textposition="inside", text_kw=None, **kwargs):
        """
        Parameters
        ----------
        xy, p1, p2 : tuple or array of two floats
            Center position and two points. Angle annotation is drawn between
            the two vectors connecting *p1* and *p2* with *xy*, respectively.
            Units are data coordinates.

        size : float
            Diameter of the angle annotation in units specified by *unit*.

        unit : str
            One of the following strings to specify the unit of *size*:

            * "pixels": pixels
            * "points": points, use points instead of pixels to not have a
              dependence on the DPI
            * "axes width", "axes height": relative units of Axes width, height
            * "axes min", "axes max": minimum or maximum of relative Axes
              width, height

        ax : `matplotlib.axes.Axes`
            The Axes to add the angle annotation to.

        text : str
            The text to mark the angle with.

        textposition : {"inside", "outside", "edge"}
            Whether to show the text in- or outside the arc. "edge" can be used
            for custom positions anchored at the arc's edge.

        text_kw : dict
            Dictionary of arguments passed to the Annotation.

        **kwargs
            Further parameters are passed to `matplotlib.patches.Arc`. Use this
            to specify, color, linewidth etc. of the arc.

        """
        self.ax = ax or plt.gca()
        self._xydata = xy  # in data coordinates
        self.vec1 = p1
        self.vec2 = p2
        self.size = size
        self.unit = unit
        self.textposition = textposition

        super().__init__(self._xydata, size, size, angle=0.0,
                         theta1=self.theta1, theta2=self.theta2, **kwargs)

        self.set_transform(IdentityTransform())
        self.ax.add_patch(self)

        self.kw = dict(ha="center", va="center",
                       xycoords=IdentityTransform(),
                       xytext=(0, 0), textcoords="offset points",
                       annotation_clip=True)
        self.kw.update(text_kw or {})
        self.text = ax.annotate(text, xy=self._center, **self.kw)

    def get_size(self):
        factor = 1.
        if self.unit == "points":
            factor = self.ax.figure.dpi / 72.
        elif self.unit[:4] == "axes":
            b = TransformedBbox(Bbox.from_bounds(0, 0, 1, 1),
                                self.ax.transAxes)
            dic = {"max": max(b.width, b.height),
                   "min": min(b.width, b.height),
                   "width": b.width, "height": b.height}

            factor = dic[self.unit[5:]]
        return self.size * factor

    def set_size(self, size):
        self.size = size

    def get_center_in_pixels(self):
        """return center in pixels"""
        return self.ax.transData.transform(self._xydata)

    def set_center(self, xy):
        """set center in data coordinates"""
        self._xydata = xy

    def get_theta(self, vec):
        vec_in_pixels = self.ax.transData.transform(vec) - self._center
        return np.rad2deg(np.arctan2(vec_in_pixels[1], vec_in_pixels[0]))

    def get_theta1(self):
        return self.get_theta(self.vec1)

    def get_theta2(self):
        return self.get_theta(self.vec2)

    def set_theta(self, angle):
        pass

    # Redefine attributes of the Arc to always give values in pixel space
    _center = property(get_center_in_pixels, set_center)
    theta1 = property(get_theta1, set_theta)
    theta2 = property(get_theta2, set_theta)
    width = property(get_size, set_size)
    height = property(get_size, set_size)

    def show_angle(self):
        return (self.theta2 - self.theta1) % 360

    # The following two methods are needed to update the text position.
    def draw(self, renderer):
        self.update_text()
        super().draw(renderer)

    def update_text(self):
        c = self._center
        s = self.get_size()

        angle_span = (self.theta2 - self.theta1) % 360
        angle = np.deg2rad(self.theta1 + angle_span / 2)

        r = s / 2
        if self.textposition == "inside":
            r = s / np.interp(angle_span,
                              [60, 90, 135, 180], [3.3, 3.5, 3.8, 4])
        self.text.xy = c + r * np.array([np.cos(angle), np.sin(angle)])
        if self.textposition == "outside":
            def R90(a, r, w, h):
                if a < np.arctan(h/2/(r+w/2)):
                    return np.sqrt((r+w/2)**2 + (np.tan(a)*(r+w/2))**2)
                else:
                    c = np.sqrt((w/2)**2+(h/2)**2)
                    T = np.arcsin(c * np.cos(np.pi/2 - a + np.arcsin(h/2/c))/r)
                    xy = r * np.array([np.cos(a + T), np.sin(a + T)])
                    xy += np.array([w/2, h/2])
                    return np.sqrt(np.sum(xy**2))

            def R(a, r, w, h):
                aa = (a % (np.pi/4))*((a % (np.pi/2)) <= np.pi/4) + \
                    (np.pi/4 - (a % (np.pi/4)))*((a % (np.pi/2)) >= np.pi/4)
                return R90(aa, r, *[w, h][::int(np.sign(np.cos(2*a)))])

            bbox = self.text.get_window_extent()
            X = R(angle, r, bbox.width, bbox.height)
            trans = self.ax.figure.dpi_scale_trans.inverted()
            offs = trans.transform(((X-s/2), 0))[0] * 72
            self.text.set_position([offs*np.cos(angle), offs*np.sin(angle)])

def find_mid(T: np.ndarray, R: np.ndarray, Imax=None):
    """Return element of times `T` that corresponds to half-maximal (negative) current in `R`

    Args:
        T (np.ndarray): array of timepoints
        R (np.ndarray): array of current
        Imax (int, optional): maximal current. If 0, computed by `min(R)`. Defaults to 0.

    Returns:
        float: `i`-th element of `T` such that `R[i]` is the half-maximal change in `R`, relative to `Imax`
    """    
    if Imax is None:
        Imax = min(R)
        
    Imid = 0.5*(Imax + max(R))
    return T[ np.argmin(np.abs(R - Imid)) ]

def find_max_IV(R1: np.ndarray, R2: np.ndarray, khz: int):
    """Find maximal difference between ramp arms along voltage axis
    Currently non functional.

    Args:
        R1 (np.ndarray): current of first ramp
        R2 (np.ndarray): current of second ramp, in natural time axis
        khz (int): sampling rate

    Returns:
        float: maximum difference current between `R1` and `R2`
    """    
    # what probably needs to happen is: 
    # (1) set indices of both series with their respective voltage time series, then 
    # (2) concatenate (merging indices),
    # (3) compute maximum of column differences 
    
    df1 = pd.Series(R1).rolling(10*khz).mean()
    df2 = pd.Series(R2).rolling(10*khz).mean()
    return (df1 - df2).max()
    

def unique_legend(ax, leg_kwargs={}):
    """
    Filter out duplicate labels and handles and add to axis legend of `ax`
    """
    if isinstance(ax, list):
        leg_tuples = [a.get_legend_handles_labels() for a in ax]
        h = [t for tup in leg_tuples for t in tup[0]]
        l = [t for tup in leg_tuples for t in tup[1]]
        ax = ax[0] 
    else:
        h, l = ax.get_legend_handles_labels()

    by_lab = OrderedDict(zip(l, h))

    ax.legend(by_lab.values(), by_lab.keys(), **leg_kwargs)

def piecewise_linear(x, x0, y0, k1, k2):
    return np.piecewise(x, [x < x0], [lambda x: k1*x + y0-k1*x0, lambda x: k2*x + y0-k2*x0])

class analyze_ramp_dt():
    def __init__(self, protocol, df, tmid, ntraces, khz):
        """
        protocol = dataframe of voltage command 
        df = dataframe containing leak-subtracted data for a ramp_dt protocol 
        tmid = list of indices for middle of ramps 
        ntraces = number of traces 
        khz = sampling frequency in khz 
        """
        self.df = df
        self.tmid = tmid
        self.ntraces = int(ntraces)
        self.khz = int(khz)
        self.protocol = protocol

        rcParams['font.size'] = 12
        rcParams['font.family'] = 'Verdana'
        rcParams['font.weight'] = 'normal'
        rcParams['axes.linewidth'] = 2
        rcParams['axes.labelweight'] = 'bold'
        plt.style.use("dark_background")

    def split_ramp(self, time: np.ndarray, df: pd.DataFrame, tmid: int, flip=True):
        """
        Return ramp split at tmid and corresponding timepoints 
            time = np.array 
        Post-split section of ramp is automatically flipped unless `flip`=False

        Returns time and ramp values for hyperpolarizing (1) and depolarizing (2) arms of each ramp as numpy arrays
        """
        t1 = np.array(time[:tmid])
        t2 = np.array(time[tmid:df.shape[0]]) - time[tmid]

        df = df.values
        r1 = df[:tmid]

        if flip:
            r2 = np.flip(df[tmid:])
        else:
            r2 = df[tmid:]

        n = abs(len(t2) - len(t1))
        if n > 0:
            if len(t1) < len(t2):
                t2 = t2[:-n]
                r2 = r2[:-n]
            else:
                t1 = t1[n:]
                r1 = r1[n:]

        return t1, t2, r1, r2, time[tmid] 

    def find_RampMids(self, splitted: list,
                      relative=dict(dTmids=False, peak=False),
                      return_dTmids=True):
        """
        Find times of 50% maximal current change, a.k.a. "Tmids"

        Args:
            `splitted` = list of outputs from self.split_ramp 
            
            relative (dict): dictionary with keys "dTmids" and "peak", each with boolean values
                * if `relative['dTmids']=True`, the difference between Tmids is calculated using real times, i.e. the time of the ramp midpoint is added to the second Tmid before subtraction, and the second Tmid is then always greater than the first. 
                * if `relative['peak']=True`, then Tmids are computed by setting the maximal current to that within each ramp arm. If False, the sweep's maximal current is used instead.
            
            return_dTmids: 
                * if True, returns a 1D array corresponding to the difference [deactivation Tmid - activation Tmid]. 
                * Else, returns a 2D array (N x 2), where 
                    * N is the number of sweeps, and 
                    * the first and second columns hold Tmids for activation and deactivation, respectively.

        Returns: 
            if return_dTmids=True, 2D np.array of midpoints for each sweep where rows are sweeps and columns are ramp arms
            else, 1D array containing difference between second and first ramp arms 
        """
        # N x 2 
        Tmids = np.zeros((len(splitted), 2))
        
        # maximum current for `find_mid`, assuming relative[peak]=True
        Imax = None
        
        for i in range(len(splitted)):
            t1, t2, r1, r2 = splitted[i][:4]

            if not relative["peak"]:
                Imax = min((r1.min(), r2.min()))
                
            # reverse because r2 is flipped, and we want 
            # the time beginning from the midpoint, whereas 
            # `r2` alone would yield larger values
            Tmids[i, 1] = find_mid(t2, r2[::-1], Imax=Imax)
            Tmids[i, 0] = find_mid(t1, r1, Imax=Imax)
            
            if not relative["dTmids"]:
                # splitted[i][4] = time of the voltage midpoint,
                # i.e. addition results in 'true' time of half-maximal current, rather
                #      than the window indexed beginning *after* this time 
                Tmids[i, 1] += splitted[i][4]
                
        if return_dTmids:
            return np.abs( Tmids[:, 1] - Tmids[:, 0] )
        else:
            return Tmids

    def moving_avg(self, x, w):
        """
        Compute moving average for given window size, w, and data array `x`
        """
        return np.convolve(x, np.ones(w), 'valid') / w

    def get_int(self, dI, dx=1, t1=0):
        """
        Return area between ramp arms, with respect to variable `dx`, which is the step difference between timepoints.
        dI = difference in current between ramp arms 

        If rombs doesn't work, use simps, which requires the sampling times of the first ramp, `t1`
        If simps doesn't work, use trapz, which is the least accurate. 
        """
        # romberg method
        try:
            int_dI = romb(dI, dx=dx)
        # simpsons method
        except:
            int_dI = simps(dI, x=t1)

        if math.isnan(int_dI) or abs(int_dI) == np.inf:
            int_dI = trapz(dI, x=t1)

        return int_dI

    def create_plots(self, nrows=2, ncols=4, figsize=(14, 7)):
        fig, ax = plt.subplots(
            nrows, ncols, figsize=figsize, constrained_layout=True)

        ax[0, 0].set_title("Current (pA)")
        ax[1, 0].set_title(r"Difference in Current (pA)")
        ax[1, 0].set_xlabel("Time (s)", labelpad=12)

        ax[0, 1].set_title(r"Total $Q$ in Difference Current (nC)")
        ax[1, 1].set_xlabel("Ramp Slope (mV/s)", labelpad=12)
        ax[1, 1].set_ylabel(r"Delay in Peak (s)", fontsize=11, labelpad=10)

        ax[0, 2].set_title("Ramp Midpoints (s)")
        ax[1, 2].set_title("Difference in Ramp Midpoints (s)")
        ax[1, 2].set_xlabel("Ramp Slope (mV/s)", labelpad=12)

        ax[0, 3].set_title("Transition Potentials (mV)")
        ax[1, 3].set_title("dI/dt (pA/s)")
        ax[1, 3].set_xlabel("Ramp Slope (mV/s)", labelpad=12)

        return fig, ax

    def CollectSplitRamps(self):
        """
        Apply `self.split_ramp` to each trace in the data
        Returns 
            `split_outputs` = N-list, where each element is a 4-list 
            `split_outputs[i] = [t1, r1, t2, r2]`, 
            e.g. `t1, r1` are times and current values of the first ramp segment 
        """
        time = self.df.index.values
        time -= time[0]

        split_outputs = []
        for i in range(self.ntraces):
            y = self.df.iloc[:, i].dropna()

            # get times and current values for each ramp segment (split at the middle)
            # e.g. t1, r1 = times and current for the first segment
            t1, t2, r1, r2, t2_0 = self.split_ramp(time, y, self.tmid[i])
            split_outputs.append([t1, t2, r1, r2, t2_0])

        self.split_outputs = split_outputs
        return split_outputs

    def Hdynamic(self, x, y, v, return_slopes="fit", show_fit=False, show_slopes=False):
        """
        Get voltages of transition and mean rates for slow and fast phases 
        `return_slopes` = if 'fit', then returns slopes from piecewise linear fit,
            if 'mean', then computes the mean of time derivative (computed by spline interpolation) for the slow and fast segments. 
            'mode' and 'median' are equivalently computed on the spline derivative.
        `show_slopes` = shows histogram and plots of returned slopes compared to filtered data and fitted piecewise line  
        """
        # window for fitting polynomials in savgol filter
        w = math.floor(len(y)/8)
        # ensure window is an odd integer
        w = w if (w % 2 > 0) else w + 1
        # range of data to be filtered = from first index to peak (minimum) current
        ym = np.argmin(y)
        xm = x[:ym]
        # apply savgol fitler
        yfilt = savgol_filter(y[:ym], w, 2)

        # fit two straight lines between start and minimum of current
        popt, perr = curve_fit(piecewise_linear, xm, yfilt,
                               p0=[0.75*len(y), -20, -7e-3, -0.5],
                               bounds=([1000., -1e3, -5, -10.],
                                       [1e5, 1e3, 5., 10.])
                               )

        print(
            "Piecewise linear fit (x0, y0, m1, m2): \n    ", popt
            # popt, "\n S.D. \n", np.sqrt(np.diag(perr))
        )

        # require that ratio of fast/slow slopes >= 10
        # try:
        #     r = abs(popt[3]/popt[2])
        # except:
        #     popt[2] = 0

        if any(s == 0 for s in popt[2:]) or abs(popt[3]/popt[2]) < 5:
            print("Fast/slow ratio < 5 or zero slope. Re-fit with single line.")
            popt = np.polyfit(xm, yfilt, deg=1)

            print("Single linear fit: ", popt)
            ylin = np.polyval(popt, xm)

        else:
            # piecewise linear fit to smoothed data
            ylin = piecewise_linear(xm, *popt)

        if len(popt) > 2:
            # identify points where filtered data and linear fit deviate by < 1pA
            idx = np.argwhere(np.abs(yfilt - ylin) < 1)
            idx = xm[idx].flatten().astype(int)

            # count number of intercepts preceding intercept of linear fit
            i1 = sum(((idx < popt[0]).flatten() > 0)) - 1
            # first intercept after `i1` greater than intercept of linear fit
            i2 = i1 + \
                next((i for i, t in enumerate(idx[i1:]) if t > popt[0]), 50)

            # convert i1, i2 from indices to time values
            i1 = idx[i1]
            i2 = idx[i2]

        else:
            i1 = len(xm)
            i2 = np.nan

        if show_fit:
            f, ax = plt.subplots()
            ax.plot(x, y, c="gray", alpha=0.25, lw=1)
            ax.plot(xm, yfilt, c="y", lw=2, label="Savgol")
            ax.plot(xm, ylin, c="r", lw=2.5, ls=":", label="Fit")

            if not math.isnan(i2):
                ax.axvline(xm[i1], c="w", alpha=0.5, ls="--")
                ax.axvline(xm[i2], c="w", alpha=0.5, ls="--")

            plt.show()

        # transition potentials: low, mean, and high
        if math.isnan(i2):
            TVs = [np.nan]*3
        else:
            TVs = [v[i1], np.mean(v[i1:i2]), v[i2]]

        if return_slopes == "fit":
            if len(popt) > 2:
                slopes = popt[2:]
            else:
                slopes = [popt[0], np.nan]
        else:
            spl = UnivariateSpline(xm, yfilt, k=3)
            dy1 = spl.derivative(1)(xm)

            if show_slopes:
                # rates of current change
                f, ax = plt.subplots(2, 1)

                # show returned slopes
                f2, ax2 = plt.subplots()
                ax2.plot(xm, yfilt, lw=2)
                ax2.plot(xm, ylin, ls="--", lw=2, c="w", label="Fit")

            slopes = []
            for i, ind in enumerate([i1, i2]):

                if math.isnan(ind):
                    continue

                if i > 0:
                    dy1_i = dy1[ind:]
                else:
                    dy1_i = dy1[:ind]

                if return_slopes == "mean":
                    slopes.append(np.mean(dy1_i))
                elif return_slopes == "median":
                    slopes.append(np.median(dy1_i))
                elif return_slopes == "mid":
                    slopes.append(dy1_i[int(dy1_i.shape[0]/2)])
                else:
                    hist, bins = np.histogram(dy1_i, bins="auto")
                    ind = np.argmax(hist)
                    slopes.append(np.mean(bins[ind:ind+2]))

                if show_slopes:
                    ax[i].hist(dy1_i, bins="auto")
                    ax[i].axvline(slopes[i], label="%.3e" % ind)
                    ax[i].legend()
                    ax2.axvline(ind, c="w", ls="--", alpha=0.5, lw=2)

            if show_slopes:
                if len(slopes) > 1:
                    ylin2 = piecewise_linear(
                        xm, popt[0], popt[1], slopes[0], slopes[1])
                else:
                    popt[0] = slopes[0]
                    ylin2 = np.polyval(popt, xm)

                ax2.plot(xm, ylin2, ls="--", lw=2, c="r", label=return_slopes)
                ax2.legend()
                plt.show()

        return TVs, slopes

    def H(self, plot=False, pdf=None):
        """
        Overlap (flip) ramp arms, then either subtract current directly, and integrate areas to subtract charge  

        `plot` = bool; whether to show visualization  
        `pdf` = multipage PDF object to save figures in  

        *Returns*: `rates`, `dA`, `vA`, `dI`, `dT`, `Tmids_linear`  
        `rates` = rate of voltage ramps 
        `dA` = difference in area under each ramp (\int I dt = Q)  
        `vA` = same as above, but over the voltage axis, i.e. \int I dv = P (W)  
        `dI` = differencei n current  
        `dT` = difference in time of (negative) peak current in depolarizing ramp  
        `Tmids_linear` = difference in time for current under the ramp protocol to develop by 50% 
        """
        time = self.df.index.values
        time -= time[0]
        dt = time[1]

        dA = []     # difference in area under each ramp (\int I dt = Q)
        # same as above, but over the voltage axis, i.e. \int I dv = P (W)
        vA = []
        dI = []     # difference current between ramps
        # difference in time of (negative) peak current in depolarizing ramp
        dT = []
        dT_max = []  # time of max difference current
        rates = []  # rate of change in voltage over time, in mV/s
        TVs = []    # transition potentials, each element is a tuple, of which [low, mean, high] mV
        dIdt = []   # current slopes computed from piecewise linear fits using `self.Hdynamic`

        # total change in voltage 
        v = self.protocol.iloc[:, 0].dropna()
        vrange = math.floor(v.min() - v.max())
        
        # times and current for each ramp arm 
        split_outputs = self.CollectSplitRamps()
        
        for i in range(self.ntraces):
            # voltage protocol
            v = self.protocol.iloc[:, i].dropna().values
            rates.append( vrange / self.tmid[i] )

            # get times and current values for each ramp segment (split at the middle)
            # e.g. t1, r1 = times and current for the first segment
            t1, t2, r1, r2, _ = split_outputs[i]
            # find_max_IV(v, r1, r2)

            # transition potentials
            tv1, s1 = self.Hdynamic(t1, r1, v, show_fit=False)
            tv2, s2 = self.Hdynamic(t2, r2, v[len(t1):], show_fit=False)

            TVs.append([tv1, tv2])
            dIdt.append([s1, s2])

            # take moving average over 5ms
            dT_i = np.array([self.moving_avg(r, 5*self.khz) for r in [r1, r2]])
            # minima of each ramp segment
            dT_i = np.argmin(dT_i, axis=1) + 5*self.khz
            # dT_i.shape = (2, )
            # get delay between minima (peak current) in either ramp
            dT.append(1e-3*(dT_i[0] - dT_i[1]))

            # compute difference in current between ramp arms
            dI.append(r1 - r2)
            # rolling average over 2ms of difference current
            s = pd.Series(dI[i]).rolling(
                2*self.khz).mean().dropna().reset_index(drop=True)
            # time of max difference current (take absolute value to be polarity-invariant)
            dT_max.append(s.abs().argmax()/(self.khz*1000))

            # difference in area between arms of ramp, with respect to time
            Q = self.get_int(dI[i], dt, t1)
            # area can't be negative, so if it is, redo after flipping the sign of dI\
            if Q < 0:
                Q = self.get_int(dI[i] * -1, dt, t1)
            dA.append(Q)

            # difference in area ... with respect to voltage
            dv = v[1] - v[0]
            W = self.get_int(dI[i], dv, v[:len(dI[i])])

            if W < 0:
                W = self.get_int(dI[i] * -1, dv, v[:len(dI[i])])

            vA.append(W)

        # midpoints of ramps in seconds
        Tmids_linear = self.find_RampMids(split_outputs) / 1e3

        # save data to dataframe
        dI = pd.DataFrame(data=dI).T.apply(lambda x: pd.Series(x).dropna())
        dI.index *= self.khz

        df = pd.DataFrame(
            data={
                "dA": dA, "vA": vA,
                "dT": dT, "dT_max": dT_max,
                "dTmids": Tmids_linear,
                "TV_act": [tv[0][0] for tv in TVs],
                "TV_de": [tv[1][2] for tv in TVs],
                "dIdt_act": [x[0] for x in dIdt],
                "dIdt_de": [x[1] for x in dIdt]
            },
            index=rates
        )
        
        if plot:
            self.plot_Hstats(df, dI, pdf=pdf)
            self.plot_H_IV(df, pdf=pdf)
            
        # check for NaN values of dIdt resulting from linear fit 
        for i, col in enumerate(["dIdt_act", "dIdt_de"]):
            
            if np.nan in df.loc[:,col].iloc[0]:
                ind = abs(df.loc[:,col].iloc[0].index(np.nan) - 1)
                df.loc[:,col] = [y[ind] for y in df.loc[:,col]]
                
        return df, dI

    def plot_Hstats(self, df_stats, dI, pdf=None, fig_kw={}):
        fig, ax = self.create_plots(**fig_kw)

        df_stats.index *= 1e3
        rates = df_stats.index.values
        cols = df_stats.columns

        plot_kw = dict(marker='o', markersize=5, lw=1)

        for i in range(self.ntraces):
            t1, t2, r1, r2, _ = self.split_outputs[i]
            clr = cmap((i+1)/self.ntraces)

            # convert to seconds
            t1 = t1 * 1e-3
            t2 = t2 * 1e-3

            # raw current
            ax[0, 0].plot(t1, r1, c=clr, lw=1)
            ax[0, 0].plot(t2, r2, c=clr, lw=1, alpha=0.5)

            # difference current
            ax[1, 0].plot(t1, dI.iloc[:, i].dropna(), c=clr, lw=1)

        # integrated charge
        ax[0, 1].plot(rates, df_stats.loc[:, "dA"]/1e6,  **plot_kw)

        # difference in delay to peak current
        ax[1, 1].plot(df_stats.loc[:, "dT"], label="Delay", **plot_kw)

        # time of max difference current
        ax1_1_b = ax[1, 1].twinx()
        ax1_1_b.plot(df_stats.loc[:, "dT_max"], marker="x",
                     markersize=6, lw=1, c='y', label=r"Peak dI")
        ax1_1_b.set_ylabel(r"Time of Peak $\Delta$I (s)",
                           fontsize=10, rotation=-90, labelpad=12)

        # legend for delay and time of max current
        h, l = ax[1, 1].get_legend_handles_labels()
        h1, l1 = ax1_1_b.get_legend_handles_labels()
        ax[1, 1].legend(h+h1, l+l1, fontsize=10)

        # width of ramp midpoints between arms
        # Tmids = df_stats.loc[:, ["Tmids_act", "Tmids_de"]]
        Tmids = df_stats.loc[:, 'dTmids']
        # difference between the above between arms = hpol - depol
        ax[1, 2].plot(rates, Tmids, **plot_kw)

        # transition potentials
        TVs = df_stats.loc[:, ["TV_act", "TV_de"]]

        # twin axes for difference in transition potentials
        # check that no columns are all NaN
        # if TVs.dropna(how="any", axis=0).shape[0] > 0:
        #     ax_tv_b = ax[0, 3].twinx()
        #     ax_tv_b.plot(TVs.iloc[:, 0] - TVs.iloc[:, 1], c="r", lw=1)
        #     ax_tv_b.set_ylabel(r"$\mathbf{\Delta}$ (mV)", c="r",
        #                        fontsize=12, rotation=0)

        # current slope / voltage slope
        dIdt = df_stats.loc[:, ["dIdt_act", "dIdt_de"]]

        # check that no columns are all NaN
        if dIdt.dropna(how="any", axis=0).shape[0] > 0:
            # twin plot for fast/slow ratio
            ax_dIdt_b = ax[1, 3].twinx()
            ax_dIdt_b.set_ylabel("Fast/Slow", fontsize=10, labelpad=12, rotation=-90)

            ax[1, 3].set_ylabel("Fast (pA/s)", fontsize=10, labelpad=12)

        for i, s in enumerate(["Hpol.", "Depol."]):
            clr = cmap((i+1)/2)
            if len(Tmids.shape) > 1:
                ax[0, 2].plot(Tmids.iloc[:, i], 
                              c=clr, label=s, **plot_kw)
                
            ax[0, 3].plot(TVs.iloc[:, i], c=clr, label=s, **plot_kw)

            # separate [slow, fast] slopes into separate columns
            dIdt_i = pd.DataFrame(dIdt.iloc[:, i].tolist(), index=dIdt.index)

            if dIdt_i.iloc[:, 1].dropna().shape[0] > 0:
                # plot fast slope if two slopes are available
                ax[1, 3].plot(dIdt_i.iloc[:, 1]*1e3, c=clr,
                              label="Fast", **plot_kw)

                # compute fast/slow ratio
                dIdt_i = dIdt_i.iloc[:, 1] / dIdt_i.iloc[:, 0]
                ax_dIdt_b.plot(dIdt_i, c=clr, ls="--",
                               label="Ratio", **plot_kw)

            else:
                # only one column is all non-NaN if single linear fit
                ax[1, 3].plot(
                    dIdt_i.iloc[:, 0]*1e3, c=clr, label="Fast",
                    fillstyle="none", **plot_kw
                )

        ax[0, 2].legend(handlelength=0.5, fontsize=10)
        ax[0, 3].legend(handlelength=0.5, fontsize=10)

        if dIdt.dropna(how="any", axis=0).shape[0] > 0:
            unique_legend(
                [ax[1, 3], ax_dIdt_b],
                leg_kwargs={"markerscale": 0, "fontsize": 10}
            )
        else:
            unique_legend(ax[1, 3], leg_kwargs={
                          "markerscale": 0, "fontsize": 10})

        # nbins of xtick labels for summary stats w.r.t. time
        # for i in range(2):
        #     for j in range(1, 4):
        #         ax[i,j].locator_params(axis='y', nbins=4)
        #         ax[i,j].locator_params(axis='x', nbins=5)

        #     # show at least 4 labels on the current vs. time plots
        #     ax[i, 0].locator_params(axis='x', nbins=5)

        fig.suptitle("Hysteresis Summary")
        # fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        if pdf is not None:
            pdf.savefig(bbox_inches='tight')
        plt.show()
        plt.close()

    def plot_H_IV(self, df, pdf=None):
        """
        Hysteresis plots on current-voltage data 
        """
        rates = df.index.values

        # current - voltage plots
        fig_H, ax_H = plt.subplots(
            1, 2, figsize=(11, 4), constrained_layout=True)
        ax_H[0].set_xlabel("Voltage (mV)", labelpad=12)
        ax_H[1].set_xlabel("Rate (mV/s)", labelpad=12)
        ax_H[0].set_ylabel("Current (pA)", labelpad=12)
        ax_H[1].set_ylabel("Power (pW)", labelpad=12)

        # plot voltage vs current
        for i in range(self.ntraces):
            clr = cmap((i+1)/self.ntraces)

            # current values for ith trace
            y = self.df.iloc[:, i].dropna().values
            # voltage
            v = self.protocol.iloc[:, i].dropna().values

            # plot voltage vs current
            ax_H[0].plot(v, y, c=clr, lw=1.5, label="%.1f" % rates[i])

            # difference current
            # ax_H[0].plot(v[:len(dI[i])], dI[i], c=clr, lw=1,
            #               alpha=0.8, label="%.1f" % rates[i] )

        # divide watts by 1e6 = femtoW -> pW
        ax_H[1].plot(rates, df.loc[:, "vA"]/1e3,
                     marker='o', markersize=5, lw=1, label="")
        ax_H[1].locator_params(axis='y', nbins=5)
        # ax_H[1].locator_params(axis='y', nbins=5)

        ax_H[0].legend(loc='lower right', title="Rate (mV/s)")
        ax_H[0].locator_params(axis='y', nbins=5)
        ax_H[0].locator_params(axis='x', nbins=6)
        # fig_H.tight_layout()

        if pdf is not None:
            pdf.savefig(bbox_inches='tight')

        plt.show()
        plt.close()

    def H_de(self, plot=False, pdf=None, filter=False):
        """
        Hysteresis analysis for `ramp_de` protocol, comparing varying-slope deactivating ramps with the constant-duration and constant-voltage activating prepulses.  

        Arguments are the same as in `analyze_ramp_dt.H(...)`
        """
        raise Exception("`H_de` is not implemented yet.")

    def compare_dt_de(self, df_de):
        """
        Compare `ramp_dt` and `ramp_de` recordings from the same cell.  
        `df_de` = dataframe containing `ramp_de` test pulses  
        """
        raise Exception("`compare_dt_de` is not implemented yet.")
