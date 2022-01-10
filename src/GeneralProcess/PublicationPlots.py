import pandas as pd 
import numpy as np 
import math 
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.offsetbox import AnchoredOffsetbox
from matplotlib.backends.backend_pdf import PdfPages

cmap = plt.cm.get_cmap("gist_rainbow")

def Set_RC_Defaults(pub=False):
    """
    Set rcParams 
    
    `pub` = if True, sets defaults for publication standards; else, whatever is convenient 
    """
    plt.style.use("default")
    
    rcParams['axes.labelweight'] = 'bold' 
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.weight'] = 'normal'
    
    if pub:
        rcParams['font.size'] = 14
        rcParams['axes.linewidth'] = 1
        rcParams['font.sans-serif'] = 'Arial'
        rcParams['svg.fonttype'] = 'none'
        rcParams['pdf.use14corefonts'] = True

    else:
        rcParams['font.size'] = 12 
        rcParams['axes.linewidth'] = 2
        rcParams['font.sans-serif'] = 'Verdana'
    
def in_to(dims, scale=1):
    """
    dims is a tuple in inches (x, y)
    
    `scale` = scale factor, 
    e.g. if scale = 1e-3,
        1e-3 * cm/in * in = mm 
    """
    s = scale/2.54
    
    if isinstance(dims, tuple):
        return (dims[0]/s, dims[1]/s)
    else:
        return dims/s 
    
def remove_spines(ax, which=[]):
    """
    Remove spines in `which` from `ax`
    `which` = 'left', 'right', 'bottom', 'top', or 'all'
    """
    if len(which) > 0:
        if "all" in which:
            which = ['left', 'right', 'bottom', 'top']
        
        for side in which:
            ax.spines[side].set_visible(False)      
        
    # return ax 
    
def modify_constrained_layout(fig, paddict):
    """
    Apply padding to figure with a constrained layout.
    
    `fig` = figure 
    `pads` = dictionary containing one or more of ['w_pad', 'h_pad', 'wspace', 'hspace'] as keys, with floats as values 
    """
    fig.set_constrained_layout_pads(**paddict)
class AnchoredScaleBar(AnchoredOffsetbox):
    def __init__(self, transform, origin=(0, 0), 
                flipx=False, flipy=False, sizex=0, sizey=0, 
                loc=4, pad=0.1, borderpad=0.2, sep=2, 
                labelx=None, labely=None, label_fontsize=12, 
                xtextprop={}, ytextprop={},
                prop=None, barcolor="black", barwidth=2, 
                **kwargs):
        """
        Draw a horizontal and/or vertical  bar with the size in data coordinate
        of the give axes. A label will be drawn underneath (center-aligned).
        
        From: https://gist.github.com/dmeliza/3251476
        
        - transform : the coordinate frame (typically axes.transData)
        - origin : origin of scalebars, e.g. (0, 0) for bottom left, (1, 1) for top right
        - flipx, flipy : whether to flip scalebar orientation along x or y axes 
        - sizex,sizey : width of x,y bar, in data units. 0 to omit
        - labelx,labely : labels for x,y bars; None to omit
        - loc : position in containing axes
        - pad, borderpad : padding, in fraction of the legend font size (or prop)
        - sep : separation between labels and bars in points.
        - **kwargs : additional arguments passed to base class constructor
        """
        from matplotlib.patches import Rectangle
        from matplotlib.offsetbox import AuxTransformBox, VPacker, HPacker, TextArea, DrawingArea
        bars = AuxTransformBox(transform)
        if sizex:
            if flipx:
                rect = Rectangle(origin, -sizex, 0, ec=barcolor, lw=barwidth, fc="none")
            else:
                rect = Rectangle(origin, sizex, 0, ec=barcolor, lw=barwidth, fc="none")
            
            bars.add_artist(rect)
            
        if sizey:
            if flipy:
                rect = Rectangle(origin, 0, -sizey, ec=barcolor, lw=barwidth, fc="none")
            else:
                rect = Rectangle(origin, 0, sizey, ec=barcolor, lw=barwidth, fc="none")
                
            bars.add_artist(rect)

        if sizex and labelx:
            self.xlabel = TextArea(labelx, minimumdescent=False, textprops=xtextprop)
            bars = VPacker(children=[bars, self.xlabel], align="center", pad=0, sep=sep)
        if sizey and labely:
            self.ylabel = TextArea(labely, textprops=ytextprop)
            bars = HPacker(children=[self.ylabel, bars], align="center", pad=0, sep=sep)

        AnchoredOffsetbox.__init__(self, loc, pad=pad, borderpad=borderpad,
                                   child=bars, prop=prop, frameon=False, **kwargs)
        
def add_scalebar(ax, matchx=True, matchy=True, hidex=True, hidey=True, **kwargs):
    """ 
    Add scalebars to axes
    Adds a set of scale bars to *ax*, matching the size to the ticks of the plot
    and optionally hiding the x and y axes
    
    From: https://gist.github.com/dmeliza/3251476 
    
    - ax : the axis to attach ticks to
    - matchx,matchy : if True, set size of scale bars to spacing between ticks
                    if False, size should be set using sizex and sizey params
    - hidex,hidey : if True, hide x-axis and y-axis of parent
    - **kwargs : additional arguments passed to AnchoredScaleBars
    
    Returns created scalebar object
    """
    def f(l):
        """
        `l` = location of major ticks 
        """
        return len(l)>1 and (l[1] - l[0])
    
    if matchx:
        kwargs['sizex'] = f(ax.get_xticks())
        kwargs['labelx'] = str(kwargs['sizex'])
    if matchy:
        kwargs['sizey'] = f(ax.get_yticks())
        kwargs['labely'] = str(kwargs['sizey'])
        
    sb = AnchoredScaleBar(ax.transData, **kwargs)
    ax.add_artist(sb)

    if hidex : ax.xaxis.set_visible(False)
    if hidey : ax.yaxis.set_visible(False)
    if hidex and hidey: ax.set_frame_on(False)

    return sb
    
def FindScaleBarLengths(ax, frac=0.25):
    """
    Find reasonable scale bar lengths as a fraction `frac` of current x- and y-ranges 
    `ax` = axis object
    `frac` = fraction of axis (in data coordinates) that scale bar will represent
    """
    
    def IntegerScale(s):
        # extract from tuple `s` 
        lo, hi  = s 
        
        # choose scalebars that are 25% of total range 
        d = (hi - lo) * frac 
                
        # order of `d` as integer power of 10 
        e = int( 10 ** (math.floor(math.log10(d))) )

        if e > 0:
            return int(d/e)*e 
        elif int(d) > 0:
            return d 
        else:
            return int(math.ceil(d))

    return IntegerScale(ax.get_xlim()), IntegerScale(ax.get_ylim())

def add_annotation(ax, text, xy, coords='data', dx=0, dy=0, 
                   offset=50, arrows=True, arrow_kwargs={}, **kwargs):
    """
        Modifies `ax` in place with annotation 
        
        `ax` = axis 
        `text` = text 
        `xy` = (x, y) bottom/left position of annotation 
        `coords` = coordinate system. Default is 'data', i.e. follows data plotted in `ax`
        `dx` = if greater than 0, arrows will extend between `(x, y)` and `(x + dx, y)`. Text will be placed at `(x + dx/2 - offset, y)`
        `dy` = if greater than 0, arrows will extend between `(x, y)` and `(x, y + dy)`. Text will be placed at `(x, y + dy/2 - offset)`
        `offset` = offset for position of `text`. If less than 1, then it will be used as a fraction of `dx` or `dy`
        `arrows` = whether to use arrows or not 
        `kwargs` = font properties to ax.text 
    """
    
    # default appearance of labelling box 
    textprops = {'color': 'black', 'ha': 'center', 'va': 'center', 
                'bbox': dict(boxstyle="round", fc="white", ec="black", pad=0.3)}
    
    # if 0 <= offset <= 1:
    #     if dx > 0: 
    #         offset *= dx 
    #     elif dy > 0: 
    #         offset *= dy 
    
    if kwargs:
        for key in kwargs.keys():
            textprops[key] = kwargs[key]
    
    if dx > 0:
        xy = (xy[0], xy[1])
        xytext = (xy[0] + dx, xy[1])
        xylab = (xy[0] + dx/2 - offset, xy[1])
    elif dy > 0:
        xy = (xy[0], xy[1])
        xytext = (xy[0], xy[1] + dy)
        xylab = (xy[0], xy[1] + dy/2 - offset)
    else:
        raise Exception("At least one of `dx` or `dy` must be non zero.")
    
    # arrows that extend outwards from label below
    if arrows:
        arrowprops = dict(arrowstyle="<|-|>")
        
        if arrow_kwargs:
            for k in arrow_kwargs.keys():
                arrowprops.update({k : arrow_kwargs[k]})
                
        ax.annotate("", xy=xy, xytext=xytext, xycoords=coords, arrowprops=arrowprops)

    # label and box 
    ax.text(xylab[0], xylab[1], text, textprops)

class VoltageProtocolLabels():
    
    def __init__(self, protocol=None, label_info=None, n_sweeps=None, khz=2,
                t_buffer=0, units={}):
        """
        `protocol` = dataframe containing truncated protocol 
        `label_info` = dictionary containing (start time, duration, level) for each epoch 
        `khz` = sampling rate
        `n_sweeps` = number of sweeps 
        `threshold` = threshold for label offsets, in mV, or units of y-axis in `ax` 
        `t_buffer` = time offset for labels, in units of x-axis of `ax`
        `units` = units of data 
        """
        
        self.label_info = label_info
        
        self.N = n_sweeps 
        self.df = protocol 
        self.khz = khz 
        
        self.t_buffer = t_buffer
        self.units = units  
    
    def IsOverlap(self, t, v, threshold, max_iter=20):
        """
        Check whether there are voltages in `threshold` of `v` at time `t` in the voltage protocol.
        `ref` = reference, i.e. voltage protocol
        `t_buffer` = if greater than zero, voltages within `t_buffer * self.khz` of `t` are considered, rather than just the voltages at `t`
        """
        
        t_buffer = self.t_buffer
        
        if self.units['x'] == 's':
            t = int(t * 1000*self.khz)
        else:
            t = int(t * self.khz)
            
        i = 0
            
        if t_buffer > 0:
            t_buffer *= self.khz 
            
            # to account for 'horizontal' (i.e. along time axis) overlap, consider 100ms around `t`
            voltages = self.df.iloc[(t - t_buffer):(t + t_buffer), :]
            dv = threshold - (voltage - v).abs().min().min()
            
            while ((voltages - (v + dv)).abs() < threshold).any().any():
                dv += 0.1*threshold
                i += 1 

                if i > max_iter:
                    break  
            
            return dv 
                
        else:
            voltages = self.df.iloc[t, :]
            dv = threshold - (voltages - v).abs().min()
            
            while ((voltages - (v + dv)).abs() < threshold).any():
                dv += 0.1*threshold 
                i += 1 
                
                if i > max_iter:
                    break 
            
            # print(v, dv, (voltages - (v + dv)).abs())
            return dv
                
    def HandleOverlap(self, t, v, dv, max_iter=20):
        """
        Add voltage offset (increment = `dv`) until overlap is below threshold `self.threshold`
        """
        
        dv_ = 0             
        iter_ = 0
                
        while self.IsOverlap(t, v + dv_):
            dv_ += dv 
            iter_ += 1 
            
            if iter_ > max_iter:
                print("`HandleOverlap reached maximum iterations. \
                    \n The procedure will be repeated by assigning the opposite sign \
                    to the voltage offset.")
        
        if self.IsOverlap(t, v + dv_):
            # reset variables 
            dv_ = 0             
            iter_ = 0
            
            # evaluate overlap with reversed sign of `dv_`
            while self.IsOverlap(t, v - dv_):
                dv_ -= dv 
                iter_ += 1 
                
                if iter_ > max_iter:
                    print("`HandleOverlap` reached max iterations after flipping the sign of `dv`. \
                        \n Removing overlap at the given coordinates may require more iterations.")
            return dv_ 
        else:
            return dv_     
    
    def AddOffsetLabel(self, ax, t, v, offset, t_swap=None, kwargs={}, max_dv=20):
        """
        Wraps `IsOverlap` and `HandleOverlap`
        `kwargs` = kwargs for `ax.text`
        `t_swap` = if initial evaluation of IsOverlap() is true, evaluate `HandleOverlap` at time `t_swap` rather than `t`
        """
        
        dv_ = self.IsOverlap(t, v, offset)

        if (dv_ is None) or (dv_ > max_dv):
            if t_swap is not None:
                # reverse vertical alignment 
                if kwargs['va'] == 'bottom':
                    kwargs['va'] = 'top'
                elif kwargs['va'] == 'top':
                    kwargs['va'] = 'bottom'
                    
                dv_ = self.IsOverlap(t_swap, v, offset)
                
                if (dv_ is None) or (dv_ > max_dv):
                    print("Offset location failed with `t` and `t_swap`. Maybe lower threshold?")
                else:
                    ax.text(t_swap, v + dv_, "%d" % v, **kwargs)
            else:
                print("Offsetting failed with `t`. Maybe provide `t_swap`.")    
        else:
            ax.text(t, v + dv_, "%d" % v, **kwargs)
            
    def AddTimeOffset(self, ts, vs, dt):
        
        for i, v in enumerate(vs):
            if v == 0:
                ts[i] += 0.5*dt[i] 
            else:
                k = 0.53 - math.floor(math.log10(abs(v)))*0.04
                    
                ts[i] += k*dt[i]
                
        return ts 
            
class PubPlotting():
    def __init__(self, df, intervals, filename, khz=2, 
            save_path=None, pdfs=None, pub=False, linew=None, 
            dims=(), units={"x" : "s", "y_i" : "pA", "y_v" : "mV"}, 
            scalebar_frac=0.25, show_pro_sb=False, 
            paddict={}, inner_paddict={},
            show_zero_current=False, show_zero_voltage=False,
            fig_style=None, show=True, pub_bounds=True,
            annotation_kwargs = {'to_label' : 'x', 'multiple' : False}, leaksub=True 
        ):
        """
        Make figures from ephys files. 
        
        Arguments provided at call to `PubPlotting()`
        `df` = dataframe containing current and voltage time courses 
        `filename` = filename 
        `intervals` = dictionary containing epochs for each trace in the file, e.g. intervals[self.k_][0] = first epoch of first sweep (in samples, not time units)
        `khz` = sample frequency in khz 
        `pdfs` = PdfPages object; if not None, figures will be appended to `pdfs` 
        
        Arguments [that can be] provided at call to `process()` 
        `save_path` = where figures will be saved, defaults to None (no saving) 
        `pub` = whether to use publication standards for rcParams. See `Set_RC_Defaults`
        `linew` = linewidth 
        `dims` = dimension of figure, only applied if `pub = True`. Otherwise, `dims = (12, 5.5)`
        `units` = dictionary of units for x-, current y-, and voltage y-axes 
        `scalebar_frac` = fraction of axes in data coords covered by scalebars
            - if `float`, horizontal and vertical scalebars will use the same fraction 
            - if `tuple`, specifies fractions of (horizontal, vertical) scalebars separately 
        `show_pro_sb` = whether to show protocol scalebars, default False. Only applies if other arguments enable protocol plotting.
        `paddict` = padding for subplots; dictionary with one or more of `[w_pad, h_pad, hspace, wspace]` as keys. 
        `inner_paddict` = inner padding (i.e. within the plot, not between plots) dictionary with one or more of `[w_pad, h_pad]` as keys, which will take values that are fractions of the current axis 
        `show_zero_current` = whether to show zero current line as gray dashes 
        `show_zero_voltage` = whether to show zero voltage line as gray dashes 
        `fig_style` = string containing one or more of the following, separated by underscores:
            - 'nopro' = don't plot voltage protocol
            - 'labpro' = add labels to voltage protocol 
            - 'labprobox' = add box to labels in voltage protocol, only applied if 'labpro' is also specified 
            - 'trunc' = truncate current and protocol to only show first 3 pulses 
            - 'trunc#' = truncate current and protocol to only show first # pulses
            * 'env' = envelope protocol, so fit and show exponential to peak return currents (not implemented yet) 
            e.g. `fig_style = "nopro_trunc"
        `show` = whether to show plots 
        `pub_bounds` = whether to use 'publication' bounds on figure dimensions 
        `annotation_kwargs` = keyword arguments that are passed to `self.AddAnnotations()`, only applied if `annotate` in `fig_style`
        
        """
        # check if time units are s or ms, convert to ms (will change to s later if needed)
        # if (df.index[1] - df.index[0]) < 0.1:
        #     df.index *= 1e3 
                
        # data properties 
        self.df = df 
        self.N = int(df.shape[1]/2)
        self.fname = filename 
        self.intervals = intervals 
        self.khz = khz 
        self.units = units 
        
        # first key in `intervals` because some intervals dictionaries may not have `0` in the beginning
        self.k_ = list(self.intervals.keys())[0]
        
        # output 
        self.save_path = save_path 
        self.pdfs = pdfs 
        self.show = show 
        self.leaksub = leaksub 
        
        # figure properties
        Set_RC_Defaults(pub=pub)
        self.pub = pub 
        self.linew = linew 
        self.scalebar_frac = scalebar_frac
        self.show_pro_sb = show_pro_sb
        self.paddict = paddict 
        self.inner_paddict = inner_paddict 
        self.show_zero_current = show_zero_current
        self.show_zero_voltage = show_zero_voltage
        self.fig_style = fig_style.split("_") 
        self.label_info = None 
        self.annotation_kwargs = annotation_kwargs
                
        # figure dimensions 
        if pub_bounds and pub:
            if len(dims) == 2:
                self.dims = dims 
            else:
                print("No dimensions provided for figure size with `pub = True`. \
                    Using (85/25.4, 2.5)")
                self.dims = (85/25.4, 2.5)
        elif dims is not None:
            self.dims = dims 
        else:
            self.dims = (12, 5.5)
                
        if pub:
            # min, max values for figure size in each dimension (in inches) 
            self.x = [85/25.4, 7]
            self.y = [2, 9]
            self.dpi = 300             
        else:
            self.x = None 
            self.y = None 
            self.dpi = 300 
            
    def CheckFigDims(self, dims):
        """
        for publication plots, follow the default min and max dims in self.x and self.y 
        if self.pub is True, check that `dims` is in bounds of self.x and self.y 
        """
        if self.pub: 
            if len(dims) != 2:
                raise Exception(" `dims` must have 2 elements.")
            
            if not (self.x[0] <= dims[0] <= self.x[1]):
                return False 
            elif not (self.y[0] <= dims[1] <= self.y[1]):
                return False 
            else:
                return True 
        else:
            return True 
            
    def CreateFigure(self):
        """
        Create different types of figures/axes/gridspecs to fit the `mode` of plotting.
        
        # Major formats
        ## Standard:
            - only keep left y-axis and bottom-most x-axis (e.g. time for protocol, but not current, if protocol is enabled)
            - 1:1 and 7:1 current:protocol ratios for x- and y-dims, respectively, if protocol
        ## Publication (`self.pub = True`)
            - scale bars instead of x- or y-spines
            - protocol in an inset, with current taking up (3/4, :) and protocol (1/4, 1/2:) of the gridspec (# rows, # columns)
            - voltages labelled  
        
        ## Additional
            - if `nopro` in `self.fig_style.split("_")`, then voltage protocol is not plotted 

        Returns all generated objects, e.g. fig, axs, gridspec 
        If one of [fig, axs, gridspec] are not used, the corresponding variable is a NoneType 
        """
        
        # check figure dimensions are in bounds, False only possible if self.pub is True 
        # if not self.CheckFigDims(self.dims):
        #     raise Exception("Fig Dims are not in bounds")
                
        if self.pub:
            if "nopro" in self.fig_style:
                fig, ax1 = plt.subplots(figsize=self.dims, constrained_layout=True)
                remove_spines(ax1, which=["all"])
                
                return fig, ax1, None 
            
            else:
                fig = plt.figure(figsize=self.dims)
                gs = fig.add_gridspec(10, 10)
                
                ax1 = fig.add_subplot(gs[:-1, :])    #current 
                ax2 = fig.add_subplot(gs[8:, 5:])   #voltage protocol
                
                # set facecolor to be transparent, so that we can show overlapping gridspecs
                ax1.set_facecolor('none')
                # ax1.patch.set_alpha(0)
                ax2.set_facecolor('none')
                # ax2.patch.set_alpha(0)
                
                remove_spines(ax1, which=["all"])
                remove_spines(ax2, which=["all"])
                
                # remove ticks from protocol gridspec 
                ax2.tick_params(axis='both', which='both', 
                        bottom=False, top=False, left=False, right=False, 
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False
                )
                
                return fig, [ax1, ax2], gs
        
        else:
            if "nopro" in fig_style:
                fig, ax1 = plt.subplots(figsize=dims, constrained_layout=True)
                remove_spines(ax1, which=["right", "top"])
                
                ax1.tick_params(axis='both', length=6, width=3, labelsize=12)
                ax1.xaxis.set_ticks_position('bottom')
                
                return fig, ax1, None 
            
            else:
                fig = plt.figure(figsize=dims, constrained_layout=True)
                gs = fig.add_gridspec(7,1)
                
                ax1 = plt.subplot(gs[:5])       #current 
                ax2 = plt.subplot(gs[5:])       #voltage protocol
                
                for a in [ax1, ax2]:
                    remove_spines(a, which=["right", "top"])
                    a.tick_params(axis='both', length=6, width=3, labelsize=12)
                    a.xaxis.set_ticks_position('bottom')
                                
                return fig, [ax1, ax2], gs 
        
    def AddScalebars(self, axs, origin=(0, 0), fs=5):
        """
        Add scalebars to `axs` using `add_scalebars`
        
        `axs` = axis object, or list of axes 
        `frac` = length of scalebar as fraction of each dimension of axis length
        - allows specifying different values along x and y dimensions by providing a tuple, e.g. (x, y) = (0.2, 0.4)
        `origin` = origin of scalebars 
        
        `units` = dictionary of units for respective dimensions 
        `fs` = fontsize of labels 
        """
        
        def GetScalebarLengths(a, frac=self.scalebar_frac):
            """
            Get scalebar lengths for axis `a` 
            """
            if not isinstance(frac, float):
                dx, _ = FindScaleBarLengths(a, frac=frac[0])
                _, dy = FindScaleBarLengths(a, frac=frac[1])
            else:
                dx, dy = FindScaleBarLengths(a, frac=frac)
            
            return dx, dy 
        
        def RaiseUnit(dy, key, pre=['p', 'n', 'u', 'm']):
            """
            Raise unit by 1000 if possible.
            """
            tup = (dy, self.units[key])
            
            if dy >= 1000:
                if self.units[key][0] in pre:
                    new_unit = pre.index(self.units[key][0]) + 1
                    
                    if new_unit < len(pre):
                        tup = (dy/1000, pre[new_unit] + self.units[key][1])
                    else:
                        tup = (dy/1000, self.units[key][1])
                
            return "%d %s" % tup 
        
        # separation between labels and scalebars 
        sep = fs*0.8
        
        if isinstance(axs, list):
            for i, ax in enumerate(axs):
                dx, dy = GetScalebarLengths(ax)
                labelx = "{x} {u}".format(x=dx, u=self.units["x"])
                                                
                if i == 0:
                                            
                    labely = RaiseUnit(dy, "y_i")
                    loc = 'lower left'
                    bbox = (-0.1, -0.1)
                    kwargs = dict(labely=labely, loc=loc, origin=origin,
                                bbox_to_anchor=bbox, bbox_transform=ax.transAxes)
                else:
                    labely = RaiseUnit(dy, "y_v")
                    loc = 'lower left'
                    bbox = (-0.3, -0.3)
                    kwargs = dict(labely=labely, loc=loc, origin=origin,
                                bbox_to_anchor=bbox, bbox_transform=ax.transAxes)
                                
                add_scalebar(ax, matchx=False, matchy=False,                        
                            sizex=dx, sizey=dy, labelx=labelx, 
                            label_fontsize=fs, sep=sep, 
                            **kwargs) 
            
        else:
            dx, dy = GetScalebarLengths(axs)
            
            labelx = "{x} {u}".format(x=dx, u=self.units["x"])
            labely = RaiseUnit(dy, "y_i")
            
            loc = 'lower left'
            bbox = (-0.1, -0.1)
            kwargs = dict(labely=labely, loc=loc, origin=origin,
                        bbox_to_anchor=bbox, bbox_transform=axs.transAxes)
            
            add_scalebar(axs, matchx=False, matchy=False,                        
                        sizex=dx, sizey=dy, labelx=labelx, 
                        label_fontsize=fs, sep=sep, **kwargs) 

    def PrepareVoltageLabels(self):
        """
        For each epoch in the protocol (element in `self.intervals`), get the time and voltage at the midpoint of the epoch
        
        Returns: 
            `times` = time of midpoints
            `levels` = voltage at midpoints 
        """
        # isolate voltage protocol 
        df_v = self.df.iloc[:, self.N:]
        
        # convert index of df_v to ms (since intervals are in ms) 
        # if (df_v.index[1] - df_v.index[0]) < 0.1:
        #     df_v.index *= 1e3 
        
        # iterator for index of sweeps in `intervals`
        # we use this when indexing `intervals`, but 
        # `range(1, self.N)` for other objects of size `self.N`
        sweeps = list(self.intervals.keys())
        
        try:
            # check whether all sweeps have the same length of steps
            if all( (len(self.intervals[i]) == len(self.intervals[self.k_])) for i in sweeps[1:]):
                n_ = len(self.intervals[self.k_])
            # if not, analyze the minimum nubmer of steps 
            else:
                n_ =  min( len(self.intervals[i]) for i in sweeps )
        except:
            print("Intervals: ", self.intervals)
            print("Number of sweeps: ", self.N)
            raise Exception("Unequal number of intervals and sweeps")
                        
        # for each epoch present in all sweeps, collect the following:
        times = []      # start times 
        durations = []  # durations 
        voltages = []   # voltages 
        
        for j, v in enumerate(self.intervals.values()):
            # create list of epoch start times 
            times.append([0])
            times[j].extend([ v[i] for i in range(n_ - 1) ])
            
            # durations of each epoch 
            dt = [v[0]]
            dt.extend(
                [ (v[i] - v[i-1]) for i in range(1, n_) ]
            )
            
            # check if the last epoch is assigned 0 zero duration (e.g. return to holding)\
            if dt[-1] == 0:
                dt[-1] = np.nan 
            
            durations.append(dt)
            
            # levels for each of the (i) epochs of the jth sweep  
            volts_j = df_v.iloc[:,j]
            
            # find voltage of j-th epoch for each sweep
            # add small offset of 100ms to account for capacitance
            volts = [
                int( math.floor(volts_j.iloc[ times[j][i] + 20*self.khz ]) / 5) * 5
                for i in range(n_)
            ]
                                
            voltages.append(volts)
        
        # check for nan values in last durations 
        if any( np.isnan(d[-1]) for d in durations ):
            x = [d[-1] for d in durations if not np.isnan(d[-1])][0]
            
            for j in [j for j in range(len(durations)) if np.isnan(durations[j][-1])]:
                durations[j][-1] = x 
    
        # hold (t0, dt, y) for each label, where
        # t0 = start time of each epoch 
        # dt = duration of each epoch 
        # y = level of each epoch 
        label_info = {} 
        
        for j in range(n_):
            """
                We only analyze `n_` epochs, where `n_` is either the number of epochs, if all sweeps have the same number of epochs; else, the minimum number of epochs.
                
                For each epoch, check if all durations are the same, or if all voltages are the same. 
                
                If both are all the same, then we update `label_info` with key `j` with a single tuple `(t, v)`, where `t` and `v` are the time and voltage of the j-th epoch, which are identical for all sweeps 
                
                Otherwise, we update `label_info` with key `j` and value `[(t_min, v_min), (t_max, v_max)]`, where the subscripts `_min` and `_max` correspond to the minimum and maximum, respectively, values of the epoch duration and/or voltage.
                
                Returns the dictionary `label_info` as described above.
            """            
            
            # check that no sweeps were assigned zero duration in the jth epoch 
            if any(durations[i][j] == 0 for i in range(1, self.N)):
                # index of zero duration sweeps 
                dt_zero = [i for i in range(self.N) if durations[i][j] == 0]
                # all non zero durations
                dt_nonzero = [d[i][j] for d in durations if (i not in dt_zero)]
                
                # if all other sweeps are uniform duration, then re-assign with this value 
                if all( (d == dt_nonzero[0] for d in dt_nonzero[1:]) ):
                    for i in dt_zero:
                        durations[i][j] = dt_nonzero[0] 
                        
                else:
                    raise Exception("A zero duration sweep was assigned for epoch {j} of \
                        sweep(s) {dt_zero}. Further, the durations of non-zero sweeps \
                        are non-uniform. Thus, re-assignment of the zero duration sweeps \
                        is ambiguous.".format(j=j, dt_zero=dt_zero))
            
            # check if jth epoch is the same duration for all sweeps 
            if all( (durations[i][j] == durations[0][j]) for i in range(1, self.N) ):
                
                # check if voltage is the same for jth epoch of all sweeps 
                # i.e. fixed duration, fixed voltage 
                if all( (voltages[i][j] == voltages[0][j]) for i in range(1, self.N) ):
                    label_info.update(
                        {j : (times[0][j], durations[0][j], voltages[0][j])}
                    )
                                
                # if not, append information for minimum and maximum level epochs 
                # i.e. fixed duration, varying voltage 
                else:
                    v_j = [v[j] for v in voltages]
                    
                    i_min = np.argmin(v_j)
                    i_max = np.argmax(v_j)    
                    
                    label_info.update(
                        {j : [
                            ( times[i_min][j], durations[i_min][j], min(v_j) ), 
                            ( times[i_max][j], durations[i_max][j], max(v_j) ) 
                        ]}
                    )
                
            # if not, then duration varies 
            else:
            
                # check if voltage is uniform
                # i.e. varying duration, fixed voltage 
                if all( (voltages[i][j] == voltages[0][j]) for i in range(1, self.N) ):
                    
                    # select epochs with minimum and maximum duration 
                    dt_j = [dt[j] for dt in durations]
                    
                    i_min = np.argmin(dt_j)
                    i_max = np.argmax(dt_j)

                    label_info.update(
                        {j :[
                            ( times[i_min][j], durations[i_min][j], voltages[i_min][j] ), 
                            ( times[i_max][j], durations[i_max][j], voltages[i_max][j] )
                        ]}
                    )
                
                # voltage varies, duration varies 
                # select epochs with minimum and maximum voltage 
                else: 
                    v_j = [v[j] for v in voltages]
                    
                    i_min = np.argmin(v_j)
                    i_max = np.argmax(v_j)    
                    
                    label_info.update(
                        {j : [
                            ( times[i_min][j], durations[i_min][j], min(v_j) ), 
                            ( times[i_max][j], durations[i_max][j], max(v_j) ) 
                        ]}
                    )
        
        # save `label_info`
        self.label_info = label_info 
        
        return label_info 
    
    def AddVoltageLabels(self, ax, Labeller, static_offset=10.0):
        """
        Add voltage labels prepared by `self.PrepareVoltageLabels()` to voltage protocol subplot `a` with kwargs `kwargs`
        
        `ax` = axis 
        `Labeller` = instance of `VoltageProtocolLabels`
        `offset` = size of offset 
        `static_offset` = if True, uses 10mV. 
            If a float, this value is used instead. 
            Else, a relative offset of 5% of the total range of the y-axis is used
        """
                
        # dictionary of labels of (time, voltage) of j-th epoch, where `j in [1, n_]`, where `n_` is the number of epochs shared across all traces.
        # for fixed-voltage, fixed duration epoch: {j : (t, v)}
        # if voltage or duration vary: {j : [ (t_min, v_min), (t_max, v_max) ]}
        label_info = self.PrepareVoltageLabels()
        Labeller.label_info = label_info 
        print(Labeller.label_info)
        # exit()
        
        # convert samples to seconds 
        ToSeconds = lambda x: x / (1e3 * self.khz)

        # labels for voltages plotted, to avoid repeating 
        voltages_plotted = [] 
        
        # voltage offset
        if isinstance(static_offset, float):
            threshold = static_offset
        elif static_offset == True:
            threshold = 10 
        elif static_offset == False:
            # useful if swapping `t_max` for `t_min` as x-location of `v_max` doesn't remove overlap
            # dv = 0.02*(v_max - v_min)                
            # rather than using v_max, v_min (which are epoch-dependent), 
            # global ylims is more reliable
            ylims = ax.get_ylim()
            threshold = 0.1*abs(ylims[1] - ylims[0])
            
        for i, val in enumerate(label_info.values()):
            offset = threshold 
            
            if i > 2:
                print("Only 3 labels will be plotted")
                break 
            
            if isinstance(val, tuple):
                t, dt, v = val 
                
                # avoid plotting same voltage 
                if (i > 0) and (v in voltages_plotted):
                    continue 
                else:
                    voltages_plotted.append(v)
                
                # set kwargs for text appearance
                # first sweep (holding potential label)
                if i == 0:
                    kwargs={'ha':'center', 'va':'center'}     
                    
                else:
                    # place label below voltage if `v` is most negative voltage at `t`
                    if (self.df.iloc[(t + 50*self.khz), self.N:] >= v).all():
                        kwargs={'ha':'center', 'va':'top'}
                        offset *= -1 
                    else:
                        kwargs={'ha':'center', 'va':'bottom'}
                    
                if i == 0:
                    # redefine dt using xlims 
                    xlim = ax.get_xlim()
                    dt = abs(xlim[1] - xlim[0])
                    ax.text(self.t0 - 0.075*dt, v, "%d" % v, **kwargs)
                    
                    voltages_plotted.append(v)
                    continue 
                else:
                    # move x-coordinate of label to middle of epoch 
                    t += 0.5*dt 
                    
                    if self.units["x"] == "s":
                        t = ToSeconds(t)
                        dt = ToSeconds(dt)
                    else:
                        t *= 1/self.khz 
                        dt *= 1/self.khz
                        
                    # check if ramp endpoint, i.e. both sides of `v` != v 
                    if (t > 50) and (self.df.iloc[[t-10, t+10], (i + self.N)] != v).all():
                        t -= 0.25*dt     
                        
                    Labeller.AddOffsetLabel(ax, t, v, offset, kwargs=kwargs)
                    
            elif isinstance(val, list):                
                # unpack time, duration, and level of epochs 
                # where val[0] and val[1] differ in duration and/or level 
                t_min, dt_min, v_min = val[0]
                t_max, dt_max, v_max = val[1] 
                
                # convert to s if needed 
                if self.units["x"] == "s":
                    t_min = ToSeconds(t_min)
                    t_max = ToSeconds(t_max)
                    dt_min = ToSeconds(dt_min)
                    dt_max = ToSeconds(dt_max)
                # else, convert to time units using sample frequency 
                else:
                    t_min *= 1/self.khz
                    t_max *= 1/self.khz 
                    dt_min *= 1/self.khz
                    dt_max *= 1/self.khz
                                
                # offset for labels 
                # move times to middle of epoch (= 0.5*dt for <10, 0.4*dt for <100, and 0.3*dt for <1000)
                t_min, t_max = Labeller.AddTimeOffset([t_min, t_max], [v_min, v_max], [dt_min, dt_max])
                
                # maximum voltage 
                kwargs = dict(color='k', va='bottom', ha='center')
                
                # avoid plotting repeat voltages 
                if (v_min in voltages_plotted) or (v_max in voltages_plotted):
                    continue 
                
                # if voltages are the same, but varying duration, just plot one label 
                elif v_min == v_max:
                    voltages_plotted.append(v_min)
                    
                    # since v_min and v_max are the same, move label to the middle of max duration epoch
                    t_min += dt_max/2 
                    
                    # if the most positive voltage at time `t`, then plot above 
                    if (self.df.iloc[(int(t_min) + 50*self.khz), self.N:] <= v_min).all():
                        kwargs['va'] = 'bottom'
                        
                    elif (i > 0):
                        # slice of the last epoch for all sweeps 
                        if isinstance(label_info[i-1], list):
                            last_ = self.df.iloc[(label_info[i-1][0][0] + 50*self.khz), self.N:] 
                        else:
                            last_ = self.df.iloc[(label_info[i-1][0] + 50*self.khz), self.N:]
                        
                        if (last_ > v_min).any():
                            # if any steps for the previous epoch have voltage > v_min, plot v_min below
                            kwargs['va'] = 'top'
                            offset *= -1 
                            
                    Labeller.AddOffsetLabel(ax, t_min, v_min, offset, kwargs=kwargs)
                    continue              
                else:
                    voltages_plotted.append(v_min)
                    voltages_plotted.append(v_max)
                
                # label of v_max 
                kwargs['va'] = 'bottom'
                
                # if v_max is the most positive voltage at `t`
                if (self.df.iloc[(int(t_max) + 50*self.khz), self.N:] <= v_min).all():
                    # label is added at `t_max` if there is no overlap at (t_max, v_max)
                    # otherwise, appropriate offset for `v_max` will be found at (t_min, v_max)
                    Labeller.AddOffsetLabel(ax, t_max, v_max, offset, t_swap=t_min, kwargs=kwargs)
                else:
                    # start with `t_min`, with `t_max` as `t_swap`
                    Labeller.AddOffsetLabel(ax, t_min, v_max, offset, t_swap=t_max, kwargs=kwargs)
                
                # label of v_min 
                kwargs['va'] = 'top'
                Labeller.AddOffsetLabel(ax, t_min, v_min, -offset, kwargs=kwargs)
                
            else:
                raise Exception("Type of value in dictionary `label_info` must be tuple or list, currently \n ", type(val))
        
    def ApplyInnerPadding(self, ax):
        """ 
        Some visual modifications for voltage protocol 
            - values and spacing (default=50) of y-ticks
        
        Inner padding specified in `self.inner_paddict`, with keys `h_pad` and/or `w_pad`
        
        e.g. `h_pad` = vertical padding
            if `h_pad` > 1, then units assumed to be volts. else, fraction of current range.
        """
        
        def RoundDown(L):
            """ Return elements of `L` rounded down to largest integer power of 10 """
            # return largest integer power of 10 that is multiple of a number `a` 
            E = lambda a : int(10 ** math.floor(math.log10(abs(a))))
            
            # apply E to all elements of L
            for i, x in enumerate(L):
                e = E(x) 
                L[i] = int(x/e)*e
            
            return L 
        
        if "h_pad" in self.inner_paddict.keys():
            h_pad = self.inner_paddict["h_pad"]
            
            ylo, yhi = ax.get_ylim()
            dy = abs(yhi - ylo)
            
            if h_pad < 1: 
                h_pad *= dy
                
            ax.set_ylim(bottom=(ylo - h_pad), top=(yhi + h_pad))
                
        if "w_pad" in self.inner_paddict.keys():
            w_pad = self.inner_paddict["w_pad"]
            
            xlo, xhi = ax.get_xlim()
            dx = abs(xhi - xlo) 
            
            if w_pad < 1: 
                w_pad *= dx 
            
            ax.set_xlim(left=(xlo - w_pad), right=(xhi + w_pad))
            
    def Std_AxesSpinesTicks(self, axs, seconds=True):
        """
        Set up visuals, e.g. tick locations, appearance of spines 
        `axs` = axes 
        """
        for j, a in enumerate(axs):
            if j == 1: #voltage 
                a.set_xlabel("Time (%s)" % self.units["x"], fontsize=12, labelpad=10)
                
                # add downwards offset to bottom spines  
                a.spines['bottom'].set_position(('outward', 10))
                
            else:
                a.yaxis.set_ticks_position('left') 
                # a.spines['bottom'].set_visible(False)
                a.locator_params(axis='y', nbins=4)
        
            #four tick marks per axis 
            a.locator_params(axis='x', nbins=6)
                    
            #empty xtick labels for current 
            if j == 0:                
                a.set_xticklabels([""])
                a.set_xticks([]) 
    
    def AddAnnotations(self, ax, data, to_label=None, multiple=False, 
                    whitespacevline=False, yfrac=-0.05):
        """
        Add annotations using `add_annotation()` to `ax` using information in `self.label_info`
        
        `data` = dataframe containing current time courses 
        `to_label` = which variable ('x' for time or 'y' for voltage) that will determine 
            how the annotation will appear. 
            e.g. if both voltage and duration of an epoch vary between sweeps, then we will 
            fall back to `to_label` to determine how the annotation will be added. 
            Else, if only one of duration or level vary, then `to_label` is not used.
        `multiple` = whether to add multiple annotations, if (supposedly) present 
            (i.e. multiple varying epochs)
        `whitespacevline` = whether to add a vertical annotation when there is substantial whitespace
            between start of annotated epoch and annotation 
        """
        
        if self.label_info is None:
            raise Exception("`AddAnnotations()` requires `self.label_info` to not be None. \
                This can be done by calling `self.PrepareVoltageLabels`. If this was done, \
                the procedure failed or occurred incorrectly.")
        
        def do_annotation(v, yfrac=yfrac):
            """
            Adds annotation of a list `v` containing (start time, duration, level) tuples for minimum and maximum instances of a given epoch for two sweeps. 
            
            `yfrac` = vertical offset for horizontal annotations, as 
                    fraction of y-axis range (maximum - minimum)
            """
            if to_label is None:
                print("Since 'to_label' is None, the default annotation behaviour is \
                    to select the axis of variation. The label will be the size of \
                    maximum variation. if this is undesired, pass 'x' or 'y' to `to_label`")

                # determine if duration and/or level vary 
                if (v[1][1] != v[0][1]) and (v[1][2] != v[0][2]):
                    raise Exception("Both level and duration vary, which requires `to_label` to \
                        not be None to select which will determine annotation.")
            
            if (to_label == 'x') or ((to_label is None) and (v[1][1] != v[0][1])):    
                
                if v[1][1] != v[0][1]:
                    dx = v[1][1] if (v[1][1] > v[0][1]) else v[0][1] 
                else:
                    dx = v[0][1] 
                
                # find y-coordinate for label position
                # find most negative current within epoch
                # offset v[i][0] by 50 to avoid capacitive currents 
                if all([ v[i][1] > 50 for i in range(2) ]):
                    y = min(
                        [data.iloc[(v[i][0] + 50):(v[i][0] + v[i][1]), :].min().min() for i in range(2)]
                    )
                else:
                    y = min(
                        [data.iloc[v[i][0]:(v[i][0] + v[i][1]), :].min().min() for i in range(2)]
                    )
                
                # lower `y` by offset equal to 5% of y limits 
                ylims = ax.get_ylim()
                if yfrac != 0:
                    y += abs(ylims[1] - ylims[0]) * yfrac
                
                # if there is substantial vertical whitespace at the beginning of the plot
                # add an additional vertical line at the left extreme of the annotation
                if whitespacevline:
                    # mean distance from `y`
                    mean_dy = [
                        (data.iloc[v_i[0]:(v_i[0] + v_i[1]), :] - y).mean().mean() > 0 for v_i in v 
                    ]
                    
                    if all(mean_dy):
                        if self.units['x'] == 's':
                            t0 = v[0][0] / (1000*self.khz)
                        else:
                            t0 = v[0][0] / self.khz 
                            
                        add_annotation(ax, "", (t0, y), dy = abs(0.98*y))
                
                # x coordinate is the start time + half of the duration,
                # which we convert to time units 
                x = v[0][0]
                if self.units['x'] == 's':
                    dx *= 1/(1000*self.khz)
                    x *= 1/(1000*self.khz)
                else:
                    dx *= 1/self.khz 
                    x *= 1/self.khz 

                # label will be the duration + `self.units['x']`
                lab = "%d%s" % (dx, self.units['x'])
                
                # add annotation with offset equal to 2% of `dx`
                add_annotation(ax, lab, (x, y), dx=dx, offset=0.02)
                
            elif (to_label == 'y') or ((to_label is None) and (v[1][2] != v[0][2])):
                print("Default behaviour of `AddAnnotations` is to add label of varying axis. \
                    `y`, or voltage, of epochs was detected to vary. The label will be added to \
                    `ax`. This may be undesirable if `ax` does not contain the voltage protocol.")
                
                dy = abs(v[1][2] - v[0][2])
                if dy == 0:
                    dy = v[1][2]
                
                # set label as difference in level + self.units['y_v']
                lab = "%d%s" % (dy, self.units['y_v'])
                
                # find y-coordinate as most negative current value between v[0] and v[1]
                y = [data.iloc[v[i][0] + 100, :].min() for i in range(2)]
                y = y[0] if (y[0] < y[1]) else y[1]
                
                # redefine dy as the maximum difference in current 
                dy = [data.iloc[v[i][0]:(v[i][0] + v[i][1]), :].dropna() for i in range(2)]
                dy = (dy[0] - dy[1]).abs()
                # find time of maximum
                x = dy.max(axis=1).argmax()
                # find maximum difference current by taking max along rows, then columns 
                dy = dy.max(axis=0).max()
                
                if dy == 0:
                    dy = (data.iloc[(v[0][0] + v[0][1]), :] - data.iloc[v[0][0], :]).abs().max()
                                
                # move `y` up by half maximal difference in current 
                y += dy/2                  
                
                if self.units['x'] == 's':
                    x *= 1/(1000*self.khz)
                else:
                    x *= 1/self.khz 

                add_annotation(ax, lab, (x, y), dy=dy, offset=0.02)
        
        # self.label_info = {i : tup, j : list}, where i and j are 0-indexed sweep numbers
        # when values are tuples, all sweeps are uniform at this epoch, 
        #   and the tuple contains (start time, duration, level) of the epoch
        # when the value is a list, then the list is a list of tuples as described above 
        
        # for annotation to occur, we need an epoch that varies in either voltage and/or duration
        # find keys of `self.label_info` that have lists as values 
        varying = [i for i in self.label_info.keys() if isinstance(self.label_info[i], list)]
        if len(varying) == 0:
            print("No varying epochs were found in `self.label_info`. Annotation was not done.")
            return None 
        
        # if there are more than one pulse that varies, the selection is ambiguous,
        # and annotation will not proceed 
        elif len(varying) > 1:
            print("More than one varying epoch was found, making selection of annotation ambiguous. \
                The default behaviour is to take the first occurrence. \
                If multiple annotations are desired, then pass `multiple=True`")
            print([self.label_info[i] for i in varying])
            
            if multiple:
                for v in varying:
                    do_annotation(self.label_info[v])
            else:
                do_annotation(self.label_info[varying[0]])
        
        else:
            # v[0] = min, v[1] = max (duration/voltage)
            # v = self.label_info[varying[0]]
            do_annotation(self.label_info[varying[0]])
        
    def make_figure(self):
        """
        Make figures for ephys traces   
        
        `format` = if "leaksub" -> adds '_leaksub' as a suffix to saved files and "Leak-subtracted" to y-axis label 
        """    
        
        if self.leaksub:
            print("Pub plots using leak-subtracted data. For raw data, pass 'leaksub : False' in `do_pubplots`.")
            format = 'leaksub'
        else:
            print("Pub plots using raw data. If pub plots for leak-subtracted data are desired, pass 'leaksub : True' in `do_pubplots`.")
            format = 'x'
        
        # set linewidth
        if self.linew is None:
            if self.pub:
                lw = 1 
            else:          
                lw = 1.5 
        else:
            lw = self.linew 
            
        ### Create figure 
        fig, axs, gs = self.CreateFigure()
        
        ### Prepare data 
        # convert ms to seconds 
        if self.units["x"] == "s":
            self.df.index *= 1/1000 
        
        # split current and voltage 
        df_i = self.df.iloc[:, :self.N] 
        df_v = self.df.iloc[:, self.N:] 
                        
        # start of test pulses; add 200ms of holding to the start
        intervals = self.intervals 
        t0 = intervals[self.k_][0] - 200*self.khz 
        
        while (df_v.iloc[t0, :] == df_v.iat[t0-20, 0]).all() and (t0 < 500*self.khz):
            t0 -= 20
        
        # start time for platting data 
        self.t0 = t0/(1000*self.khz) if (self.units['x'] == 's') else t0/self.khz 
        
        ### Truncate data 
        if self.fig_style:
            
            # find `trunc` if in `self.fig_style`
            if any( (x[:5] == "trunc") for x in self.fig_style ):
                s = [x for i, x in enumerate(self.fig_style) if x[:5] == "trunc"][0]
                
                # find number of pulses to keep 
                if len(s) > 5:
                    m = int(s[5])
                else:
                    if len(intervals[self.k_]) > 3:
                        m = 3 
                    else:
                        m = len(intervals[self.k_]) - 1
                                        
                # determine if position of mth epoch is uniform, or changes between sweeps
                sweeps = list(intervals.keys())
                                
                if any( (intervals[i][m] != intervals[self.k_][m]) for i in sweeps[1:] \
                        if len(intervals[i]) > m):
                    trunc_i = []
                    trunc_v = [] 
                    for i, j in enumerate(sweeps):
                        t = intervals[j][m]
                        trunc_i.append(df_i.iloc[:t, i])
                        trunc_v.append(df_v.iloc[:t, i])
                        
                    df_i = pd.concat(trunc_i, axis=1).apply(lambda x: pd.Series(x).dropna())
                    df_v = pd.concat(trunc_v, axis=1).apply(lambda x: pd.Series(x).dropna())
                                            
                else:
                    m = intervals[self.k_][m]
                    df_i = df_i.iloc[:m*self.khz, :]
                    df_v = df_v.iloc[:m*self.khz, :]
            
            # create VoltageProtocolLabels object
            if "labpro" in self.fig_style:
                Labeller = VoltageProtocolLabels(protocol=df_v, n_sweeps=self.N, 
                                                khz=self.khz, units=self.units)
            
        # truncate last 25% if total duration is > 50s and change in current < 50 pA 
        elif ((df_v.shape[0] / self.khz) > 50e3):
            t = int(0.75*df_v.shape[0])
            
            i0 = df_v.iloc[t:, :].min(axis=0).values
            imax = df_v.iloc[t:, :].max(axis=0).values
            
            if (np.abs(imax - i0) < 50).all():
                df_i = df_i.iloc[:t, :]
                df_v = df_v.iloc[:t, :] 
                
        ### Plot data
        if "nopro" in self.fig_style:
            axs.plot(df_i.iloc[t0:, 0], c='r', lw=(lw+0.5))
            axs.plot(df_i.iloc[t0:, 1:], c='k', lw=lw)    
            
            if self.inner_paddict:
                self.ApplyInnerPadding(axs)
                
            if self.show_zero_current:
                axs.axhline(0, c='k', alpha=0.5, ls=':', lw=(lw+1))
        else:
            ax1, ax2 = axs 
            
            ax1.plot(df_i.iloc[t0:, 0], c='r', lw=(lw+0.5))
            ax2.plot(df_v.iloc[t0:, 0], c='r', lw=(lw+0.5))
            
            if self.show_zero_current:
                ax1.axhline(0, c='k', alpha=0.5, ls=':', lw=(lw+1))
            if self.show_zero_voltage:
                ax2.axhline(0, c='k', alpha=0.5, ls='--', lw=(lw+1))
            
            if df_v.shape[1] > 1:        
                ax2.plot(df_v.iloc[t0:, 1:], c='k', lw=lw)
                ax1.plot(df_i.iloc[t0:, 1:], c='k', lw=lw)
            
            # apply inner padding before adding objects (e.g. labels, scalebars), because 
            # these depend on data coordinates
            if self.inner_paddict:
                self.ApplyInnerPadding(ax1)
                self.ApplyInnerPadding(ax2)
                
            # add labels to voltage protocol 
            if self.pub:
                if "labpro" in self.fig_style:
                    self.AddVoltageLabels(ax2, Labeller)
            else:
                if format == "leaksub":
                    ax1.set_ylabel("Leak-subtracted\nCurrent (%s)" % self.units["y_i"], 
                                labelpad=12, fontsize=14)
                else:
                    ax1.set_ylabel("Current (%s)" % self.units["y_i"], labelpad=12, fontsize=14)
                ax2.set_ylabel("Voltage (%s)" % self.units["y_v"], labelpad=12, fontsize=14)
                        
        # add scalebars (for publication) 
        if self.pub:
            if self.show_pro_sb:
                self.AddScalebars(axs)
            else:
                self.AddScalebars(axs[0])
                
        # for 'standard' plots: add axes labels, etc. 
        # add x-axis label, remove spines, add negative offset to bottom spine, and 
        # set nbins and position of ticks 
        else:    
            self.Std_AxesSpinesTicks(axs)
            
        # add annotations
        if "annotate" in self.fig_style:
            if "nopro" in self.fig_style:
                pass
            else:
                self.AddAnnotations(ax1, df_i, **self.annotation_kwargs)
                        
        ### Save output 
        if self.pdfs is None:
            if self.save_path:
                
                # filename of saved figure 
                if self.pub:
                    s = " __ ".join([self.fname, "pub", format])
                else:
                    s = " __ ".join([self.fname, "pub", format])
                    
                plt.savefig(self.save_path + r"%s.svg" % s, bbox_inches='tight', dpi=self.dpi)
                plt.savefig(self.save_path + r"%s.png" % s, bbox_inches='tight', dpi=self.dpi)
                
                print("Successfully saved .png and .svg files for \
                        < %s > at: \n %s" % (s, self.save_path))
        else:
            self.pdfs.savefig(bbox_inches='tight', dpi=self.dpi)
        
        if self.show:
            plt.show()
            
        # exit()
            
