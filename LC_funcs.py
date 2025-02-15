import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def binning_weighted_mean(t, y, ye, binsize):

    intt = np.floor((t - np.min(t))/binsize)
    intt_unique = np.unique(intt)
    n_unique = len(intt_unique)
    tbin = np.zeros(n_unique)
    ybin = np.zeros(n_unique)
    yebin = np.zeros(n_unique)

    for i in range(n_unique):
        index, = np.where(intt==intt_unique[i])
        tbin[i] = np.mean(t[index])
        w = 1/ye[index]/ye[index]
        ybin[i] = np.sum(y[index]*w)/np.sum(w)
        yebin[i] = np.sqrt(1/np.sum(w))

    return tbin, ybin, yebin



def binning_median(t, y, binsize):

    intt = np.floor((t - np.min(t))/binsize)
    intt_unique = np.unique(intt)
    n_unique = len(intt_unique)
    tbin = np.zeros(n_unique)
    ybin = np.zeros(n_unique)

    for i in range(n_unique):
        index = np.where(intt==intt_unique[i])
        tbin[i] = np.mean(t[index])
        ybin[i] = np.median(y[index])

    return tbin, ybin


def binning_equal_interval(t, y, ye, binsize, t0):
    intt = np.floor((t - t0)/binsize)
    intt_unique = np.unique(intt)
    n_unique = len(intt_unique)
    tbin = np.zeros(n_unique)
    ybin = np.zeros(n_unique)
    yebin = np.zeros(n_unique)

    for i in range(n_unique):
        index = np.where(intt==intt_unique[i])
        tbin[i] = t0 + float(intt_unique[i])*binsize + 0.5*binsize
        w = 1/ye[index]/ye[index]
        ybin[i] = np.sum(y[index]*w)/np.sum(w)
        yebin[i] = np.sqrt(1/np.sum(w))

    return tbin, ybin, yebin


def outcut_median(t, y, ye, sigma_cut):
    tcut = t.copy()
    ycut = y.copy()
    yecut = ye.copy()
    nout = 1
    cut_list = []
    while(nout > 0):
        median = np.median(ycut)
        sigma = np.abs((ycut - median)/yecut)
        index, = np.where(sigma < sigma_cut)
        cut_list.append(index)
        nout = len(tcut) - len(index)
        tcut = tcut[index]
        ycut = ycut[index]
        yecut = yecut[index]

    return cut_list

def outcut_median_alt(t, y, ye, sigma_cut):
    tcut = t.copy()
    ycut = y.copy()
    yecut = ye.copy()
    nout = 1
    while(nout > 0):
        median = np.median(ycut)
        sigma = np.abs((ycut - median)/yecut)
        index = np.where(sigma < sigma_cut)
        nout = len(tcut) - len(index[0])
        tcut = tcut[index[0]]
        ycut = ycut[index[0]]
        yecut = yecut[index[0]]

    return tcut,ycut,yecut

def outcut_polyfit_numpy(t, y, ye, order, sigma_cut):
    tcut = t.copy()
    ycut = y.copy()
    yecut = ye.copy()
    nout = 1
    while(nout > 0):
        p = np.polyfit(tcut, ycut, order)
        ymodel = np.polyval(p, tcut)

        resi = ycut - ymodel
        sdev = np.sqrt(np.sum(resi**2)/float(len(resi)))
        sigma = np.abs(resi/sdev)
        index = np.where(sigma < sigma_cut)
        nout = len(tcut) - len(index[0])
        tcut = tcut[index[0]]
        ycut = ycut[index[0]]
        yecut = yecut[index[0]]

    return p, tcut, ycut, yecut

'''
def outcut_polyfit(t, y, ye, order, sigma_cut):
    tint = np.int(np.min(t))
    tcut = t - tint
    ycut = y.copy()
    yecut = ye.copy()
    nout = 1
    p0 = np.zeros(order+1)
    p0[-1] = np.median(y)
    index_return = []
    result = minimize(calc_chi2_polyfit, p0, args=(tcut, ycut, yecut), method='Nelder-Mead')

    while(nout > 0):
        p0 = result.x
        result = minimize(calc_chi2_polyfit, p0, args=(tcut, ycut, yecut), method='Nelder-Mead')
        ymodel = np.polyval(result.x, tcut)
        plt.plot(tcut, ycut, zorder=0)
        plt.plot(tcut, ymodel,zorder=1)
        resi = ycut - ymodel
        sdev = np.sqrt(np.sum(resi**2)/float(len(resi)))
        sigma = np.abs(resi/sdev)
        condition = sigma < sigma_cut #condition for point to keep
        index = np.where(condition)
        nout = len(tcut) - len(index[0])
        plt.scatter(tcut[~condition], ycut[~condition], color='red', marker="x",zorder=2)
        tcut = tcut[index[0]]
        ycut = ycut[index[0]]
        yecut = yecut[index[0]]
        index_return.extend(index[0])
        plt.show()
    plt.plot(t, y)
    plt.scatter(t[~np.unique(index_return)], ycut[~np.unique(index_return)], color='red', marker="x",zorder=2)

    return result.x, tcut+tint, ycut, yecut, np.unique(np.array(index_return))
'''




def outcut_polyfit(t, y, ye, order, sigma_cut):
    tint = int(np.min(t))  # Fixed deprecated np.int
    tcut = t - tint
    ycut = y.copy()
    yecut = ye.copy()

    # Use polyfit for better initial guess
    p0 = np.polyfit(tcut, ycut, order)

    kept_mask = np.ones(len(t), dtype=bool)  # Boolean mask tracking kept points
    nout = 1  # Start with a dummy value to enter loop

    while nout > 0:
        # Fit polynomial using chi-square minimization
        result = minimize(calc_chi2_polyfit, p0, args=(tcut, ycut, yecut), method='Nelder-Mead')
        p0 = result.x  # Update initial guess with best fit

        # Compute residuals and standard deviation
        ymodel = np.polyval(result.x, tcut)
        plt.plot(tcut, ycut, zorder=0)
        plt.plot(tcut, ymodel, zorder=1)

        resi = ycut - ymodel
        sdev = np.std(resi, ddof=1)  # Use unbiased std deviation
        sigma = np.abs(resi / sdev)  # Compute sigma values

        # Identify points to keep and remove
        condition = sigma < sigma_cut  # Points to keep
        kept_mask[kept_mask] = condition
        nout = len(tcut) - np.sum(condition)  # Count removed points

        # Plot removed points in red
        plt.scatter(tcut[~condition], ycut[~condition], color='red', marker="x", zorder=2)
        plt.show()

        # Filter only kept points for next iteration
        tcut, ycut, yecut = tcut[condition], ycut[condition], yecut[condition]


    # Final plot
    plt.plot(t, y, label="Original Data")
    plt.scatter(t[~kept_mask], y[~kept_mask], color='red', marker="x", zorder=2, label="Removed Points")
    plt.legend()
    plt.show()

    return result.x, t[kept_mask], y[kept_mask], ye[kept_mask], kept_mask  # Also return cut points

def outcut_smoothing(t, y, nsample, sigma_cut):

    tcut = t.copy()
    ycut = y.copy()
    nout=1

    while(nout>0):

        ymed = moving_median(tcut, ycut, nsample)
        resi = ycut - ymed
        sdev = np.sqrt(np.sum(resi**2)/float(len(resi)))

        sigma = np.abs(resi)/sdev
        index = np.where(sigma < sigma_cut)
        nout = len(tcut) - len(index[0])
        tcut = tcut[index[0]]
        ycut = ycut[index[0]]

    return tcut, ycut, ymed, sdev


def moving_average(x, y, nsample):
    v = np.ones(nsample)/float(nsample)
    yav_tmp = np.convolve(y, v, mode='valid')
    f = interp1d(x[int(nsample/2.0): -int(nsample/2.0)], yav_tmp, kind='linear', fill_value='extrapolate')
    return f(x)


def moving_median(x, y, nsample):

    median_tmp = np.zeros(len(x) - nsample)
    x_tmp = np.zeros(len(x) - nsample)
    for i in range(len(x) - nsample):
        median_tmp[i] = np.median(y[i:i+nsample])
        x_tmp[i] = x[i+int(nsample/2.0)]
    f = interp1d(x_tmp, median_tmp, kind='linear', fill_value='extrapolate')

    return f(x)


def polyfunc(p, x, order):
    y = 0
    for i in range(order+1):
        y += p[i]*x**(order-i)

    return y


def calc_chi2_polyfit(p, x, y, ye):
    ymodel = np.polyval(p, x)
    return np.sum( (y - ymodel)**2 / ye / ye )



def weighted_linfit(x, y, ye):

    # y = a + b * x
    w = 1./ye/ye
    sumwx = np.sum(w*x)
    sumwy = np.sum(w*y)
    sumwxy = np.sum(w*x*y)
    sumwxx = np.sum(w*x*x)
    sumw = np.sum(w)
    delta = (sumwx)**2 - sumw * sumwxx
    a = (sumwx * sumwxy - sumwxx * sumwy) / delta
    b = (sumwx * sumwy - sumw * sumwxy) / delta
    ae = np.sqrt(abs(sumwxx / delta))
    be = np.sqrt(abs(sumw / delta))

    return a, b, ae, be

def percentile(array):
    med = np.median(array)
    low1 = np.percentile(array, 15.85)
    hig1 = np.percentile(array, 84.15)
    low2 = np.percentile(array, 2.275)
    hig2 = np.percentile(array, 97.725)
    low3 = np.percentile(array, 0.135)
    hig3 = np.percentile(array, 99.865)
    return med, low1, hig1, low2, hig2, low3, hig3
