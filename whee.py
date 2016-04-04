import numpy as np
import matplotlib.pyplot as plt
import pyfits
import scipy.stats
from scipy.ndimage.filters import gaussian_filter
from scipy.optimize import curve_fit
from glob import glob
import os
import scipy as sp
from scipy.special import erf, erfinv
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from astropy.convolution import Gaussian2DKernel, convolve, convolve_fft
from astropy.convolution.kernels import Model2DKernel
import cv2


def draw_recline(ax):
    ax.plot([-20,20], [-19,21], "r-")
    ax.plot([-20,20], [-20,20], "r--")
    ax.plot([-20,20], [-21,19], "r:")
    ax.xaxis.grid()
    ax.yaxis.grid()
    ax.set_xlabel(r"log H$\alpha$ SB (erg/s/cm$^2$/arcsec$^2$)")
    ax.set_ylabel(r"log L$\alpha$ SB (erg/s/cm$^2$/arcsec$^2$)")

def lognorm_pdf(x, sigma, mu, scale):
    return scale /x/sigma/sp.sqrt(2.*sp.pi) * sp.e**((sp.log(x)-mu)**2./-2./sigma**2.)

def lognorm_cdf(x, sigma, mu, scale):
    return 0.5 + 0.5 * erf( (sp.log(x)-mu) / sp.sqrt(2) / sigma)

def lognorm_cdf_inv(y, sigma, mu):
    return sp.exp(erfinv(2.*y-1.)*sp.sqrt(2)*sigma + mu)


def stupid():
    
    # Read in the files and scale to flux units then surface brightness

    La_file = 'LARS01/l01_Z0.008_ubin_liLa3.fits'
    Ha_file = 'LARS01/l01_Z0.008_ubin_liHa3.fits'
    Ha_binfile = 'LARS01/l01_v2d_20111124_liHa2.fits'
    Hb_binfile = 'LARS01/l01_v2d_20111124_liHb2.fits'
    cont_file = 'LARS01/l01_Z0.008_ubin_coLa3.fits'
    
    fscale = 1e-18
    pxscale  = 0.04

    la    = pyfits.getdata(La_file) * fscale / pxscale**2 
    ha    = pyfits.getdata(Ha_file) * fscale / pxscale**2
    cont  = pyfits.getdata(cont_file) * fscale / pxscale**2

    ha[np.isnan(ha)] = 0
    intLa   = ha*8.7
    iPosLa    = la>0.

    haBin = pyfits.getdata(Ha_binfile)
    hbBin = pyfits.getdata(Hb_binfile)

    k1216 = 11.9845486588
    k6563 = 3.32579787155
    k4861 = 4.5980851371

    dec = haBin[hbBin > 6]/hbBin[hbBin > 6]
    ebv = (2.5/(k4861-k6563)) * np.log10(np.array(dec)/2.86)
    pyfits.writeto('test.fits', haBin/hbBin)

    t = plt.hist(dec, bins=sp.arange(0,6,0.1), normed=True)
    num = t[0]
    be = t[1]
    bc = (be[:-1]+be[1:])/2.

    # Fit lognormal distribution to the extinction
    popt, pcov = curve_fit(lognorm_pdf, bc, num)
    f = lognorm_pdf(bc, popt[0], popt[1], popt[2])

    # Draw randomly from E(B-V)

    randIm    = sp.rand(la.shape[0],la.shape[1])
    decRand  = lognorm_cdf_inv(randIm, popt[0], popt[1])

    ebvRand      = (2.5/(k4861-k6563))*sp.log10(decRand/2.86)
    ebvRand[ebvRand<0]=0. 
    fesc1216_lin = 10.**(-0.4*ebvRand*k1216)
    fesc1216_3x  = 10.**(-0.4*ebvRand*k1216*1.15)
    fesc1216_10x = 10.**(-0.4*ebvRand*k1216*3.0)

    smcont = gaussian_filter(cont, 1)/fscale
    steps = 10**np.linspace(0, np.log10(smcont.max()), 35)[::-1]
    print steps
    g, xc, yc = np.where([smcont] == np.max(smcont))
    
    pp, qq, ss, rawr = [], [], [], []
    for rr in [0, 0.1, 0.2, 0.3,0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        ss.append(rr)

        frac, r = [], []
        for i in xrange(33):
            inner = smcont >= steps[i+1]
            ann = (smcont < steps[i+1]) & (smcont >= steps[i+2])
            inner_avg = smcont[inner].mean()
            ann_avg = smcont[ann].mean()
            frac.append(ann_avg/inner_avg)
            a = np.where(ann, 1, 0)
            xs, ys = np.where(a == 1)
            rs = np.sqrt((xs-xc)**2 + (ys-yc)**2)
            r.append(np.median(rs))

        plt.plot(r, steps[:-2])
        plt.show()
        plt.clf()
        
        ind = (np.abs(np.array(frac)-0.2)).argmin()
        bigr = 4*r[ind]
        bigind = (np.abs(np.array(r)-bigr)).argmin()
        smr = rr*r[ind]
        smind = (np.abs(np.array(r)-smr)).argmin()

        bigmask = smcont > steps[bigind]
        smmask = smcont > steps[smind]
        totmask = smcont >= steps[bigind]
        part1 = smcont > steps[smind]
        bigmask[part1]=False
    
        iMask     = totmask==1
        iMask_hi  = smmask==1
        iMask_low = bigmask==1

        iShow_low = iMask_low & iPosLa
        iShow_hi  = iMask_hi & iPosLa
        iShow     = iMask & iPosLa

        val1, xedge, yedge = np.histogram2d(sp.log10(ha[iShow]).real, sp.log10(la[iShow]).real, bins = 40)
    
        k, s, u = [], [], []
        for i in range(0,30,1):
            k.append(i)
            hi_temp_smIntLa = gaussian_filter(intLa, i, mode="wrap")

            q, v = [], []
            for b in range(0,40,1):
                s.append(b)
                c = []
                low_temp_smIntLa = gaussian_filter(intLa, b, mode="wrap")
                tot_temp_smIntLa = (low_temp_smIntLa * bigmask) + (hi_temp_smIntLa * smmask * fesc1216_lin)
            
                val2, xedge, yedge = np.histogram2d(sp.log10(ha[iShow]).real, sp.log10(tot_temp_smIntLa[iShow]).real, bins = 40)
                print val2
                print val1
                for j in xrange(val1.shape[0]):
                    for l in xrange(val1.shape[1]):
                        #tt = (0.5 + np.sqrt(val1[j,l] + 0.25))**2
                        tt = val1[j,l]
                        c.append((val1[j,l] - val2[j,l])**2/tt)
                    
                v.append(np.sum(c))
            
            n = np.array(v).argmin()
            print v[n]
            rawr.append(v[n])
            u.append(v[n])
            pp.append(s[n])
            print 'Current low kernel = ', s[n]
    
        m = np.array(u).argmin()
        print 'Hi Kernel = ', k[m]
        print 'Current small Rp = ', rr
        qq.append(k[m])

    a = np.array(rawr).argmin()
    print a
    print rawr[a]


def plot():

    La_file = 'LARS01/l01_Z0.008_ubin_liLa3.fits'
    Ha_file = 'LARS01/l01_Z0.008_ubin_liHa3.fits'
    Hb_binfile = 'LARS01/l01_v2d_20111124_liHb2.fits'
    Ha_binfile = 'LARS01/l01_v2d_20111124_liHa2.fits'
    cont_file = 'LARS01/l01_Z0.008_ubin_coLa3.fits'
    
    fscale = 1e-18
    pxscale  = 0.04

    la    = pyfits.getdata(La_file) * fscale / pxscale**2 
    ha    = pyfits.getdata(Ha_file) * fscale / pxscale**2
    cont  = pyfits.getdata(cont_file) * fscale / pxscale**2

    ha[np.isnan(ha)] = 0
    intLa   = ha*8.7
    
    haBin = pyfits.getdata(Ha_binfile)
    hbBin = pyfits.getdata(Hb_binfile)

    k1216 = 11.9845486588
    k6563 = 3.32579787155
    k4861 = 4.5980851371
    
    dec = haBin[hbBin > 3]/hbBin[hbBin > 3]
    ebv = (2.5/(k4861-k6563)) * np.log10(np.array(dec)/2.86)

    t = plt.hist(dec, bins=sp.arange(0,6,0.1), normed=True)
    plt.close()
    num = t[0]
    be = t[1]
    bc = (be[:-1]+be[1:])/2.

    # Fit lognormal distribution to the extinction
    popt, pcov = curve_fit(lognorm_pdf, bc, num)
    f = lognorm_pdf(bc, popt[0], popt[1], popt[2])

    # Draw randomly from E(B-V)
    randIm    = sp.rand(la.shape[0],la.shape[1])
    decRand  = lognorm_cdf_inv(randIm, popt[0], popt[1])

    ebvRand      = (2.5/(k4861-k6563))*sp.log10(decRand/2.86)
    ebvRand[ebvRand<0]=0. 
    fesc1216_lin = 10.**(-0.4*ebvRand*k1216)
    fesc1216_3x  = 10.**(-0.4*ebvRand*k1216*1.15)
    fesc1216_10x = 10.**(-0.4*ebvRand*k1216*3.0)

    smcont = gaussian_filter(cont, 20)/fscale
    steps = 10**np.linspace(0, np.log10(smcont.max()), 50)[::-1]
    g, xc, yc = np.where([smcont] == np.max(smcont))
    
    kern_hi = 14
    kern_low = 9
    Rp = 1.4

    frac, r = [], []
    for i in xrange(47):
        inner = smcont > steps[i+2]
        ann = (smcont < steps[i+2]) & (smcont > steps[i+3])
        inner_avg = smcont[inner].mean()
        ann_avg = smcont[ann].mean()
        frac.append(ann_avg/inner_avg)
        a = np.where(ann, 1, 0)
        xs, ys = np.where(a == 1)
        rs = np.sqrt((xs-xc)**2 + (ys-yc)**2)
        r.append(np.median(rs))
    
    ind = (np.abs(np.array(frac)-0.2)).argmin()
    
    bigr = 4*r[ind]
    bigind = (np.abs(np.array(r)-bigr)).argmin()
    smr = Rp*r[ind]
    smind = (np.abs(np.array(r)-smr)).argmin()

    bigmask = smcont > steps[bigind]
    smmask = smcont > steps[smind]
    totmask = smcont >= steps[bigind]
    part1 = smcont > steps[smind]
    bigmask[part1]=False

    #plt.imshow(bigmask, origin='lower')
    #plt.show()
    #plt.close()
    
    iMask     = totmask==1
    iMask_hi  = smmask==1
    iMask_low = bigmask==1
    iPosLa    = la>0.

    iShow_low = iMask_low & iPosLa
    iShow_hi  = iMask_hi & iPosLa
    iShow     = iMask & iPosLa

    smIntLa_hi = gaussian_filter(intLa, kern_hi, mode='wrap')
    smIntLa_low = gaussian_filter(intLa, kern_low, mode='wrap')
    tot_smIntLa = (smIntLa_low * bigmask) + (smIntLa_hi * smmask * fesc1216_lin)

    fig = plt.figure(figsize=(10,10))
    ax1 = fig.add_subplot(2,2,1, xlim=[-20,-12], ylim=[-20,-11])
    ax2 = fig.add_subplot(2,2,2, xlim=[-20,-12], ylim=[-20,-11])
    ax3 = fig.add_subplot(2,2,3,)
    ax4 = fig.add_subplot(2,2,4,)
    
    ax1.plot(sp.log10(ha[iShow]), sp.log10(la     [iShow]), "k.")
    ax2.plot(sp.log10(ha[iShow]), sp.log10(tot_smIntLa[iShow]), "k.")
    draw_recline(ax1)
    draw_recline(ax2)
    ax2.text(0.1, 0.9, 'Kernel (bright,dim) = '+str(kern_hi)+', '+str(kern_low), ha='left', va='center', transform=ax2.transAxes)
    ax2.text(0.1, 0.85, 'Fraction = '+str(Rp), ha='left', va='center', transform=ax2.transAxes)

    im3 = ax3.hexbin(sp.log10(ha[iShow]), sp.log10(la[iShow]), cmap = 'jet', gridsize = 40, vmin=0, vmax = 2500, extent=[-20, -12, -20, -11])
    #fig.colorbar(im3, ax=ax3, orientation='horizontal')
    ax3.xaxis.grid()
    ax3.yaxis.grid()
    ax3.set_xlabel(r"log H$\alpha$ SB (erg/s/cm$^2$/arcsec$^2$)")
    ax3.set_ylabel(r"log L$\alpha$ SB (erg/s/cm$^2$/arcsec$^2$)")

    im4 = ax4.hexbin(sp.log10(ha[iShow]), sp.log10(tot_smIntLa[iShow]), cmap='jet', gridsize = 40, vmin=0, vmax = 2500, extent=[-20, -12, -20, -11])
    #fig.colorbar(im4, ax=ax4, orientation='horizontal')
    ax4.xaxis.grid()
    ax4.yaxis.grid()
    ax4.set_xlabel(r"log H$\alpha$ SB (erg/s/cm$^2$/arcsec$^2$)")
    ax4.set_ylabel(r"log L$\alpha$ SB (erg/s/cm$^2$/arcsec$^2$)")

    fig.subplots_adjust(bottom = 0.12)
    
    cbaxes = fig.add_axes([0.125, 0.053, 0.78, 0.01]) 
    cb = plt.colorbar(im3, cax = cbaxes, orientation='horizontal')
    cbaxes.set_xlabel('log Number')
    
    plt.savefig('SB_plots/SBplot_LARS01')
    plt.close(fig)

    a = la*totmask
    a[np.where(a == 0)] = np.nan
    b = ha*totmask
    b[np.where(b == 0)] = np.nan
    
    print 'Summed obs. Halpha:', np.nansum(ha[iShow])
    print 'Summed obs. Lya (em):', np.nansum(la[iShow])
    print 'Summed obs. Lya (em+abs):', np.nansum(a)
    print 'Summed sim. Lya:', np.nansum(tot_smIntLa[iShow])
    print 'Lya(em+abs)/Ha:', np.nansum(a)/np.nansum(b)
    print 'Lya(em)/Ha:', np.nansum(la[iShow])/np.nansum(ha[iShow])
    print 'Lya(sim)/Ha:', np.nansum(tot_smIntLa[iShow])/np.nansum(ha[iShow])


    sm = gaussian_filter(tot_smIntLa, 20)/fscale
    steps = 10**np.linspace(0, np.log10(sm.max()), 50)[::-1]
    g, xc, yc = np.where([sm] == np.max(sm))
    
    frac, r = [], []
    for i in xrange(47):
        inner = sm > steps[i+2]
        ann = (sm < steps[i+2]) & (sm > steps[i+3])
        inner_avg = sm[inner].mean()
        ann_avg = sm[ann].mean()
        frac.append(ann_avg/inner_avg)
        a = np.where(ann, 1, 0)
        xs, ys = np.where(a == 1)
        rs = np.sqrt((xs-xc)**2 + (ys-yc)**2)
        r.append(np.median(rs))

    ind = (np.abs(np.array(frac)-0.2)).argmin()

    R20 = r[ind]
    print 'Lya R20:', R20
    
    sm = gaussian_filter(ha*totmask, 20)/fscale
    steps = 10**np.linspace(0, np.log10(sm.max()), 50)[::-1]
    g, xc, yc = np.where([sm] == np.max(sm))
    
    frac, r = [], []
    for i in xrange(47):
        inner = sm > steps[i+2]
        ann = (sm < steps[i+2]) & (sm > steps[i+3])
        inner_avg = sm[inner].mean()
        ann_avg = sm[ann].mean()
        frac.append(ann_avg/inner_avg)
        a = np.where(ann, 1, 0)
        xs, ys = np.where(a == 1)
        rs = np.sqrt((xs-xc)**2 + (ys-yc)**2)
        r.append(np.median(rs))

    ind = (np.abs(np.array(frac)-0.2)).argmin()
    
    R20 = np.median(r[ind])
    print 'Ha R20:', R20

    return


def comp_plots():

    hik, lowk, hi_Rp, ha, lya_emabs, lya_em, lya_sim, rat_emabs, rat_em, rat_sim, hayes, haR20, laR20, z, scale, hrat = np.loadtxt('ratios.dat', unpack=True, usecols=(2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17))
    labels = np.genfromtxt('ratios2.dat', unpack=True, usecols=(1), dtype='str')

    haR20 = haR20 * 0.04 * scale  # R20 in kpc
    laR20 = laR20 * 0.04 * scale
    lowk = lowk * 0.04 * scale
    
    x = np.arange(0, 20, 0.01)
    
    
    fig = plt.figure(figsize=(16,10))
    ax1 = fig.add_subplot(2,3,1)
    ax2 = fig.add_subplot(2,3,2)
    ax3 = fig.add_subplot(2,3,3,)
    ax4 = fig.add_subplot(2,3,4,)
    ax5 = fig.add_subplot(2,3,5,)

    ax1.plot(rat_sim, rat_em, 'ro')
    ax1.set_xlabel('simulated Lya(em)/Ha')
    ax1.set_ylabel('measured Lya(em)/Ha')
    ax1.plot(x, x, 'r--')
    ax1.set_xlim(1,11)
    ax1.set_ylim(1,11)
    for label, p, y in zip(labels, rat_sim, rat_em):
        ax1.annotate(label, xy = (p, y), color='blue', xytext = (20, -30), textcoords = 'offset points', ha = 'right', va = 'bottom', arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

    ax2.plot(lya_sim, lya_em, 'go')
    ax2.set_xlabel('simulated Lya(em) SB')
    ax2.set_ylabel('measured Lya(em) SB')
    ax2.plot(x, x, 'g--')
    ax2.set_xlim(0, 10e-10)
    ax2.set_ylim(0, 10e-10)
    for label, p, y in zip(labels, lya_sim, lya_em):
        ax2.annotate(label, xy = (p, y),color='blue', xytext = (20, -30), textcoords = 'offset points', ha = 'right', va = 'bottom', arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

    ax3.plot(rat_sim, hayes, 'co')
    ax3.set_xlabel('simulated Lya/Ha')
    ax3.set_ylabel('Hayes+14 measured Lya/Ha')
    ax3.plot(x, x, 'c--')
    ax3.set_xlim(0, 9)
    ax3.set_ylim(0, 9)
    for label, p, y in zip(labels, rat_sim, hayes):
        ax3.annotate(label, xy = (p, y), color='blue',xytext = (20, -30), textcoords = 'offset points', ha = 'right', va = 'bottom', arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

    ax4.plot(laR20/haR20, lowk, 'ko')
    ax4.set_xlabel('Lya R20/Ha R20')
    ax4.set_ylabel('outer kernel')
    ax4.set_xlim(0.5, 4.6)
    for label, p, y in zip(labels, laR20/haR20, lowk):
        ax4.annotate(label, xy = (p, y),color='blue', xytext = (20, -30), textcoords = 'offset points', ha = 'right', va = 'bottom', arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

    ax5.plot(laR20/haR20, hrat, 'yo')
    ax5.set_xlabel('Lya R20/Ha R20')
    ax5.set_ylabel('Hayes+13 ratio')
    ax5.set_ylim(0, 4)
    ax5.set_xlim(0, 4)
    ax5.plot(x, x, 'y--')
    for label, p, y in zip(labels, laR20/haR20, hrat):
        ax5.annotate(label, xy = (p, y),color='blue', xytext = (20, -30), textcoords = 'offset points', ha = 'right', va = 'bottom', arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

    plt.savefig('comparisons')
    plt.close(fig)

    return
