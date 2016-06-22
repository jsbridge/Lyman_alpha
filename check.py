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

def check(number):

    La_file = 'LARS07/l07_v3_liLa3.fits'
    Ha_file = 'LARS07/l07_v3_liHa3.fits'
    Hb_file = 'LARS07/l07_v3_liHb3.fits'
    Hb_binfile = 'LARS07/l07_Z0.004_v2d_liHb3.fits'
    Ha_binfile = 'LARS07/l07_Z0.004_v2d_liHa3.fits'
    cont_file = 'LARS07/l07_v3_coLa3.fits'

    
    fscale = 1e-18
    pxscale  = 0.04
    
    la    = pyfits.getdata(La_file) * fscale / pxscale**2 
    ha    = pyfits.getdata(Ha_file) * fscale / pxscale**2
    hb    = pyfits.getdata(Hb_file) * fscale / pxscale**2
    cont  = pyfits.getdata(cont_file) * fscale / pxscale**2

    #mm    = pyfits.getdata(ma)
    #la = la * mm
    #ha = ha * mm
    #hb = hb * mm
    #cont = cont * mm
    
    ha[np.isnan(ha)] = 0
    intLa   = ha*8.7
    la[np.isnan(la)] = 0
    iPosLa    = la > 0.

    haBin = pyfits.getdata(Ha_binfile)
    hbBin = pyfits.getdata(Hb_binfile)

    #k1216 = 11.9845486588   # SMC law
    #k6563 = 3.32579787155
    #k4861 = 4.5980851371

    k1216 = 10.96882      # CCM
    k6563 = 2.45495
    k4861 = 3.51976
    
    dec = haBin[hbBin > 4]/hbBin[hbBin > 4]
    ebv = (2.5/(k4861-k6563)) * np.log10(np.array(dec)/2.86)

    t = plt.hist(dec, bins=sp.arange(0,6,0.1), normed=True)
    num = t[0]
    be = t[1]
    for i, d in enumerate(num):
        if d > 1:
            num[i] = 0
            be[i] = 0
    bc = (be[:-1]+be[1:])/2.

    # Fit lognormal distribution to the extinction
    popt, pcov = curve_fit(lognorm_pdf, bc, num)
    f = lognorm_pdf(bc, popt[0], popt[1], popt[2])
    ff = open('lars'+number+'.results', 'w')
    
    for ttt in xrange(5):
    
        print 'Executing check', ttt+1

        # Draw randomly from E(B-V)
        randIm    = sp.rand(la.shape[0],la.shape[1])
        decRand  = lognorm_cdf_inv(randIm, popt[0], popt[1])

        ebvRand      = (2.5/(k4861-k6563))*sp.log10(decRand/2.86)
        ebvRand[ebvRand<0]=0. 
        fesc1216_lin = 10.**(-0.4*ebvRand*k1216)

        smcont = gaussian_filter(cont, 30)/fscale
        steps = 10**np.linspace(0, np.log10(smcont.max()), 100)[::-1]
        g, xc, yc = np.where([smcont] == np.max(smcont))
        yc += 1
        xc += 1
    
        pp, qq, ss, rawr = [], [], [], []
        pp1, qq1, ss1, rawr1 = [], [], [], []
        cat = []
        cat1 = []
        for rr in np.arange(0, 1.2, 0.1):
            ss.append(rr)

            frac, r = [], []
            for i in xrange(98):
                inner = smcont >= steps[i+1]
                ann = (smcont < steps[i+1]) & (smcont >= steps[i+2])
                inner_avg = smcont[inner].mean()
                ann_avg = smcont[ann].mean()
                frac.append(ann_avg/inner_avg)
                a = np.where(ann, 1, 0)
                xs, ys = np.where(a == 1)
                rs = np.sqrt((xs-xc)**2 + (ys-yc)**2)
                #r.append(np.median(rs))
                r.append(scipy.where(ann, 1, 0).sum()/(2*np.pi))

            ind = (np.abs(np.array(frac)-0.2)).argmin()
            
            bigr = 4*r[ind]
            bigind = (np.abs(np.array(r)-bigr)).argmin()
            smr = rr*r[ind]
            smind = (np.abs(np.array(r)-smr)).argmin()

            bigmask = smcont > steps[bigind]
            smmask = smcont > steps[smind]
            totmask = np.copy(bigmask)
            bigmask[smmask]=False
            
            iMask     = totmask==1
            iMask_hi  = smmask==1
            iMask_low = bigmask==1

            iShow_low = iMask_low & iPosLa
            iShow_hi  = iMask_hi & iPosLa
            iShow     = iMask & iPosLa
            
            val1, xedge, yedge = np.histogram2d(sp.log10(ha[iShow]).real, sp.log10(la[iShow]).real, bins = 20)
    
            k, s, u, u1 = [], [], [], []
            for i in range(0,15,1):
                k.append(i)
                hi_temp_smIntLa = gaussian_filter(intLa, i, mode="wrap")

                q, v, v1 = [], [], []
                for b in range(0,60,1):
                    s.append(b)
                    c, c1 = [], []
                    low_temp_smIntLa = gaussian_filter(intLa, b, mode="wrap")
                    tot_temp_smIntLa = (low_temp_smIntLa * bigmask) + (hi_temp_smIntLa * smmask * fesc1216_lin)
                    
                    val2, xedge, yedge = np.histogram2d(sp.log10(ha[iShow]).real, sp.log10(tot_temp_smIntLa[iShow]).real, bins = 20)
        
                    for j in xrange(val1.shape[0]):
                        for l in xrange(val1.shape[1]):
                            if val1[j,l] < 10:
                                tt = (0.5 + np.sqrt(val1[j,l] + 0.25))**2 
                                c.append((val1[j,l] - val2[j,l])**2/tt)
                            else:
                                c.append((val1[j,l] - val2[j,l])**2/val1[j,l])
                    
                    v.append(np.sum(c))
            
                n = np.array(v).argmin()
                cat.append(v)
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
        b = np.array(cat).argmin()
    
        meow = b/900
        left = b%900
        hiss = left/60
        paw = left%60
        
        print 'Rp =', meow/10.
        print 'hi kernel =', hiss
        print 'low kernel = ', paw
        ff.write('Rp = '+str(meow/10.)+'\n')
        ff.write('hi kernel = '+str(hiss)+'\n')
        ff.write('low kernel = '+str(paw)+'\n')

        kern_hi = hiss
        kern_low = paw
        Rp = meow/10.

        frac, r = [], []
        for i in xrange(98):
            inner = smcont > steps[i+1]
            ann = (smcont < steps[i+1]) & (smcont > steps[i+2])
            inner_avg = smcont[inner].mean()
            ann_avg = smcont[ann].mean()
            frac.append(ann_avg/inner_avg)
            a = np.where(ann, 1, 0)
            xs, ys = np.where(a == 1)
            rs = np.sqrt((xs-xc)**2 + (ys-yc)**2)
            #r.append(np.median(rs))
            r.append(scipy.where(ann, 1, 0).sum()/(2*np.pi))
    
        bigr = 4*r[ind]
        bigind = (np.abs(np.array(r)-bigr)).argmin()
        smr = Rp*r[ind]
        smind = (np.abs(np.array(r)-smr)).argmin()
        
        bigmask = smcont > steps[bigind]
        smmask = smcont > steps[smind]
        totmask = np.copy(bigmask)
        bigmask[smmask]=False
        
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

        im3 = ax3.hexbin(sp.log10(ha[iShow]), sp.log10(la[iShow]), cmap = 'jet', gridsize = 40, vmin=0, vmax = 3000, extent=[-20, -12, -20, -11])
        ax3.xaxis.grid()
        ax3.yaxis.grid()
        ax3.set_xlabel(r"log H$\alpha$ SB (erg/s/cm$^2$/arcsec$^2$)")
        ax3.set_ylabel(r"log L$\alpha$ SB (erg/s/cm$^2$/arcsec$^2$)")

        im4 = ax4.hexbin(sp.log10(ha[iShow]), sp.log10(tot_smIntLa[iShow]), cmap='jet', gridsize = 40, vmin=0, vmax = 3000, extent=[-20, -12, -20, -11])
        ax4.xaxis.grid()
        ax4.yaxis.grid()
        ax4.set_xlabel(r"log H$\alpha$ SB (erg/s/cm$^2$/arcsec$^2$)")
        ax4.set_ylabel(r"log L$\alpha$ SB (erg/s/cm$^2$/arcsec$^2$)")

        fig.subplots_adjust(bottom = 0.12)
    
        cbaxes = fig.add_axes([0.125, 0.053, 0.78, 0.01]) 
        cb = plt.colorbar(im3, cax = cbaxes, orientation='horizontal')
        cbaxes.set_xlabel('Number')
    
        plt.savefig('SB_plots/SBplot_LARS'+number)
        plt.close(fig)
        plt.clf()
        
        im = plt.imshow(tot_smIntLa, origin='lower', vmin = 0, vmax = 5e-14, cmap='jet')
        plt.savefig('SB_plots/galimg_LARS'+number)
        cb = plt.colorbar(im)
        plt.close()
        plt.clf()

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

        ff.write('Summed obs. Halpha: '+str(np.nansum(ha[iShow]))+'\n')
        ff.write('Summed obs. Lya (em): '+str(np.nansum(la[iShow]))+'\n')
        ff.write('Summed obs. Lya (em+abs): '+str(np.nansum(a))+'\n')
        ff.write('Summed sim. Lya: '+str(np.nansum(tot_smIntLa[iShow]))+'\n')
        ff.write('Lya(em+abs)/Ha: '+str(np.nansum(a)/np.nansum(b))+'\n')
        ff.write('Lya(em)/Ha: '+str(np.nansum(la[iShow])/np.nansum(ha[iShow]))+'\n')
        ff.write('Lya(sim)/Ha: '+str(np.nansum(tot_smIntLa[iShow])/np.nansum(ha[iShow]))+'\n')

        sm = gaussian_filter(la, 2)/fscale
        steps = 10**np.linspace(0, np.log10(sm.max()), 100)[::-1]
        g, xc, yc = np.where([sm] == np.max(sm))
    
        frac, r = [], []
        for i in xrange(98):
            inner = sm > steps[i+1]
            ann = (sm < steps[i+1]) & (sm > steps[i+2])
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
        ff.write('Lya R20: '+str(R20)+'\n')
        
        sm = gaussian_filter(ha, 2)/fscale
        steps = 10**np.linspace(0, np.log10(sm.max()), 100)[::-1]
        g, xc, yc = np.where([sm] == np.max(sm))
    
        frac, r = [], []
        for i in xrange(98):
            inner = sm > steps[i+1]
            ann = (sm < steps[i+1]) & (sm > steps[i+2])
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
        ff.write('Ha R20: '+str(R20)+'\n')
    
    ff.close()

    return


def one_check(number):

    La_file = 'LARS07/l07_v3_liLa3.fits'
    Ha_file = 'LARS07/l07_v3_liHa3.fits'
    Hb_file = 'LARS07/l07_v3_liHb3.fits'
    Hb_binfile = 'LARS07/l07_Z0.004_v2d_liHb3.fits'
    Ha_binfile = 'LARS07/l07_Z0.004_v2d_liHa3.fits'
    cont_file = 'LARS07/l07_v3_coLa3.fits'

    kern_hi = 4
    kern_low = 22
    Rp = 1

    fscale = 1e-18
    pxscale  = 0.04

    la    = pyfits.getdata(La_file) * fscale / pxscale**2 
    ha    = pyfits.getdata(Ha_file) * fscale / pxscale**2
    hb    = pyfits.getdata(Hb_file) * fscale / pxscale**2
    cont  = pyfits.getdata(cont_file) * fscale / pxscale**2

    #mm    = pyfits.getdata(ma)
    #la = la * mm
    #ha = ha * mm
    #cont = cont * mm

    la[np.isnan(la)] = 0
    ha[np.isnan(ha)] = 0
    iPosLa  = la > 0.
    intLa   = ha*8.7
    
    haBin = pyfits.getdata(Ha_binfile)
    hbBin = pyfits.getdata(Hb_binfile)

    #k1216 = 11.9845486588     # SMC Law
    #k6563 = 3.32579787155
    #k4861 = 4.5980851371

    k1216 = 10.96882      # CCM
    k6563 = 2.45495
    k4861 = 3.51976
    
    dec = haBin[hbBin > 4]/hbBin[hbBin > 4]
    ebvRand = (2.5/(k4861-k6563)) * np.log10(np.array(dec)/2.86)

    t = plt.hist(dec, bins=sp.arange(0,6,0.1), normed=True)
    plt.close()
    num = t[0]
    be = t[1]
    for i, d in enumerate(num):
        if d > 1:
            num[i] = 0
    #        be[i] = 0
    bc = (be[:-1]+be[1:])/2.

    # Fit lognormal distribution to the extinction
    popt, pcov = curve_fit(lognorm_pdf, bc, num)
    f = lognorm_pdf(bc, popt[0], popt[1], popt[2])

    #plt.plot(bc,f,"go",ls="-")
    #plt.show()
    #plt.close()

    # Draw randomly from E(B-V)
    randIm    = sp.rand(la.shape[0],la.shape[1])
    decRand  = lognorm_cdf_inv(randIm, popt[0], popt[1])

    ebvRand      = (2.5/(k4861-k6563))*sp.log10(decRand/2.86)
    ebvRand[ebvRand<0]=0.
    fesc1216_lin = 10.**(-0.4*ebvRand*k1216)
   
    smcont = gaussian_filter(cont, 30)/fscale
    steps = 10**np.linspace(0, np.log10(smcont.max()), 100)[::-1]
    g, yc, xc = np.where([smcont] == np.max(smcont))
    yc += 1
    xc += 1

    frac, r = [], []
    for i in xrange(98):
        inner = smcont > steps[i+1]
        ann = (smcont < steps[i+1]) & (smcont > steps[i+2])
        inner_avg = smcont[inner].mean()
        ann_avg = smcont[ann].mean()
        frac.append(ann_avg/inner_avg)
        a = np.where(ann, 1, 0)
        xs, ys = np.where(a == 1)
        rs = np.sqrt((xs-xc)**2 + (ys-yc)**2)
        #r.append(np.mean(rs))
        r.append(scipy.where(ann, 1, 0).sum()/(2*np.pi))
    
    ind = (np.abs(np.array(frac)-0.2)).argmin()
    bigr = 3*r[ind]
    bigind = (np.abs(np.array(r)-bigr)).argmin()
    smr = Rp*r[ind]
    smind = (np.abs(np.array(r)-smr)).argmin()
    
    smmask = smcont > steps[smind]
    bigmask = smcont > steps[bigind]

    totmask = np.copy(bigmask)
    bigmask[smmask]=False

    iMask     = totmask==1
    iMask_hi  = smmask==1
    iMask_low = bigmask==1
    iPosLa    = la>0.

    iShow_low = iMask_low & iPosLa
    iShow_hi  = iMask_hi & iPosLa
    iShow     = iMask & iPosLa

    smIntLa_hi = gaussian_filter(intLa, kern_hi, mode='wrap')
    smIntLa_low = gaussian_filter(intLa, kern_low, mode='wrap')
    tot_smIntLa = (smIntLa_low * bigmask) + (smIntLa_hi * smmask *fesc1216_lin)

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

    im3 = ax3.hexbin(sp.log10(ha[iShow]), sp.log10(la[iShow]), cmap = 'viridis', gridsize = 40, vmin=0, vmax = 2500, extent=[-20, -12, -20, -11])
    ax3.xaxis.grid()
    ax3.yaxis.grid()
    ax3.set_xlabel(r"log H$\alpha$ SB (erg/s/cm$^2$/arcsec$^2$)")
    ax3.set_ylabel(r"log L$\alpha$ SB (erg/s/cm$^2$/arcsec$^2$)")

    im4 = ax4.hexbin(sp.log10(ha[iShow]), sp.log10(tot_smIntLa[iShow]), cmap='viridis', gridsize = 40, vmin=0, vmax = 2500, extent=[-20, -12, -20, -11])
    ax4.xaxis.grid()
    ax4.yaxis.grid()
    ax4.set_xlabel(r"log H$\alpha$ SB (erg/s/cm$^2$/arcsec$^2$)")
    ax4.set_ylabel(r"log L$\alpha$ SB (erg/s/cm$^2$/arcsec$^2$)")

    fig.subplots_adjust(bottom = 0.12)
    
    cbaxes = fig.add_axes([0.125, 0.053, 0.78, 0.01]) 
    cb = plt.colorbar(im3, cax = cbaxes, orientation='horizontal')
    cbaxes.set_xlabel('log Number')
    
    plt.savefig('SB_plots/SBplot_LARS'+str(number))
    plt.close(fig)
    plt.clf()
    
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

    sm = gaussian_filter(la, 2)/fscale
    steps = 10**np.linspace(0, np.log10(sm.max()), 100)[::-1]
    g, xc, yc = np.where([sm] == np.max(sm))
    
    frac, r = [], []
    for i in xrange(98):
        inner = sm > steps[i+1]
        ann = (sm < steps[i+1]) & (sm > steps[i+2])
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
    
    sm = gaussian_filter(ha, 2)/fscale
    steps = 10**np.linspace(0, np.log10(sm.max()), 100)[::-1]
    g, xc, yc = np.where([sm] == np.max(sm))
    
    frac, r = [], []
    for i in xrange(98):
        inner = sm > steps[i+1]
        ann = (sm < steps[i+1]) & (sm > steps[i+2])
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

    hik, lowk, hi_Rp, ha, lya_emabs, lya_em, lya_sim, rat_emabs, rat_em, rat_sim, hayes, haR20, laR20, z, scale, hrat = np.loadtxt('ratios4.dat', unpack=True, usecols=(2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17))
    labels = np.genfromtxt('ratios4.dat', unpack=True, usecols=(1), dtype='str')
    
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
    ax4.set_xlim(0, 6)
    ax4.set_ylim(0, 2)
    for label, p, y in zip(labels, laR20/haR20, lowk):
        ax4.annotate(label, xy = (p, y),color='blue', xytext = (20, -30), textcoords = 'offset points', ha = 'right', va = 'bottom', arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))


    #slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(laR20/haR20,lowk)
    #bb = slope*x+intercept
    #ax4.plot(x, bb, 'r-')
    ind = np.where(labels != '07')
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress((laR20/haR20)[ind],lowk[ind])
    bb = slope*x+intercept
    ax4.plot(x, bb, 'k-')
        
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


def comp_ratios(number):

    La_file = 'LARS01/l01_v3_liLa3.fits'
    Ha_file = 'LARS01/l01_v3_liHa3.fits'
    Hb_file = 'LARS01/l01_v3_liHb3.fits'
    Hb_binfile = 'LARS01/l01_v2d_20111124_liHb2.fits'
    Ha_binfile = 'LARS01/l01_v2d_20111124_liHa2.fits'
    cont_file = 'LARS01/l01_v3_coLa3.fits'

    la    = pyfits.getdata(La_file)
    ha    = pyfits.getdata(Ha_file)
    hb    = pyfits.getdata(Hb_file)
    cont  = pyfits.getdata(cont_file)/0.04**2

    smcont = gaussian_filter(cont, 30)
    steps = 10**np.linspace(-1, np.log10(smcont.max()), 100)[::-1]
    frac, r = [], []
    for i in xrange(98):
        inner = smcont > steps[i+1]
        ann = (smcont < steps[i+1]) & (smcont > steps[i+2])
        inner_avg = smcont[inner].mean()
        ann_avg = smcont[ann].mean()
        frac.append(ann_avg/inner_avg)
        r.append(scipy.where(ann, 1, 0).sum()/(2*np.pi))

    ind = (np.abs(np.array(frac)-0.2)).argmin()
    bigr = 2*r[ind]
    bigind = (np.abs(np.array(r)-bigr)).argmin()
    mask = smcont > steps[bigind]
    
    la = la*mask
    ha = ha*mask
    hb = hb*mask

    dim = 500
    la = bin_ndarray(la, new_shape=(dim,dim), operation='mean')
    ha = bin_ndarray(ha, new_shape=(dim,dim), operation='mean')
    hb = bin_ndarray(hb, new_shape=(dim,dim), operation='mean')
    
    nn = la/ha
    mm = ha/hb
    Ha = ha
    La = la
    
    nn = nn[hb > 4]
    mm = mm[hb > 4]
    La = La[hb > 4]
    Ha = Ha[hb > 4]
    
    n = nn[~np.isnan(nn)]
    m = mm[~np.isnan(nn)]
    nn = n[~np.isnan(m)]
    mm = m[~np.isnan(m)]
    Laa = La[~np.isnan(nn)]
    Haa = Ha[~np.isnan(nn)]
    La = Laa[~np.isnan(m)]
    Ha = Haa[~np.isnan(m)]

    n = [q for q in nn if q != 0.0]
    m = [mm[i] for i, q in enumerate(nn) if q != 0.0]
    nn = [n[i] for i, q in enumerate(m) if q != 0.0]
    mm = [q for q in m if q != 0.0]
    Haa = [Ha[i] for i, q in enumerate(nn) if q != 0.0]
    Laa = [La[i] for i, q in enumerate(nn) if q != 0.0]
    Ha = [Haa[i] for i, q in enumerate(m) if q != 0.0]
    La = [Laa[i] for i, q in enumerate(m) if q != 0.0]

    popt,pcov = curve_fit(expfunc, mm, nn)
    h = (np.arange(1.5, 6, 0.1))
    
    k1216 = 10.96882      # CCM
    k6563 = 2.45495
    k4861 = 3.51976

    ebv = (2.5/(k4861-k6563)) * np.log10(np.array(mm)/2.86)

    fesc = La/(8.7*Ha*10**(-0.4*ebv*k6563))
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(mm, nn, 'ko', ms = 4)
    ax.plot(h, expfunc(h, *popt), 'b-',lw=4)
    #ax.set_yscale('log')
    #ax.set_xscale('log')
    ax.plot([0, 5.15], [20, -5], 'r--', label='Cardelli', lw = 4)
    ax.set_xlim(1.5, 8)
    ax.set_ylim(0, 11)
    #plt.tick_params(axis='x', which='minor')
    ax.xaxis.set_minor_formatter(FormatStrFormatter("%.1f"))
    ax.yaxis.set_minor_formatter(FormatStrFormatter("%.1f"))
    ax.set_xlabel(r'H$\alpha$/H$\beta$')
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.set_ylabel(r'Ly$\alpha$/H$\alpha$')
    plt.legend()
    plt.savefig('ratios/ratios'+str(number))
    plt.close(fig)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ebv, nn, 'ko', ms = 4)
    ax.xaxis.set_minor_formatter(FormatStrFormatter("%.1f"))
    ax.yaxis.set_minor_formatter(FormatStrFormatter("%.1f"))
    ax.set_xlabel(r'E(B-V)')
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.set_ylabel(r'Ly$\alpha$/H$\alpha$')
    plt.legend()
    plt.savefig('ratios/EBV_'+str(number))
    plt.close(fig)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ebv, fesc, 'ko', ms = 4)
    ax.xaxis.set_minor_formatter(FormatStrFormatter("%.1f"))
    ax.yaxis.set_minor_formatter(FormatStrFormatter("%.1f"))
    ax.set_xlabel(r'E(B-V)')
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.set_ylabel(r'f^{esc}_{Ly$\alpha$}')
    plt.legend()
    plt.savefig('ratios/fesc_'+str(number))
    plt.close(fig)


    return

    
def bin_ndarray(ndarray, new_shape, operation='sum'):
    """
    Bins an ndarray in all axes based on the target shape, by summing or
        averaging.

    Number of output dimensions must match number of input dimensions and 
        new axes must divide old ones.

    Example
    -------
    >>> m = np.arange(0,100,1).reshape((10,10))
    >>> n = bin_ndarray(m, new_shape=(5,5), operation='sum')
    >>> print(n)

    [[ 22  30  38  46  54]
     [102 110 118 126 134]
     [182 190 198 206 214]
     [262 270 278 286 294]
     [342 350 358 366 374]]

    """
    operation = operation.lower()
    if not operation in ['sum', 'mean']:
        raise ValueError("Operation not supported.")
    if ndarray.ndim != len(new_shape):
        raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape,
                                                           new_shape))
    compression_pairs = [(d, c//d) for d,c in zip(new_shape,
                                                  ndarray.shape)]
    flattened = [l for p in compression_pairs for l in p]
    ndarray = ndarray.reshape(flattened)
    for i in range(len(new_shape)):
        op = getattr(ndarray, operation)
        ndarray = op(-1*(i+1))
    return ndarray


def expfunc(x, a, b, c):
    return a * np.exp(-b * x) + c


