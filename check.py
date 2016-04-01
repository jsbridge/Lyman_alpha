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

def check():

    La_file = 'LARS10/l10_Z0.008_ubin_liLa3.fits'
    Ha_file = 'LARS10/l10_Z0.008_ubin_liHa3.fits'
    Hb_binfile = 'LARS10/l10_Z0.008_v2d_liHb3.fits'
    Ha_binfile = 'LARS10/l10_Z0.008_v2d_liHa3.fits'
    cont_file = 'LARS10/l10_Z0.008_ubin_coLa3.fits'
    ma = 'LARS10/mask10.fits'

    fscale = 1e-18
    pxscale  = 0.04
    
    la    = pyfits.getdata(La_file) * fscale / pxscale**2 
    ha    = pyfits.getdata(Ha_file) * fscale / pxscale**2
    cont  = pyfits.getdata(cont_file) * fscale / pxscale**2
    mm    = pyfits.getdata(ma)

    la = la*mm
    ha = ha*mm
    cont = cont*mm
    
    ha[np.isnan(ha)] = 0
    intLa   = ha*8.7
    la[np.isnan(la)] = 0
    iPosLa    = la>0.

    haBin = pyfits.getdata(Ha_binfile)
    hbBin = pyfits.getdata(Hb_binfile)

    k1216 = 11.9845486588
    k6563 = 3.32579787155
    k4861 = 4.5980851371
    
    dec = haBin[hbBin > 4]/hbBin[hbBin > 4]
    ebv = (2.5/(k4861-k6563)) * np.log10(np.array(dec)/2.86)

    t = plt.hist(dec, bins=sp.arange(0,6,0.1), normed=True)
    num = t[0]
    be = t[1]
    bc = (be[:-1]+be[1:])/2.

    # Fit lognormal distribution to the extinction
    popt, pcov = curve_fit(lognorm_pdf, bc, num)
    f = lognorm_pdf(bc, popt[0], popt[1], popt[2])
    ff = open('lars10.results', 'w')
    
    for ttt in xrange(10):
    
        print 'Executing check', ttt+1

        # Draw randomly from E(B-V)
        randIm    = sp.rand(la.shape[0],la.shape[1])
        decRand  = lognorm_cdf_inv(randIm, popt[0], popt[1])

        ebvRand      = (2.5/(k4861-k6563))*sp.log10(decRand/2.86)
        ebvRand[ebvRand<0]=0. 
        fesc1216_lin = 10.**(-0.4*ebvRand*k1216)

        smcont = gaussian_filter(cont, 4)/fscale
        steps = 10**np.linspace(0, np.log10(smcont.max()), 30)[::-1]
        g, xc, yc = np.where([smcont] == np.max(smcont))
    
        pp, qq, ss, rawr = [], [], [], []
        pp1, qq1, ss1, rawr1 = [], [], [], []
        cat = []
        cat1 = []
        for rr in np.arange(0, 2.5, 0.1):
            ss.append(rr)

            frac, r = [], []
            for i in xrange(28):
                inner = smcont >= steps[i+1]
                ann = (smcont < steps[i+1]) & (smcont >= steps[i+2])
                inner_avg = smcont[inner].mean()
                ann_avg = smcont[ann].mean()
                frac.append(ann_avg/inner_avg)
                a = np.where(ann, 1, 0)
                xs, ys = np.where(a == 1)
                rs = np.sqrt((xs-xc)**2 + (ys-yc)**2)
                r.append(np.median(rs))

            ind = (np.abs(np.array(frac)-0.2)).argmin()

            #plt.plot(r, steps[:-2])
            #plt.show()
            #plt.clf()
            #print r[ind]
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
            
            val1, xedge, yedge = np.histogram2d(sp.log10(ha[iShow]).real, sp.log10(la[iShow]).real, bins = 30)
    
            k, s, u, u1 = [], [], [], []
            for i in range(0,15,1):
                k.append(i)
                hi_temp_smIntLa = gaussian_filter(intLa, i, mode="wrap")

                q, v, v1 = [], [], []
                for b in range(0,30,1):
                    s.append(b)
                    c, c1 = [], []
                    low_temp_smIntLa = gaussian_filter(intLa, b, mode="wrap")
                    tot_temp_smIntLa = (low_temp_smIntLa * bigmask) + (hi_temp_smIntLa * smmask * fesc1216_lin)
                    
                    val2, xedge, yedge = np.histogram2d(sp.log10(ha[iShow]).real, sp.log10(tot_temp_smIntLa[iShow]).real, bins = 30)
        
                    for j in xrange(val1.shape[0]):
                        for l in xrange(val1.shape[1]):
                            if val1[j,l] < 10:
                                tt = (0.5 + np.sqrt(val1[j,l] + 0.25))**2 
                                c1.append((val1[j,l] - val2[j,l])**2/tt)
                                c.append(0)
                            else:
                                c1.append((val1[j,l] - val2[j,l])**2/val1[j,l])
                                c.append((val1[j,l] - val2[j,l])**2/val1[j,l])
                    
                    v.append(np.sum(c))
                    v1.append(np.sum(c1))
            
                n = np.array(v).argmin()
                n1 = np.array(v1).argmin()
                cat.append(v)
                cat1.append(v1)
                print v[n], v1[n1]
                rawr.append(v[n])
                rawr1.append(v1[n1])
                u.append(v[n])
                u1.append(v1[n1])
                pp.append(s[n])
                pp1.append(s[n1])
                print 'Current low kernel = ', s[n], s[n1]
    
            m = np.array(u).argmin()
            m1 = np.array(u1).argmin()
            print 'Hi Kernel = ', k[m], k[m1]
            print 'Current small Rp = ', rr
            qq.append(k[m])
            qq1.append(k[m1])

        a = np.array(rawr).argmin()
        a1 = np.array(rawr1).argmin()
        b = np.array(cat).argmin()
        b1 = np.array(cat1).argmin()
        #print a
        #print rawr[a]
        #print b
        meow = b/450
        meow1 = b/450
        left = b%450
        left1 = b%450
        hiss = left/30
        hiss1 = left1/30
        paw = left%30
        paw1 = left1%30
        
        print 'Rp =', meow/10., meow1/10.
        print 'hi kernel =', hiss, hiss1
        print 'low kernel = ', paw, paw1
        ff.write('Rp = '+str(meow/10.)+' '+str(meow1/10.)+'\n')
        ff.write('hi kernel = '+str(hiss)+' '+str(hiss1)+'\n')
        ff.write('low kernel = '+str(paw)+' '+str(paw1)+'\n')

        kern_hi = hiss
        kern_low = paw
        Rp = meow/10.

        frac, r = [], []
        for i in xrange(28):
            inner = smcont > steps[i+1]
            ann = (smcont < steps[i+1]) & (smcont > steps[i+2])
            inner_avg = smcont[inner].mean()
            ann_avg = smcont[ann].mean()
            frac.append(ann_avg/inner_avg)
            a = np.where(ann, 1, 0)
            xs, ys = np.where(a == 1)
            rs = np.sqrt((xs-xc)**2 + (ys-yc)**2)
            r.append(np.median(rs))
    
        bigr = 4*r[ind]
        bigind = (np.abs(np.array(r)-bigr)).argmin()
        smr = Rp*r[ind]
        smind = (np.abs(np.array(r)-smr)).argmin()

        bigmask = smcont > steps[bigind]
        smmask = smcont > steps[smind]
        totmask = smcont >= steps[bigind]
        part1 = smcont > steps[smind]
        bigmask[part1]=False

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
        #fig.colorbar(im3, ax=ax3, orientation='horizontal')
        ax3.xaxis.grid()
        ax3.yaxis.grid()
        ax3.set_xlabel(r"log H$\alpha$ SB (erg/s/cm$^2$/arcsec$^2$)")
        ax3.set_ylabel(r"log L$\alpha$ SB (erg/s/cm$^2$/arcsec$^2$)")

        im4 = ax4.hexbin(sp.log10(ha[iShow]), sp.log10(tot_smIntLa[iShow]), cmap='jet', gridsize = 40, vmin=0, vmax = 3000, extent=[-20, -12, -20, -11])
        #fig.colorbar(im4, ax=ax4, orientation='horizontal')
        ax4.xaxis.grid()
        ax4.yaxis.grid()
        ax4.set_xlabel(r"log H$\alpha$ SB (erg/s/cm$^2$/arcsec$^2$)")
        ax4.set_ylabel(r"log L$\alpha$ SB (erg/s/cm$^2$/arcsec$^2$)")

        fig.subplots_adjust(bottom = 0.12)
    
        cbaxes = fig.add_axes([0.125, 0.053, 0.78, 0.01]) 
        cb = plt.colorbar(im3, cax = cbaxes, orientation='horizontal')
        cbaxes.set_xlabel('Number')
    
        plt.savefig('SB_plots/SBplot_LARS10')
        plt.close(fig)
        plt.clf()
        im = plt.imshow(tot_smIntLa, origin='lower', vmin = 0, vmax = 5e-14)
        plt.savefig('SB_plots/galimg_LARS10')
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

        sm = gaussian_filter(tot_smIntLa, 10)/fscale
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
        ff.write('Lya R20: '+str(R20)+'\n')
        
        sm = gaussian_filter(ha*totmask, 20)/fscale
        steps = 10**np.linspace(0, np.log10(sm.max()), 50)[::-1]
        g, xc, yc = np.where([sm] == np.max(sm))
    
        frac, r = [], []
        for i in xrange(48):
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

        sm = gaussian_filter(tot_smIntLa/fscale, 1)
        #sm = tot_smIntLa/fscale
        steps = 10**np.linspace(0, np.log10(sm.max()), 35)[::-1]
        g, xc, yc = np.where([sm] == np.max(sm))
    
        frac, r = [], []
        for i in xrange(33):
            inner = sm > steps[i+1]
            ann = (sm < steps[i+1]) & (sm > steps[i+2])
            inner_avg = sm[inner].mean()
            ann_avg = sm[ann].mean()
            frac.append(ann_avg/inner_avg)
            a = np.where(ann, 1, 0)
            xs, ys = np.where(a == 1)
            rs = np.sqrt((xs-xc)**2 + (ys-yc)**2)
            r.append(np.median(rs))

        plt.plot(r, steps[:-2], 'b-')
        plt.xlabel('Radius (pix)')
        plt.ylabel('Lya SB')
        plt.text(120, 100000, 'Kernel (bright,dim) = '+str(hiss)+', '+str(paw))
        plt.xlim(0, 200)
        #plt.ylim(0, 160000)
        
        sm = gaussian_filter(la*totmask/fscale, 1)
        #sm = la*totmask/fscale
        #print np.max(sm)
        steps = 10**np.linspace(0, np.log10(sm.max()), 35)[::-1]
        g, xc, yc = np.where([sm] == np.max(sm))
    
        frac, r = [], []
        for i in xrange(33):
            inner = sm > steps[i+1]
            ann = (sm < steps[i+1]) & (sm > steps[i+2])
            inner_avg = sm[inner].mean()
            ann_avg = sm[ann].mean()
            frac.append(ann_avg/inner_avg)
            a = np.where(ann, 1, 0)
            xs, ys = np.where(a == 1)
            rs = np.sqrt((xs-xc)**2 + (ys-yc)**2)
            r.append(np.median(rs))

        plt.plot(r, steps[:-2], 'r-')
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        plt.savefig('SBvR_'+str(1+ttt))
        plt.close()
        plt.clf()
    
    ff.close()
