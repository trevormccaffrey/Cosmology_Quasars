import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Load in filters and composite spectrum
ufil = pd.read_csv("../data/SDSS_Filters/u.txt", sep="\s+").values
gfil = pd.read_csv("../data/SDSS_Filters/g.txt", sep="\s+").values
rfil = pd.read_csv("../data/SDSS_Filters/r.txt", sep="\s+").values
ifil = pd.read_csv("../data/SDSS_Filters/i.txt", sep="\s+").values
zfil = pd.read_csv("../data/SDSS_Filters/z.txt", sep="\s+").values
vb01 = pd.read_csv("../data/vandenberk01_medianSDSSspectrum.txt", sep="\s+").values

def plot_filters(z):
    #fig = plt.figure(figsize=(7,7))
    plt.plot(ufil[:,0]/(1.+z), ufil[:,1]*1.6, color="b", label="u")
    plt.fill_between(ufil[:,0]/(1.+z), ufil[:,1]*1.6, np.zeros(len(ufil[:,1])), color="b", alpha=0.4)
    plt.plot(gfil[:,0]/(1.+z), gfil[:,1]*1.6, color="g", label="g")
    plt.fill_between(gfil[:,0]/(1.+z), gfil[:,1]*1.6, np.zeros(len(gfil[:,1])), color="g", alpha=0.4)
    plt.plot(rfil[:,0]/(1.+z), rfil[:,1]*1.6, color="y", label="r")
    plt.fill_between(rfil[:,0]/(1.+z), rfil[:,1]*1.6, np.zeros(len(rfil[:,1])), color="y", alpha=0.4)
    plt.plot(ifil[:,0]/(1.+z), ifil[:,1]*1.6, color="orange", label="i")
    plt.fill_between(ifil[:,0]/(1.+z), ifil[:,1]*1.6, np.zeros(len(ifil[:,1])), color="orange", alpha=0.4)
    plt.plot(zfil[:,0]/(1.+z), zfil[:,1]*1.6, color="r", label="z")
    plt.fill_between(zfil[:,0]/(1.+z), zfil[:,1]*1.6, np.zeros(len(zfil[:,1])), color="r", alpha=0.4)
    plt.plot([2500,2500],[-0.1,1.0],"-k") #plot 2500 angstroms
    plt.plot(vb01[:,0], vb01[:,1]/10.-0.05, "-k")
    plt.ylabel("Response", fontsize=15)
    plt.xlabel("Wave", fontsize=15)
    plt.title("z = %.2f"%z, fontsize=15)
    plt.xlim(min(ufil[:,0]/(1+z)),max(zfil[:,0]/(1+z)))
    plt.ylim(0,0.97)
    plt.legend(loc="upper right", prop={"size":20}, framealpha=0.5, labelspacing=0.1, borderpad=0.1)
    #plt.show()

fig = plt.figure(figsize=(7,7))
for zz in np.arange(0.5, 3.0+0.1, 0.01):
    plot_filters(zz)
    plt.pause(0.00001)
    plt.cla()
plt.show()
