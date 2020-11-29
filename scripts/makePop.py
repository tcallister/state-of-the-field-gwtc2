import numpy as np
import sys
from binary import *
import time
from scipy.stats import gaussian_kde

part_a = np.array([])

def random_kick(sig_plane,sig_perp):

    if sig_plane==0:
        vk_x = 0
        vk_y = 0
    else:  
        vk_x = np.random.normal(loc=0.,scale=sig_plane)
        vk_y = np.random.normal(loc=0.,scale=sig_plane)

    if sig_perp==0:
        vk_z = 0
    else:
        vk_z = np.random.normal(loc=0.,scale=sig_perp)

    return vk_x,vk_y,vk_z

def special_kick(sig_plane,sig_perp):

    vk_x = 0
    vk_y = -np.random.normal(loc=sig_plane,scale=sig_plane/10.)
    vk_z = np.random.normal(loc=sig_perp,scale=sig_perp/10.)

    return vk_x,vk_y,vk_z

def random_special_kick(mu_plane,sig_plane,mu_perp,sig_perp):

    # Randomly directed in-plane kick
    v_plane = np.random.normal(loc=mu_plane,scale=sig_plane)
    angle = 2.*np.pi*np.random.random()   
    vk_x = v_plane*np.cos(angle)
    vk_y = v_plane*np.sin(angle)

    # Perpendicular kick
    vk_z = np.random.normal(loc=mu_perp,scale=sig_perp)

    return vk_x,vk_y,vk_z

def random_lognormal_kick(mu_plane,mu_perp):

    v_plane = 10.**(np.random.normal(loc=mu_plane,scale=0.1))
    angle = 2.*np.pi*np.random.random()   
    vk_x = v_plane*np.cos(angle)
    vk_y = v_plane*np.sin(angle)
    
    # Perpendicular kick
    vk_z = 10.**(np.random.normal(loc=mu_perp,scale=0.1))

    return vk_x,vk_y,vk_z

def polar_maxwellian(sig_plane,mu_perp,sig_perp):

    if sig_plane==0:
        vk_x = 0
        vk_y = 0
    else:  
        vk_x = np.random.normal(loc=0.,scale=sig_plane)
        vk_y = np.random.normal(loc=0.,scale=sig_plane)

    vk_z = np.random.normal(loc=mu_perp,scale=sig_perp)
    return vk_x,vk_y,vk_z

def genTruncatedNorm(n,mu,sig,low,high):

    if n==0:
        return

    else:
        ts = np.random.normal(loc=mu,scale=sig,size=n)
        replace = ((ts<low) + (ts>high))
        ts[replace] = genTruncatedNorm(ts[replace].size,mu,sig,low,high)
        return ts

#def getPopRecursion(n,R_min,R_max,a2_mean,a2_std,dtilt,kick,kick_args):
#trials = 0
def getPopRecursion(n,logR_mean,a2_mean,a2_std,dtilt,kick,kick_args):

    """
    global trials
    if trials>0:
        r = (200.-n)/trials
        if r<0.001 and 1/np.sqrt(200.-n)<0.1:
            print("too few!")
            trials = 0
            return
    trials += n
    """

    #logR_min = np.log10(R_min)
    #logR_max = np.log10(R_max)

    mMax = 75.
    mMin = 5.
    bq = 1.3
    alpha = -2.2

    if n==0:
        trials = 0
        return

    else:

        # Tilt angles
        t1 = genTruncatedNorm(n,0,dtilt,-np.pi,np.pi)
        t2 = genTruncatedNorm(n,0,dtilt,-np.pi,np.pi)

        # Azimuthal angles
        phi1 = 2.*np.pi*np.random.random(n)
        phi2 = 2.*np.pi*np.random.random(n)

        # Spin magnitudes
        a1 = np.zeros(n)
        a2 = genTruncatedNorm(n,a2_mean,a2_std,0,1)

        # Masses
        m1 = np.power((mMax**(1.+alpha)-mMin**(1.+alpha))*np.random.random(n) + mMin**(1.+alpha),1./(1.+alpha))
        m2 = np.power((m1**(1.+bq)-mMin**(1.+bq))*np.random.random(n)+mMin**(1.+bq),1./(1.+bq))

        # Separation
        #sep = 10.**(logR_min + (logR_max-logR_min)*np.random.random(n))
        #sep = 10.**(np.random.normal(size=n,loc=logR_mean,scale=0.5))
        #sep = 10.**logR_mean*np.ones(n)
        #logSep = genTruncatedNorm(n,logR_mean,0.5,np.log10(3.),np.inf)
        #sep = 10.**logSep
        sep = 10.**(np.log10(5.) + (np.log10(300)-np.log10(5.))*np.random.random(n))

        #period = 10.**(3.*np.random.random(n)-1)*24.*3600
        #sep = np.power(G*(m1+m2)*Msun*period**2./(4.*np.pi**2.),1./3.)/Rsun

        binaries = np.array([binary(m1[i],m2[i],a1[i],a2[i],t1[i],t2[i],phi1[i],phi2[i],sep[i]*Rsun) for i in range(n)])

        if kick=="maxwellian":
            survived = np.array([b.kick(random_kick(*kick_args),1) for b in binaries])
        elif kick=="directed":
            survived = np.array([b.kick(random_lognormal_kick(*kick_args),1) for b in binaries])
        elif kick=="polar_maxwellian":
            survived = np.array([b.kick(polar_maxwellian(*kick_args),1) for b in binaries])
        else:
            sys.exit()

        mergerTimes = np.array([b.time_to_merger() for b in binaries])/1e10/year
        to_replace = ((survived==0) + (mergerTimes>1))
        binaries[to_replace] = getPopRecursion(len(binaries[to_replace]),logR_mean,a2_mean,a2_std,dtilt,kick,kick_args)
        return binaries

if __name__=="__main__":

    times = np.array([])
    for i in range(10):
        t_start = time.time()
        try:
            getPopRecursion(200,3.,0.5,0.5,0.1,"maxwellian",[30,100])
        except RuntimeError:
            print(":(")
        t_stop = time.time()
        times = np.append(times,t_stop-t_start)
    print(times)
    print(np.mean(times),np.std(times))

    #t_start = time.time()
    #getPop(500,2e3,2e3, 10.,30.,6.37060013e-01, 1.99728659e-01,0.1)
    #t_stop = time.time()
    #print(t_stop-t_start)

    """
    times = np.array([])
    for i in range(1000):
        b = binary(10,10,0.1,0.1,0.1,0.1,0.1,0.1,1*Rsun)
        k = random_special_kick(500.,500.)

        start = time.time()
        b.kick(k,1)
        stop = time.time()
        times = np.append(times,stop-start)
    """
