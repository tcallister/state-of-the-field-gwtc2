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

    v_plane = 10.**(np.random.normal(loc=mu_plane,scale=0.3))
    angle = 2.*np.pi*np.random.random()   
    vk_x = v_plane*np.cos(angle)
    vk_y = v_plane*np.sin(angle)
    
    # Perpendicular kick
    vk_z = 10.**(np.random.normal(loc=mu_perp,scale=0.3))

    return vk_x,vk_y,vk_z

def genTruncatedNorm(n,mu,sig,low,high):

    if n==0:
        return

    elif sig==0:
        return np.ones(n)*mu

    else:
        ts = np.random.normal(loc=mu,scale=sig,size=n)
        replace = ((ts<low) + (ts>high))
        ts[replace] = genTruncatedNorm(ts[replace].size,mu,sig,low,high)
        return ts

def getPopRecursion(n,logR_mean,a2_mean,a2_std,dtilt,kick,kick_args,beta,efficiencyThreshold=1e-3,dist='logUniform'):

    genTime = 0.

    mMax = 75.
    mMin = 5.
    bq = 1.3
    alpha = -2.2

    trials = 0
    surviveSN = 0
    merge = 0

    final_binaries = np.array([])
    while final_binaries.size<n:

        trials += n

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
        m2 = np.power((m1**(1.+bq)-mMin**(1.+bq))*np.random.random(n)+mMin**(1.+bq),1./(1.+bq))/beta

        # Separation
        if dist=='logUniform':
            sep = 10.**(np.log10(5.) + (np.log10(300)-np.log10(5.))*np.random.random(n))
        elif dist=='logNormal':
            sep = 10.**np.random.normal(loc=1.1,scale=0.3,size=n)
        elif dist=='delta':
            sep = 10.**logR_mean*np.ones(n)
        else:
            sys.exit()

        binaries = [binary(m1[i],m2[i],a1[i],a2[i],t1[i],t2[i],phi1[i],phi2[i],sep[i]*Rsun) for i in range(n)]

        if kick=="maxwellian":
            survived = np.array([b.kick(random_kick(*kick_args),beta) for b in binaries])
        elif kick=="directed":
            survived = np.array([b.kick(random_lognormal_kick(*kick_args),beta) for b in binaries])
        else:
            sys.exit()

        mergerTimes = np.array([b.time_to_merger() for b in binaries])/1e10/year
        surviveSN += np.where(survived==1)[0].size
        merge += np.where(mergerTimes<1)[0].size

        successful = ((survived==1)*(mergerTimes<1))
        final_binaries = np.append(final_binaries,np.array(binaries)[successful])

        # Estimate success probability
        # If none have yet been successful, take an upper limit of 1/trials
        p_hat = max(1.*final_binaries.size/trials,1./trials)
        error = np.sqrt(p_hat*(1.-p_hat)/trials)
        if (trials>1./efficiencyThreshold) and (efficiencyThreshold-p_hat)/error > 3:
            return final_binaries,trials,surviveSN,merge,False

    return final_binaries,trials,surviveSN,merge,True

if __name__=="__main__":
    
    """
    times = np.array([])
    for i in range(10):
        t_start = time.time()
        try:
            a2_mean = 0.5
            a2_std = 0.5
            final_binaries,trials,surviveSN,merge,efficient = getPopRecursion(1000,3.0,a2_mean,a2_std,0.,"maxwellian",[100,100],0.9)
            if efficient==False:
                print("not efficient")
        except RuntimeError:
            print(":(")
        t_stop = time.time()
        times = np.append(times,t_stop-t_start)
    print(times)
    print(np.mean(times),np.std(times))
    """

    """
    ecc = np.array([bb.eccentricity for bb in b])
    a = np.array([bb.semi_a for bb in b])/Rsun
    t = np.array([bb.time_to_merger() for bb in b])/1e9/year
    print(np.mean(ecc))
    print(np.mean(a))
    print(np.mean(t))
    """
