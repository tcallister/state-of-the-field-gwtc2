import numpy as np
import emcee as mc
import h5py
from scipy.stats import gaussian_kde
from scipy.special import erf
from scipy.special import erfc
import sys
from makePop import *

mMin = 5.
mMax = 75.
alpha = -2.2
bq = 1.3

# -- Set prior bounds --
logR_min = np.log10(5.)
logR_max = 2
v_parallel_min = 0.
v_parallel_max = 6.
v_perp_min = 0.
v_perp_max = 6.
sig_perp_min = 0.
sig_perp_max = 3.
a2_mean_min = 0.
a2_mean_max = 1.
a2_std_min = 0.01
a2_std_max = 1.

# Dicts with samples: 
sampleDict = np.load("/home/thomas.callister/CBC/state-of-the-field-gwtc2/input/sampleDict_sampleRelease.pickle")

# Load mock detections
ref_m_min = 2.
ref_m_max = 100.
ref_a1 = -2.35
ref_a2 = 2.

mockDetections = h5py.File('/home/thomas.callister/CBC/state-of-the-field-gwtc2/input/o3a_bbhpop_inj_info.hdf5','r')
ifar_1 = mockDetections['injections']['ifar_gstlal'].value
ifar_2 = mockDetections['injections']['ifar_pycbc_bbh'].value
ifar_3 = mockDetections['injections']['ifar_pycbc_full'].value
detected = (ifar_1>1) + (ifar_2>1) + (ifar_3>1)
m1_det = mockDetections['injections']['mass1_source'].value[detected]
m2_det = mockDetections['injections']['mass2_source'].value[detected]
s1z_det = mockDetections['injections']['spin1z'].value[detected]
s2z_det = mockDetections['injections']['spin2z'].value[detected]
z_det = mockDetections['injections']['redshift'].value[detected]

mockDetectionsO1O2 = h5py.File('/home/thomas.callister/CBC/state-of-the-field-gwtc2/input/injections_O1O2an_spin.h5','r')
m1_det = np.append(m1_det,mockDetectionsO1O2['mass1_source'])
m2_det = np.append(m2_det,mockDetectionsO1O2['mass2_source'])
s1z_det = np.append(s1z_det,mockDetectionsO1O2['spin1z'])
s2z_det = np.append(s2z_det,mockDetectionsO1O2['spin2z'])
z_det = np.append(z_det,mockDetectionsO1O2['redshift'])

q_det = m2_det/m1_det
mtot_det = m1_det+m2_det
X_det = (m1_det*s1z_det + m2_det*s2z_det)/(m1_det+m2_det)

ref_p_z = np.power(z_det,2.)
ref_p_m1 = np.power(m1_det,ref_a1)
ref_p_m2 = (1.+ref_a2)*np.power(m2_det,ref_a2)/(m1_det**(1.+ref_a2) - ref_m_min**(1.+ref_a2))
ref_p_xeff = np.zeros(X_det.size)
for i in range(ref_p_xeff.size):
    
    X = X_det[i]
    q = q_det[i]
    
    if X<-(1.-q)/(1.+q):
        ref_p_xeff[i] = (1./(2.*q))*(1.+q)*(1.+X)*(1.+q)/2.
        
    elif X>(1.-q)/(1.+q):
        ref_p_xeff[i] = (1./(2.*q))*(1.+q)*(1.-X)*(1.+q)/2.
        
    else:
        ref_p_xeff[i] = (1.+q)/2.

new_p_z = np.power(z_det,2.7)
new_p_m1 = (1.+alpha)*np.power(m1_det,alpha)/(mMax**(1.+alpha)-mMin**(1.+alpha))
new_p_m2 = (1.+bq)*np.power(m2_det,bq)/(m1_det**(1.+bq)-mMin**(1.+bq))

pop_reweight = (new_p_m1*new_p_m2*new_p_z)/(ref_p_xeff*ref_p_m1*ref_p_m2*ref_p_z)
pop_reweight[m1_det<mMin] = 0.
pop_reweight[m2_det<mMin] = 0.

# -- Log posterior function -- 
def logposterior(c):

    # Read parameters
    #logR = c[0]
    logv_parallel = c[0]
    logv_perp = c[1]
    logsig_perp = c[2]
    a2_mean = c[3]
    a2_std = c[4]

    # Flat priors, reject samples past boundaries
    if logv_parallel<v_parallel_min or logv_parallel>v_parallel_max or logv_perp<v_perp_min or logv_perp>v_perp_max or logsig_perp<sig_perp_min or logsig_perp>sig_perp_max or a2_mean<a2_mean_min or a2_mean>a2_mean_max or a2_std<a2_std_min or a2_std>a2_std_max:
        return -np.inf

    # If sample in prior range, evaluate
    else:

        logP = 0.

        # Draw catalog
        try:
            binaries = getPopRecursion(300,1.,a2_mean,a2_std,0.1,"polar_maxwellian",[10.**logv_parallel,10.**logv_perp,10.**logsig_perp])
        except RuntimeError:
            print("Negligible survival...")
            return -np.inf

        chi_effectives = np.array([b.chi_effective() for b in binaries])
        chi_ps = np.array([b.chi_p() for b in binaries])

        chi_effective_kde = gaussian_kde(chi_effectives)
        chi_eff_grid = np.linspace(-1,1,200)
        chi_eff_norm = np.trapz(chi_effective_kde(chi_eff_grid),chi_eff_grid)
        
        #chi_p_kde = gaussian_kde(chi_ps)
        #chi_p_grid = np.linspace(0,1,200)
        #chi_p_norm = np.trapz(chi_p_kde(chi_p_grid),chi_p_grid)

        chi_eff_p_kde = gaussian_kde([np.concatenate([chi_effectives,chi_effectives]),np.concatenate([chi_ps,-chi_ps])])

        nEvents = len(sampleDict)
        p_det_xeff = chi_effective_kde(X_det)/chi_eff_norm
        det_weights = p_det_xeff*pop_reweight
        if np.max(det_weights)==0:
            return -np.inf
        Nsamp = np.sum(det_weights)/np.max(det_weights)
        if Nsamp<=4*nEvents:
            print("Insufficient mock detections:",c)
            return -np.inf
        log_detEff = -nEvents*np.log(np.sum(det_weights))
        logP += log_detEff

        for event in sampleDict:

            # Grab samples
            Xeff_sample = sampleDict[event]['Xeff']
            Xp_sample = sampleDict[event]['Xp']
            #spin_prior = sampleDict[event]['Xeff_priors']
            spin_prior = sampleDict[event]['joint_priors']
            weights = sampleDict[event]['weights']

            # Chi probability
            #p_chi = chi_effective_kde(Xeff_sample)*chi_p_kde(Xp_sample)/chi_eff_norm/chi_p_norm
            #p_chi = chi_effective_kde(Xeff_sample)/chi_eff_norm
            p_chi = chi_eff_p_kde([Xeff_sample,Xp_sample])

            # Evaluate marginalized likelihood
            nSamples = p_chi.size
            pEvidence = np.sum(p_chi*weights/spin_prior)/nSamples
            
            # Summation
            logP += np.log(pEvidence)

        print(c,logP)
        return logP
    
# -- Running mcmc --     
if __name__=="__main__":

    # Initialize walkers from random positions in mu-sigma2 parameter space
    nWalkers = 32

    #initial_logR = np.random.random(nWalkers)*0.3+1.
    #initial_vpars = np.random.random(nWalkers)*(v_parallel_max-v_parallel_min)+v_parallel_min
    #initial_vperps = np.random.random(nWalkers)*(v_perp_max-v_perp_min)+v_perp_min
    initial_logv_parallel = np.random.random(nWalkers)+2.
    initial_logv_perp = np.random.random(nWalkers)+2.
    initial_logsig_perp = np.random.random(nWalkers)+1.
    initial_a2_means = np.random.random(nWalkers)*0.3
    initial_a2_stds = np.random.random(nWalkers)*0.3+0.3
    initial_walkers = np.transpose([initial_logv_parallel,initial_logv_perp,initial_logsig_perp,initial_a2_means,initial_a2_stds])
    
    print('Initial walkers:')
    print(initial_walkers)
    
    # Dimension of parameter space
    dim = 5

    # Run
    nSteps = 2000
    sampler = mc.EnsembleSampler(nWalkers,dim,logposterior,threads=16)
    for i,result in enumerate(sampler.sample(initial_walkers,iterations=nSteps)):
        if i%10==0:
            np.save('emcee_samples_fixedR_polarMaxwellian.npy',sampler.chain)
    np.save('emcee_samples_fixedR_polarMaxwellian.npy',sampler.chain)
