import numpy as np
import acor

import matplotlib.pyplot as plt
from matplotlib import rc
import sys
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
rc('xtick', labelsize=16) 
rc('ytick', labelsize=16)

if __name__=="__main__":

    # Load sample chain
    sample_chain = np.load(sys.argv[1])

    #sample_chain = np.delete(sample_chain,[6],0)

    goodInds = np.where(sample_chain[0,:,0]!=0.0)[0]
    sample_chain = sample_chain[:,goodInds,:]
    
    print("Sample chain:")
    print sample_chain
    
    print("Shape of sample chain:")
    print np.shape(sample_chain)
    
    # Dimension of parameter space, number of steps taken, and number of walkers used
    nWalkers,nSteps,dim = sample_chain.shape
    
    # Burn first half of chain
    chainBurned = sample_chain[:,int(np.floor(nSteps/3.)):,:]
    
    print("Shape of burned chain:")
    print np.shape(chainBurned)

    # Get mean correlation length (averaging over all variables and walkers)
    corrTotal = np.zeros(dim)
    for i in range(dim):
        for j in range(nWalkers):
            (tau,mean,sigma) = acor.acor(chainBurned[j,:,i]) #acor = autocorrelation package
            corrTotal[i] += tau/(nWalkers)
    meanCorLength = np.max(corrTotal)
    
    print("Mean correlation length:")
    print(meanCorLength)

    # Down-sample by twice the mean correlation length
    chainDownsampled = chainBurned[:,::2*int(meanCorLength),:]

    print("Shape of downsampled chain:")
    print np.shape(chainDownsampled)

    # Flatten - we don't care about each individual walker anymore
    chainDownsampled = chainDownsampled.reshape((-1,len(chainDownsampled[0,0,:])))
    
    print("Shape of downsampled chain post-flattening:")
    print np.shape(chainDownsampled)
    
    # Save data 
    np.save('processed_{0}'.format(sys.argv[1]), chainDownsampled)

