#!/bin/env python

# markovrenewal.linloggmm - GMM representation in 1D linear delta-state space and 1D log delta-time
# Written by Dan Stowell May 2012.
# (c) 2012 Dan Stowell and Queen Mary University of London.
# All rights reserved.

from math import sqrt, log, exp, pi

from markovrenewal import MRPGraph

myfloat = float


SQRTTWOPI = myfloat(sqrt(2. * pi))
ONEOVER_SQRTTWOPI = myfloat(1. / SQRTTWOPI)

#import scipy.stats
#def evaluatenormal_NOT_ACTUALLY_FASTER(mean, SHOULDBEINVstdev, location):
#	"Returns value of univariate normal at a given location"
#	return scipy.stats.norm(loc=mean, scale=stdev).pdf(location)
def evaluatenormal(mean, inv_stdev, location):
	"Returns value of univariate standard normal at a given location"
	deviance = (location - mean) * inv_stdev
	return (ONEOVER_SQRTTWOPI * inv_stdev) * exp(-0.5 * deviance * deviance)

##############################################################################
# factory for doing things with MRP

def linlogMRPGraph_delta(origdata, linloggmm_delta, expectedpathlen, estsnr, maxtimedelta=5):
	"""Factory method for using MRPGraph with a LinLogGMMDelta and fixed probys derived from estimated lengths and SNRs.
	'expectedpathlen' is the expected number of emissions in a MRP sequence - if deathprob is fixed then it's exponential decay over sequence index.
	'estsnr' is an estimate of signal-to-noise ratio (NOT in dB, as a ratio) - e.g. "2" means you estimate two-thirds of the datapoints to be non-clutter.
	"""
	deathprob = (1./expectedpathlen)      # exponential decay relationship
	birthprob =   estsnr / ((1. + estsnr) * expectedpathlen)
	clutterprob = 1.     / (1.+estsnr)
	print "Probabilities derived: birth %g, death %g, clutter %g." % (birthprob, deathprob, clutterprob)

	transprobcallback = lambda a,b: linloggmm_delta.likelihood(b['statepos']-a['statepos'], b['timepos']-a['timepos'])
	birthprobcallback   = lambda a: birthprob
	deathprobcallback   = lambda a: deathprob
	clutterprobcallback = lambda a: clutterprob
	return MRPGraph(origdata, transprobcallback, birthprobcallback, deathprobcallback, clutterprobcallback, maxtimedelta)

def justGetLikelihoodRatio(data, linloggmm_delta):
	"Used by streammodels:streamingBayesFactor; runs an MRP analysis and only returns the likelihood ratio"
	# note the small "maxtimedelta" here - for tractable runtime - cf the timehop values, it must be at least double the biggest timehop!
	g = linlogMRPGraph_delta(data, linloggmm_delta, 20, 1, maxtimedelta=0.5)
	c = g.getClustersFromMinCostFlow(g.getMinCostFlow(numcutoff=50))   # NOTE numcutoff used here to truncate overlong searches; technically this reduces accuracy, and probably reduces the magnitude of likelihood ratios slightly, but the top 50 should hold the bulk of the probability.
	l = sum(map(__justGetLikelihoodRatio_getnegcost, c['clusters']))
	l = max(min(l, 700.), -700.)
	#print "log(l): %g" % l
	l = exp(l)
	return l
def __justGetLikelihoodRatio_getnegcost(clust):
	"aux func intended to speed up justGetLikelihoodRatio()"
	return -clust[0]

#######################################################################################
class LinLogGMMDelta:
	"""A GMM for a generative model with 1D state space as gaussian mixture, and timedelta a log(gaussian mixture), so each gaussian is 2D (but with no off-axis covariance) --- AND state-transition probabilities are independent of base value and purely based on deltas."""

	def __init__(self):
		self.comps = []

	def __str__(self):
		ret = "LinLogGMMDelta["
		for comp in self.comps:
			ret += comp.__str__() + ", "
		ret += "]"
		return ret

	def likelihood(self, statedelta, timedelta):
		"""Returns the likelihood of the state & time transition from wherever we are now."""
		return sum(map(lambda comp: comp.valueatloc([statedelta, log(timedelta)]), self.comps)) \
		     / sum(map(lambda comp: comp.weight, self.comps))  # normalise, just in case

	def plot(self, fpath=None, minstate=-7, maxstate=7, mintime=0.00001, maxtime=1.0, resolution=50, fontsize="xx-large"):
		import matplotlib.pyplot as plt
		import matplotlib.cm as cm
		si = [float(si)/(resolution)*(maxstate-minstate) + minstate for si in range(resolution+1)]
		ti = [float(ti)/(resolution)*(maxtime -mintime ) + mintime  for ti in range(resolution+1)]
		samples = [[self.likelihood(sii, tii) for tii in ti] for sii in si]
		plt.figure()
		plt.imshow(samples, aspect='auto', cmap=cm.binary)
		plt.xlabel('Time delta (s)', fontsize=fontsize)
		plt.ylabel('State delta'   , fontsize=fontsize)
		plt.xticks([0,len(ti)-1], [ti[0], ti[-1]], fontsize=fontsize)
		plt.yticks([0,(len(si)-1)/2,len(si)-1], [si[0], si[(len(si)-1)/2], si[-1]], fontsize=fontsize)
		if fpath == None:
			plt.show()
		else:
			plt.savefig("%s.pdf" % fpath, papertype='A4', format='pdf')
		return plt

	class Component:
		"""Represents a single 2D independent Gaussian component, 
		with a float [mean_S, mean_T], [stdev_S, stdev_T], weight.
		Note that it's a gaussian not a log-gaussian. Do any transformation outside."""
		def __init__(self, mean, stdev, weight):
			self.mean       = [myfloat(      mean[0]), myfloat(      mean[1])]
			self.inv_stdev  = [myfloat(1. / stdev[0]), myfloat(1. / stdev[1])]
			self.weight = myfloat(weight)

		def valueatloc(self, loc):
			# since components have no diagonals, we treat time and state as separate univariate normals and multiply
			return self.weight \
			      * evaluatenormal(self.mean[0], self.inv_stdev[0], loc[0]) \
			      * evaluatenormal(self.mean[1], self.inv_stdev[1], loc[1])

		def __str__(self):
			return "S: %.3g +- %.3g; T: %.3g */ %.3g; W: %.3g" % (self.mean[0], 1. / self.inv_stdev[0], exp(self.mean[1]), exp(1. / self.inv_stdev[1]), self.weight)

