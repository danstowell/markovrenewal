#!/bin/env python

# markovrenewal.linloggmm_aba - GMM representation in 1D linear delta-state space and 1D log delta-time - ABA sequence-related stuff
# Written by Dan Stowell May 2012.
# (c) 2012 Dan Stowell and Queen Mary University of London.
# All rights reserved.

from math import sqrt, log, exp, pi
from operator import itemgetter

from linloggmm import *
import evaluators

myfloat = float

"""
from linloggmm_aba import *
a = aba_segregated()
a.plot()
aba_coherent().plot()

print a
a.likelihood(0, 0.5)
a.likelihood(0, 0.75)
a.likelihood(10, 0.75)
"""

# How to plot densities for the standard models:
"""
python -c "from linloggmm_aba import *; aba_coherent().plot('output/pdf/plot_coherent')"
python -c "from linloggmm_aba import *; aba_segregated().plot('output/pdf/plot_segregated')"
"""

#######################################################################################
# Factory
def aba_coherent(tatum=0.25, statedelta=5, statedev=0.1, timedev=0.2):
	"Model expecting a unified ABAB stream"
	inst = LinLogGMMDelta()
	inst.comps.append(inst.Component([ statedelta, log(tatum)], [statedev, timedev], 1./2))
	inst.comps.append(inst.Component([-statedelta, log(tatum)], [statedev, timedev], 1./2))
	return inst

def aba_segregated(tatum=0.25, statedelta=5, statedev=0.1, timedev=0.2):
	"Model expecting separate A_A_ and _B_B streams"
	inst = LinLogGMMDelta()
	inst.comps.append(inst.Component([0, log(tatum * 2)], [statedev, timedev], 1.0))
	return inst

"""
from markovrenewal import *
from numpy import random
from linloggmm import *
from linloggmm_aba import *

rnddata = [{'timepos': round(random.rand() * 20, 2), 'statepos': round(random.rand() * 10, 2)} for _ in xrange(125)]
g = linlogMRPGraph_delta(rnddata, aba_coherent(statedev=1, timedev=1), 5, 24, maxtimedelta=5)
mcf = g.getMinCostFlow()  # vizpath='tmpgraph_mcf')
c = g.getClustersFromMinCostFlow(mcf, 'tmp_plotclusters')

# Compactly:
mcf = linlogMRPGraph_delta([{'timepos': round(random.rand() * 20, 2), 'statepos': round(random.rand() * 10, 2)} for _ in xrange(15)], aba_coherent(statedev=1, timedev=1), 10, 1, maxtimedelta=7).getMinCostFlow(vizpath='tmpgraph_mcf')
"""



"""
# Pitched data
from markovrenewal import *
from linloggmm import *
from linloggmm_aba import *
from abagen import *
a = eachpitchedabagen(dur=5)
g = linlogMRPGraph_delta(a, aba_coherent(statedev=0.1, timedev=0.1), 10, 1, maxtimedelta=5)
c = g.getClustersFromMinCostFlow(g.getMinCostFlow(), 'tmp_plot_pitchclusters_c')
g = linlogMRPGraph_delta(a, aba_segregated(statedev=0.1, timedev=0.1), 10, 1, maxtimedelta=5)
c = g.getClustersFromMinCostFlow(g.getMinCostFlow(), 'tmp_plot_pitchclusters_s')

g = linlogMRPGraph_delta(a, aba_coherent(statedev=0.0001, timedev=0.0001), 10, 0.0000000001, maxtimedelta=0.1) # enforce pure-clutter view
c = g.getClustersFromMinCostFlow(g.getMinCostFlow(), 'tmp_plot_pitchclusters_o')

c = g.getClustersFromMinCostFlow(g.getMinCostFlow(vizpath='tmp_plot_pitchgraph_c'), 'tmp_plot_pitchclusters_c')
c = g.getClustersFromMinCostFlow(g.getMinCostFlow(vizpath='tmp_plot_pitchgraph_s'), 'tmp_plot_pitchclusters_s')

aba_coherent(statedev=0.1, timedev=1).plot()
"""


"""
from linloggmm_aba import *
plot_abagen_x_linlog()
plot_abagen_x_linlog(snr=0.25, nitems=4)

python -c "from linloggmm_aba import *; plot_abagen_x_linlog()"
"""
def plot_abagen_x_linlog(plotpath='output/pdf/plot_abagen_x_linlog', nitems=4, snr=0.25, stdev=0.2, dur=10, mincentre=10, maxcentre=60, delta=5, tatum=0.5):
	"Plots a 3x3 grid of scatter plots, crossing 3 different ABA generators with 2 different inference models."
	import matplotlib.pyplot as plt
	from abagen import pitchedabagen_multiplusclutter
	# create one of each abagen synthesis, adding multiple gens and also clutter
	abatypes = ['locked', 'coherent', 'segregated']
	results = {}
	for type in abatypes:
		results[type] = pitchedabagen_multiplusclutter(nitems, snr, type, stdev, dur, mincentre, maxcentre, delta, tatum)

	# create one of each inference type
	infertypes = { \
		'coherent':   aba_coherent(  tatum=tatum, statedev=0.1, timedev=stdev), \
		'segregated': aba_segregated(tatum=tatum, statedev=0.1, timedev=stdev) }

	# run each inference on each aba, plot results
	fig = plt.figure()
	for abaindex, abaname in enumerate(abatypes):
		for inferindex, (infername, inferrer) in enumerate(infertypes.items()):
			plotnum = abaindex * 4 + inferindex + 3
			plt.subplot(3, 4, plotnum)
			g = linlogMRPGraph_delta(results[abaname], inferrer, 3 * dur, snr, maxtimedelta=5)
			clusters = g.getClustersFromMinCostFlow(g.getMinCostFlow(), plotunderway=True, plotfontsize="xx-small", plotclutter=False)

			# Evaluate clustering success:
			clusteredindices = [[hit['clusterid'] for hit in cl[1]] for cl in clusters['clusters']]
			clutter = [hit['clusterid'] for hit in clusters['other']]
			clusteredindicesall = clusteredindices + [clutter]

			if abaindex == 2:
				plt.xlabel("inferred (%s)" % infername, fontsize="small") # labels the column
			plt.ylim(6, 64)
		# this one plots clean unclustered data
		plotnum = abaindex * 4 + 1
		plt.subplot(3, 4, plotnum)
		signalonly = filter(lambda datum: datum['type'] != 'clutter', results[abaname])
		x = [datum['timepos' ] for datum in signalonly]
		y = [datum['statepos'] for datum in signalonly]
		plt.plot(x, y, 'k,')
		plt.xticks(fontsize="xx-small")
		plt.yticks(fontsize="xx-small")
		plt.ylim(6, 64)
		plt.ylabel("generator: %s" % abaname, fontsize="small") # labels the row
		if abaindex == 2:
			plt.xlabel("clean signal", fontsize="small") # labels the column
		# this one plots noisy unclustered data
		plotnum = abaindex * 4 + 2
		plt.subplot(3, 4, plotnum)
		g.getClustersFromMinCostFlow([], plotunderway=True, plotfontsize="xx-small")
		plt.ylim(6, 64)
		if abaindex == 2:
			plt.xlabel("signal in noise", fontsize="small") # labels the column
	plt.savefig("%s.pdf" % plotpath, papertype='A4', format='pdf')

"""
from linloggmm_aba import *
plot_abalinlog_online(nitems=2, dur=5, snr=0.25)
"""
def plot_abalinlog_online(plotpath='output/plot_abalinlog_online', nitems=2, snr=0.5, stdev=0.2, dur=10, mincentre=10, maxcentre=60, delta=5, tatum=0.5):
	"Plots an animation with multiple coherent ABA generators and a single model, online-updating one datapoint at a time."
	import matplotlib.pyplot as plt
	from abagen import pitchedabagen_multiplusclutter
	import os
	from subprocess import call

	# We will make ourselves a folder with lots of PNGs in it
	if not os.path.exists (plotpath):
		os.makedirs(plotpath)

	results = pitchedabagen_multiplusclutter(nitems, snr, 'coherent', stdev, dur, mincentre, maxcentre, delta, tatum)
	model = aba_coherent(tatum=tatum, statedev=0.1, timedev=stdev)

	# Create initial graph with the first few data points
	initnum = len(results) / 3
	print "Starting with %i datapoints" % initnum
	g = linlogMRPGraph_delta(results[:initnum], model, 3 * dur, snr, maxtimedelta=5)
	mcf = g.getMinCostFlow()

	# Now iterate the rest of the data points, adding one by one, and plot what happens
	plotcode=0
	for index in range(initnum, len(results)):
		print "==================Adding a datum===================="
		datum = results[index]
		# TODO: currently just recreating - get the update func working!
		#g = linlogMRPGraph_delta(results[:index], model, 3 * dur, snr, maxtimedelta=5)
		#mcf = g.getMinCostFlow()
		mcf = g.updateMinCostFlow(prevflow=mcf, newdata=[datum])
		fig = plt.figure()
		g.getClustersFromMinCostFlow(mcf, plotunderway=True, plotfontsize="xx-small")
		plt.ylim(6, 64)
		plt.xlim(0, results[-1]['timepos'])
		plt.savefig("%s/frame%0*i.png" % (plotpath, 3, plotcode), format='png')
		plotcode += 1
	# stitch together pics into an animation
	call(["ffmpeg", "-r", "10", "-i", "%s/frame%%03d.png" % plotpath, "-y", "%s/anim_abalinlog_online.mp4" % plotpath])

"""
python -c "from linloggmm_aba import *; multitest_abagen_linlog()"
"""
def multitest_abagen_linlog(nruns=50, stdev=0.2, dur=10, mincentre=10, maxcentre=60, delta=5, tatum=0.5):
	# The combination of heavy SNR and many trails makes a VERY long runtime, don't activate until later.
	snrs = {24: 16, 12: 4, 0: 1, -12:0.25, -24: 0.0625}
	nitemses = [4]   # was [1, 2, 4]
	infertypes = { \
		'coherent':   aba_coherent(  tatum=tatum, statedev=0.1, timedev=stdev), \
		'segregated': aba_segregated(tatum=tatum, statedev=0.1, timedev=stdev) }
	runtypes = [ \
		{'label': 'coh', 'abatype': 'coherent',   'infertype': 'coherent'}, \
		{'label': 'seg', 'abatype': 'segregated', 'infertype': 'segregated'} \
		]
	analysistypes = ['snrk', 'snru', 'snrug']  # SNR known/unknown, SNR unknown with greedy inference
	evalstograb = ["Fsn", "Ftrans", "Fsigtrans"]

	from abagen import pitchedabagen_multiplusclutter

	f = open('output/multitest_abagen_linlog.txt', 'w')
	f.write("runtype,known,snr,nmix,whichstat")
	for i in xrange(nruns):
		f.write(",val%i" % (i))
	f.write("\n")
	for runtype in runtypes:
		for db, snr in sorted(snrs.items(), reverse=True):
			for nitems in nitemses:
				evals = [[] for _ in analysistypes]
				print "=================================================="
				print "About to do runs of (%s, %i, %i)" % (runtype['label'], db, nitems)
				for whichrun in range(nruns):
					# Generate a dataset
					data = pitchedabagen_multiplusclutter(nitems, snr, runtype['abatype'], stdev, dur, mincentre, maxcentre, delta, tatum, 1./nitems)
					
					for anaindex, anatype in enumerate(analysistypes):
						if anatype=='snrk':
							toldsnr = snr
						else:
							toldsnr = 1.0
						# Run the inference
						g = linlogMRPGraph_delta(data, infertypes[runtype['infertype']], 3 * dur, toldsnr, maxtimedelta=3)
						greedynotfull = (anatype == 'snrug')
						#if greedynotfull:
						#	print "GREEDY"
						c = g.getClustersFromMinCostFlow(g.getMinCostFlow(greedynotfull=greedynotfull))

						# Evaluate clustering success:
						clusteredindices = [[hit['clusterid'] for hit in cl[1]] for cl in c['clusters']]
						clutter = [hit['clusterid'] for hit in c['other']]
						clusteredindicesall = clusteredindices + [clutter]
						#meh ari[anaindex].append(evaluators.adjusted_rand_index(clusteredindicesall))
						cur_evals = evaluators.cluster_eval_stats(clusteredindices, clutter, string=False)
						cur_evals['Ftrans']    = evaluators.fmeasure_transitions_fromaba(data, c)
						cur_evals['Fsigtrans'] = evaluators.fmeasure_transitions_fromaba(data, c, sigtransonly=True)
						evals[anaindex].append(cur_evals)

				# Now calc the stats aggregated over runs, and write to file 
				for anaindex in xrange(len(analysistypes)):
					for eg in evalstograb:
						f.write("%s,%s,%g,%i,%s" % (runtype['label'], analysistypes[anaindex], db, nitems, eg))  # start a new row
						vals = map(itemgetter(eg), evals[anaindex])
						for val in vals:
							f.write(',%g' % (val))
						f.write("\n")
				f.flush()
	f.close()

#######################################################
if __name__ == "__main__":
	plot_abalinlog_online(nitems=4, dur=5, snr=0.25, tatum=0.25)

