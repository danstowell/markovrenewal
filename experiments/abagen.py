#!/bin/env python

# stream_mrp - Markov Renewal Process model of auditory streaming
# Written by Dan Stowell May 2012.
# (c) 2012 Dan Stowell and Queen Mary University of London.
# All rights reserved.

"""
from abagen import *
import matplotlib.pyplot as plt
plotabagen()

a = eachpitchedabagen()
x = [item['timepos'] for item in a]
y = [item['statepos'] for item in a]
plt.plot(x, y, 'bo')
plt.ylim(0, 60)
plt.show()

"""

from numpy import *
from numpy import random
from operator import itemgetter

import matplotlib.pyplot as plt

def generateABA(type, stdev, dur=30, tatum=0.25):
	"""Generates ABABABAB sequences. 'type' selects from:
	  'locked' - strictly phase-locked, as in standard streaming stimulus
	          (which is ambiguous about whether segregated or not).
	  'coherent' - phase-locked, and with random rate drift that affects
                  to 0s and 1s. This should imply a coherent stream.
	  'segregated' - the 0s and 1s have independent rate drifts, which
                  should imply two segregated streams.
          'clutter' - random data with the same marginals as the other types,
                  poisson-distributed and with 0 or 1 randomly sampled.
	"""
	
	history = []
	if type == 'locked' or type == 'coherent':
		# only one generator needed - simply iterate two-way phase until we've got enough
		# Phase is integer 0,1
		phase = 0
		timepos = random.rand() * 0.0001
		while timepos < dur:
			if phase == 0:
				value = 0
				jump = tatum
			else:
				value = 1
				jump = tatum
			history.append({'timepos': timepos, 'prob1': value, 'type': type})
			phase = (phase + 1) % 2
			if type == 'coherent':
				# add the wibble to the jump size
				jump *= exp(random.randn() * stdev)
			timepos += jump
	elif type == 'segregated':
		# we need TWO INDEPENDENT generators
		# The [0,0,0,0,0,0] generator
		timepos = random.rand() * 0.0001
		while timepos < dur:
			jump = tatum * 2
			history.append({'timepos': timepos, 'prob1': 0, 'type': type})
			jump *= exp(random.randn() * stdev)
			timepos += jump

		# The [1,1,1,1,1,1] generator
		timepos = random.rand() * 0.0001 + tatum    # offset of 'tatum' is so the pattern starts off as ABA_
		while timepos < dur:
			jump = tatum * 2
			history.append({'timepos': timepos, 'prob1': 1, 'type': type})
			jump *= exp(random.randn() * stdev)
			timepos += jump
	elif type == 'clutter':
		# poisson distributed clutter with random labels, and same density (i.e. 0.75 events per tatum, and 2/3 of them are 0s)
		# http://en.wikipedia.org/wiki/Poisson_process#Homogeneous
		# "If for every t > 0 the number of arrivals in the time interval [0,t] follows the Poisson distribution with mean lambda * t, 
		#  then the sequence of inter-arrival times are independent and identically distributed exponential random variables having mean 1 / lambda"
		timepos = random.rand() * 0.0001
		oneoverlambda = tatum / 0.75
		while timepos < dur:
			jump = random.exponential(oneoverlambda)
			if random.rand() < 1./2.:
				prob1 = 0
			else:
				prob1 = 1
			history.append({'timepos': timepos, 'prob1': prob1, 'type': type})
			timepos += jump
	else:
		raise ValueError("unknown type '%s'" % type)
	
	# Finally, drop any items outside the dur range (may be over) and then sort
	history = filter(lambda x: x['timepos'] < dur, history)
	history.sort(cmp=lambda a,b: cmp(a['timepos'], b['timepos']))
	return history

"""
python -c "from abagen import *; plotabagen()"
"""
def plotabagen(stdev=0.1, dur=10):
	"Plot an example showing each of the generator types"
	
	fig = plt.figure()
	for plotindex, type in enumerate(['locked', 'coherent', 'segregated', 'clutter']):
		a = generateABA(type, stdev, dur)
		x = [item['timepos'] for item in a]
		y = [item['prob1'] for item in a]
		ax = fig.add_subplot(611 + plotindex)
		ax.plot(x, y, 'ro')
		plt.ylim(-0.9, 1.9)
		plt.xlim(0, dur)
		plt.yticks([0, 1], ['A', 'B'])
		plt.xticks([])
		plt.ylabel(type, fontsize='small')
	plt.xticks(range(dur+1))
	plt.xlabel("seconds", fontsize='small')
	plt.savefig("output/pdf/plot_generators.pdf", papertype='A4', format='pdf')
	fig.show()
	return fig

def pitchedabagen(type, stdev=0.1, dur=10, mincentre=10, maxcentre=60, delta=5, tatum=0.25):
	"Generates a singale ABAB signal, with a random centre pitch"
	results = []
	a = generateABA(type, stdev, dur, tatum)
	centrepitch = random.rand() * (maxcentre-mincentre) + mincentre
	for datum in a:
		datum['statepos'] = centrepitch + ((datum['prob1'] - 0.5) * delta)
		results.append(datum)
	return results

def pitchedabagen_multiplusclutter(nitems, snr, type, stdev=0.1, dur=10, mincentre=10, maxcentre=60, delta=5, tatum=0.25, lenfrac=1.0):
	results = []
	clusterid = 0
	for _ in range(nitems):  # signal
		onetrack = pitchedabagen(type, stdev, dur, mincentre, maxcentre, delta, tatum)
		if lenfrac != 1.:
			# "offset" shunts the signal up a bit
			offset = (random.rand() * (1. - lenfrac)) * dur 
			for datum in onetrack:
				datum['timepos'] += offset
		# groundtruth cluster labels used for evaluating. note that seg makes it more complicated because it
		# generates two independent streams, so we need to make sure we label them as such.
		for datum in onetrack:
			if type=='segregated' and datum['prob1'] > 0.5:
				clusterid_aug = 1
			else:
				clusterid_aug = 0
			datum['clusterid'] = clusterid + clusterid_aug
		results.extend(onetrack)
		clusterid += 2
	for _ in range(int(len(results)/snr)):  # clutter
		results.append({'timepos': random.rand() * dur, 'statepos': random.rand() * (delta+maxcentre-mincentre) + mincentre - (delta*0.5), 'type': 'clutter', 'clusterid': -1})
	for i, datum in enumerate(results):
		datum['datumid'] = i
	results.sort(key=itemgetter('timepos'))
	return results

def eachpitchedabagen(stdev=0.1, dur=10, mincentre=10, maxcentre=60, delta=5):
	"Generates a mixture of ABAB signals, each with a centre pitch"
	results = []
	for plotindex, type in enumerate(['locked', 'coherent', 'segregated', 'clutter']):
		results.extend(pitchedabagen(type, stdev, dur, mincentre, maxcentre, delta))
	results.sort(key=itemgetter('timepos'))
	return results

