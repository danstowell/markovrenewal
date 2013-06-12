#!/bin/env python

# markovrenewal - Markov Renewal Process mixture inference
# Written by Dan Stowell May 2012.
# (c) 2012 Dan Stowell and Queen Mary University of London.
# All rights reserved.

from markovrenewal import *
import random
from math import log, exp

verbose = False

def makeRandomData(size, maxhop = 5, timespan = 5):
	data = [{\
		'index': i,
		'timepos': round(random.random() * timespan, 2), \
		#'statepos': round(random.random() * 10, 0), \
		'birthprob': round(random.random() * 0.9, 2) + 0.1, \
		'deathprob': round(random.random() * 0.9, 2) + 0.1, \
		'clutterprob': round(random.random() * 0.9, 2) + 0.1, \
		'transprobs': {}
		} for i in xrange(size)]
	# now make a proper transition table
	for i, datum in enumerate(data):
		kids = filter(lambda x: (x['timepos'] > datum['timepos']) and (x['timepos'] <= datum['timepos']+maxhop), data)
		t = [round(random.random(), 2) + 0.1 for _ in kids]
		sumt = sum(t)
		t = map(lambda x: x / sumt, t)
		for kidi, kid in enumerate(kids):
			datum['transprobs'][kid['index']] = t[kidi]
	return data

def floatcmp(a, b):
	"Compares floats with a little wiggleroom for roundoff error"
	return abs(a - b) < 1e-12

#########################################################################
print "Markov Renewal Process graph algorithm: running unit tests."

maxhop = 5
print "================================================================================"

for size in [10, 20, 50]:
	print "== Size %i ==" % size
	data = makeRandomData(size, maxhop)
	if verbose:
		print "Random test data:"
		for datum in data: print datum

	def transprobcallback(x, y):
		try:
			return x['transprobs'][y['index']]
		except IndexError:
			print "Tried to access index %i in item %i's transprob list, which has values for %s" % (x['index'], y['index'], str(x['transprobs'].keys()))
			raise
	birthprobcallback   = lambda x: x['birthprob']
	deathprobcallback   = lambda x: x['deathprob']
	clutterprobcallback = lambda x: x['clutterprob']

	g = MRPGraph(data, transprobcallback, birthprobcallback, deathprobcallback, clutterprobcallback, maxtimedelta=maxhop)

	mcf =  g.getMinCostFlow()

	clres = g.getClustersFromMinCostFlow(mcf)

	print "Check that all data points have been included, once only, in the cluster output:"
	indices = [x['index'] for x in clres['other']]
	for cl in clres['clusters']:
		indices.extend([anorig['index'] for anorig in cl[1]])
	indices.sort()
	if indices == range(size):
		print " ...pass"
	else:
		for icl, cl in enumerate(clres['clusters']):
			print str(icl) + str([anorig['index'] for anorig in cl[1]])
		print [x['index'] for x in clres['other']]
		raise ValueError(indices)

	print "Check that each cluster's time-positions are ordered:"
	for cl in clres['clusters']:
		for i in xrange(len(cl[1])-1):
			if cl[1][i]['timepos'] > cl[1][i+1]['timepos']:
				raise ValueError([anorig['timepos'] for anorig in cl[1]])
	print " ...pass"

	print "Check that each cluster's cost matches the probabilities in our data:"
	passed = True
	for cl in clres['clusters']:
		# The cost should be the neg sum of the logs of the first-birth, last-death, and transitions, minus the clutterprobs
		costb = -log(cl[1][0]['birthprob'])
		costd = -log(cl[1][-1]['deathprob'])
		# remember transition costs are scaled inside MRPGraph using the survival proby, hence the multiplication here:
		costt = [-log(cl[1][index]['transprobs'][cl[1][index+1]['index']] * (1.0 - cl[1][index]['deathprob'])) for index in xrange(len(cl[1])-1)]
		costc = [log(datum['clutterprob']) for datum in cl[1]]

		costrecalc = costb + costd + sum(costt) + sum(costc)

		if not floatcmp(cl[0], costrecalc):
			print ("%g != %g"  % (cl[0], costrecalc))
			passed = False
		elif verbose:
			print ("%g == %g"  % (cl[0], costrecalc))
		if verbose or not passed:
			vpairmap = [g.getVpairForDatum(x) for x in cl[1]]
			print "     indices: " + str([anorig['index'] for anorig in cl[1]])
			print "     birth: we say %g, it says %g" % (exp(-costb), exp(-g.source.outarccosts[vpairmap[0].vin.index]))
			print "     death: we say %g, it says %g" % (exp(-costd), exp(-vpairmap[-1].vout.outarccosts[g.sink.index]))
			for index, acostt in enumerate(costt):
				print "     trans[%i]: we say %g, it says %g" % \
				    (index, exp(-acostt), exp(-vpairmap[index].vout.outarccosts[vpairmap[index+1].vin.index]))
			for index, acostc in enumerate(costc):
				print "     clutter[%i]: we say %g, it says %g" % \
				    (index, exp(acostc), exp(vpairmap[index].vin.outarccosts[vpairmap[index].vout.index]))
	if not passed: raise ValueError
	print " ...pass"

	print "Check that running updateMinCostFlow() on a flow with no new data (already optimised using getMinCostFlow()) makes no change:"
	newmcf = g.updateMinCostFlow(mcf, None)
	if mcf != newmcf:
		print "mcf is:"
		print mcf
		print "newmcf is:"
		print newmcf
		raise ValueError
	else:
		print " ...pass"

	print "Check that running updateMinCostFlow() on a flow with one path deleted recovers a flow (typically an equivalent one) of same cost:"
	if len(mcf)==0:
		print " ...not possible to test, as the current flow is an empty one"
	else:
		newmcf = g.updateMinCostFlow(mcf[1:], None)
		oldsum = sum([fp.cost for fp in mcf])
		newsum = sum([fp.cost for fp in newmcf])
		if floatcmp(oldsum, newsum):
			print " ...pass"
		else:
			print mcf
			print newmcf
			raise ValueError("%g != %g" % (oldsum, newsum))

	print "Check that updateMinCostFlow() on the orig graph finds the same cost as the original result found by getMinCostFlow():"
	newmcf = g.updateMinCostFlow([], None)
	oldsum = sum([fp.cost for fp in mcf])
	newsum = sum([fp.cost for fp in newmcf])
	if floatcmp(oldsum, newsum):
		print " ...pass"
	else:
		print mcf
		print newmcf
		raise ValueError("%g != %g" % (oldsum, newsum))
	print "Check that updateMinCostFlow() on the orig graph finds the same # paths as the original result found by getMinCostFlow():"
	if len(mcf)==len(newmcf):
		print " ...pass"
	else:
		print mcf
		print newmcf
		raise ValueError("%i != %i" % (len(mcf), len(newmcf)))

	print "Check that getMinCostPath() matches getMinCostPath_BellmanFord() (on a MRPGraph, not a residual)"
	path1 = g.getMinCostPath()
	path2 = g.getMinCostPath_BellmanFord()
	if path1 == path2:
		print " ...pass"
	else:
		print path1.cost
		print path2.cost
		raise ValueError("%g != %g" % (path1, path2))


###############################################################################################
print "================================================================================"
print "UNIT TESTS PASSED. Press any key."
raw_input()
print "Now running a speed comparison - you can quit if you like."

from timeit import Timer
setup = """
from __main__ import makeRandomData, maxhop, transprobcallback, birthprobcallback, deathprobcallback, clutterprobcallback
from markovrenewal import MRPGraph
size = 200
data = makeRandomData(size, maxhop)
g = MRPGraph(data, transprobcallback, birthprobcallback, deathprobcallback, clutterprobcallback, maxtimedelta=maxhop)
"""

t1 = Timer("g.getMinCostPath()",             setup)
t2 = Timer("g.getMinCostPath_BellmanFord()", setup)
time1 = t1.timeit(number=10)
time2 = t2.timeit(number=10)
print "Run times:"
print "g.getMinCostPath():             %g" % time1
print "g.getMinCostPath_BellmanFord(): %g" % time2
raw_input()

t1 = Timer("g.getMinCostFlow()",            setup)
t2 = Timer("g.updateMinCostFlow([], None)", setup)
time1 = t1.timeit(number=10)
time2 = t2.timeit(number=10)
print "Run times:"
print "g.getMinCostFlow():            %g" % time1
print "g.updateMinCostFlow([], None): %g" % time2


