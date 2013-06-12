#!/bin/env python

# measures used to evaluate MRP processes
# by Dan Stowell 2012

# If you run this module as a script, it shows some simple example clusterings and evaluates them.

from math import log, sqrt
from operator import itemgetter

def cluster_many_eval_stats(mixset, clusters, activitycurve_t, printindices=False):
	"""Runs many evaluation measures, inc cluster_eval_stats() on the indices but also activity curve etc"""
	# First the label-indices stuff:
	clusteredindices = [[hit['nn_gtboutindex'] for hit in cl[1]] for cl in clusters['clusters']]
	clutter = [hit['nn_gtboutindex'] for hit in clusters['other']]
	if printindices:
		print "Clusters' groundtruth labels:"
		for ci in clusteredindices: print ci
		print "Clutter groundtruth labels:"
		print clutter
	results = cluster_eval_stats(clusteredindices, clutter, string=False)

	# Now the other types of evaluation:
	results['Ftrans'] = fmeasure_transitions_reshape(mixset, clusters)
	results['Fsigtrans'] = fmeasure_transitions_reshape(mixset, clusters, sigtransonly=True)

	activitycurve_e = calcactivitycurve([cl[1] for cl in clusters['clusters']])
	results['corr_act'] = correlateActivityCurves(activitycurve_e, activitycurve_t)

	return results

def cluster_eval_stats(clusteredindices, clutter, string=True):
	"""Runs a standard set of evaluation measures given a clustering lits-of-lists and a clutter list (both holding GT indices).
	Returns a dict."""
	without_fps = filter(lambda ilist: len(ilist)>0, [filter(lambda x: x!=-1, ilist) for ilist in clusteredindices])
	signoise_sig = []
	for ilist in clusteredindices:
		signoise_sig.extend(map(lambda i: i!=-1, ilist))
	signoise_noi = map(lambda i: i!=-1, clutter)
	results = { \
# These are deactivated because I decided Fsn and Ftrans were the sanest evaluation stats for chch
#		"ARI-c": adjusted_rand_index(clusteredindices), \
#		"ARI--c": adjusted_rand_index(without_fps), \
#		"Unc": uncertainty_coef(clusteredindices), \
#		"Unc--c": uncertainty_coef(without_fps), \
#		"UncSN": uncertainty_coef([signoise_sig, signoise_noi]), \
		"Fsn": fmeasure(clusteredindices, clutter),
		}
	if string:
		results = ",    ".join(map(lambda k: "%s: %-5s" % (k, "%.3g" % results[k]), [  #"ARI-c", "ARI--c", "Unc", "Unc--c", "UncSN", 
							"Fsn"]))
	return results


def nchoose2(n):
	return sum(xrange(n))  # n-choose-2 == triangle numbers

def rand_index(clusteredindices):
	"supply a list of lists: each list is a cluster we found, and the indices in the lists are groundtruth cluster IDs"
	relist = []
	for j, foundclust in enumerate(clusteredindices):
		for i in foundclust:
			relist.append((i,j))
	matchtot = 0
	tot = 0
	for i in xrange(len(relist)):
		for j in xrange(i):
			tot += 1
			if (relist[i] == relist[j]) or ((relist[i][0] != relist[j][0]) and (relist[i][1] != relist[j][1])):
				matchtot += 1
	return float(matchtot)/tot

def adjusted_rand_index(clusteredindices):
	"supply a list of lists: each list is a cluster we found, and the indices in the lists are groundtruth cluster IDs"
	# Here we find the total number of items within each 2D bin and marginal (1D) bin, with "i" an index over true clust and "j" an index over found clust
	jsums = [len(foundclust) for foundclust in clusteredindices]
	n = sum(jsums)
	isums = {}
	eachcounts = []
	for j, foundclust in enumerate(clusteredindices):
		ourcounts = {}
		for i in foundclust:
			if i not in isums:
				isums[i] = 0
			isums[i] += 1
			if i not in ourcounts:
				ourcounts[i] = 0
			ourcounts[i] += 1
		eachcounts.extend(ourcounts.values())
	isums = isums.values()
	if sum(isums) != n:
		raise ValueError("Totals should be equal for isums (%g) and jsums (%g)" % (sum(isums), n))
	# These are the components of the ARI - see http://en.wikipedia.org/wiki/Rand_index#Adjusted_Rand_index for example
	index = float(sum(nchoose2(nij) for nij in eachcounts))
	suma  = float(sum(nchoose2(ni) for ni in isums))
	sumb  = float(sum(nchoose2(nj) for nj in jsums))
	expectedindex = suma * sumb / nchoose2(n)
	maxindex = 0.5 * (suma + sumb)

	adjustedindex = (index - expectedindex) / (maxindex - expectedindex)
	return adjustedindex



def entropy(distrib):
	"entropy H(i) =  - sum{ P(i) log(P(i)) }"
	normer = 1.0 / sum(p for p in distrib) # in case not normalised
	return - sum(p * normer * log(p * normer) for p in distrib)

def conditional_entropy(clustersofindices):
	# conditional entropy H(i|j) =  - sum{  P(i,j)        log(P(i|j))  }
	# conditional entropy H(i|j) =  - sum{  P(j) * P(i|j) log(P(i|j))  }
	condent = 0.0
	totalnumdata = sum(len(ilist) for ilist in clustersofindices)
	for j, ilist in enumerate(clustersofindices):
		if len(ilist)==0: continue
		cond_distrib = {}
		#print ilist
		for i in ilist:
			if i not in cond_distrib:
				cond_distrib[i] = 0
			cond_distrib[i] += 1
		normer = 1.0 / len(ilist)
		for i in cond_distrib:
			cond_distrib[i] *= normer
		p_j = float(len(ilist)) / totalnumdata
		condent += - sum(p_j * p_igj * log(p_igj) for p_igj in cond_distrib.values())
	return condent

def uncertainty_coef(clusteredindices):
	"""The un-symmetrised uncertainty coefficient.
	Supply a list of lists: each list is a cluster we found, and the indices in the lists are groundtruth cluster IDs"""
	if len(clusteredindices) == 0:
		print "Warning: empty list provided to uncertainty_coef()"
		return 0.0
	# entropy of GROUND-TRUTH
	distrib_gt = {}
	for ilist in clusteredindices:
		for i in ilist:
			if i not in distrib_gt:
				distrib_gt[i] = 0
			distrib_gt[i] += 1
	entropy_gt = entropy( distrib_gt.values() )
	if entropy_gt == 0:
		print "uncertainty_coef() warning: entropy is zero"
		return 1.0
	try:
		condent = conditional_entropy(clusteredindices)
	except:
		print clusteredindices
		raise
	return (entropy_gt - condent) / entropy_gt

def fmeasure(clusteredindices, clutter):
	"""F-measure for signal/clutter separation, not for clustering.
	-1 is the clutter label, so -1 in 'clutter' is true negative, all else is false negative;
	conversely in the clusterindices a -1 is a FP and all else is a TP.	
	"""
	if len(clusteredindices) == 0:
		print "Warning: empty list provided to fmeasure()"
		return 0.0
	tp = 0
	fp = 0
	for ilist in clusteredindices:
		for i in ilist:
			if i == -1:
				fp += 1
			else:
				tp += 1
	fn = 0
	for i in clutter:
		if i != -1:
			fn += 1
	precision = float(tp)/(tp+fp)
	recall    = float(tp)/(tp+fn)
	if precision==0. and recall==0.:
		print "fmeasure() WARNING: precision and recall both 0"
		return 0.0
	return 2 * precision * recall / (precision+recall)

def fmeasure_transitions_reshape(mixset, mrpclusters, sigtransonly=False):
	"Reshapes the input data appropriately and calls fmeasure_transitions(). Not too elegant overall."
	foundseqs = [[datum['datumindex'] for datum in cl[1]] for cl in mrpclusters['clusters']]
	trueseqs = [] # this will be v like mixset but without the "item" grouping layer above the "bout" grouping layer
	for sourceindex, item in enumerate(mixset):
		trueseqs.extend([[datum['datumindex'] for datum in bout] for bout in item['bouts']])
	return fmeasure_transitions(trueseqs, foundseqs, sigtransonly=sigtransonly)

def fmeasure_transitions_fromaba(aba, mrpclusters, sigtransonly=False):
	"Reshapes the input data appropriately (aba from pitchedabagen_multiplusclutter()) and calls fmeasure_transitions(). Not too elegant overall."
	foundseqs = [[datum['datumid'] for datum in cl[1]] for cl in mrpclusters['clusters']]
	trueseqs = {}
	for datum in aba:
		if datum['clusterid'] != -1: # ignore noise
			if datum['clusterid'] not in trueseqs:
				trueseqs[datum['clusterid']] = []
			trueseqs[datum['clusterid']].append(datum)
	for seq in trueseqs.values():
		seq.sort(key=itemgetter('timepos'))
	trueseqs = [[datum['datumid'] for datum in seq] for seq in trueseqs.values()] # reduce down to IDs
	return fmeasure_transitions(trueseqs, foundseqs, sigtransonly=sigtransonly)

def fmeasure_transitions(trueseqs, foundseqs, sigtransonly=False):
	"""F-measure, but evaluated for transitions (i.e. the arcs in the graph representation).
	The two args should each be a list of lists, sequences of groundtruth ID NUMBERS in the correct order (i.e. take care of time sorting).
	No need to supply clutter since by definition it contains no transitions.
	If 'sigtransonly', then arcs are only counted if BOTH of their events were correctly classed as signal."""

	truepairs  = []
	foundpairs  = []
	for alist in trueseqs:
		truepairs.extend([(alist[i],alist[i+1]) for i in xrange(len(alist)-1)])
	for alist in foundseqs:
		foundpairs.extend([(alist[i],alist[i+1]) for i in xrange(len(alist)-1)])

	if sigtransonly:
		"""to restrict consideration to arcs where both items were correctly classed as signal,
		this would mean we only consider arcs where both items'' ID numbers are found in BOTH the true list and the found list."""
		gtidlist = []
		for aseq in trueseqs: gtidlist.extend(aseq)
		foundpairs = [apair for apair in foundpairs if (apair[0] in gtidlist) and (apair[1] in gtidlist)]

		foundidlist = []
		for aseq in foundseqs: foundidlist.extend(aseq)
		truepairs = [apair for apair in truepairs if (apair[0] in foundidlist) and (apair[1] in foundidlist)]

	# ok now we're ready to do the standard calc
	tp = 0
	fp = 0
	fn = 0
	for apair in truepairs:
		if apair in foundpairs:
			tp += 1
		else:
			fn += 1
	for apair in foundpairs:
		# tp already counted
		if not apair in truepairs:
			fp += 1
	if tp == 0:
		print "Warning: tp==0 in fmeasure_transitions()"
		return 0.0
	else:
		precision = float(tp)/(tp+fp)
		recall    = float(tp)/(tp+fn)
	if precision==0. and recall==0.:
		print "fmeasure_transitions() WARNING: precision and recall both 0"
		return 0.0
	return 2 * precision * recall / (precision+recall)

############################################################################
# Activity curves:

def calcactivitycurve(bouts):
	"Given some bouts (real or inferred), creates a curve showing how many are active at any given time point"
	deltas = []
	for bout in bouts:
		startpos = min([x['timepos'] for x in bout])
		endpos   = max([x['timepos'] for x in bout])
		deltas.append({'timepos': startpos, 'delta': 1})
		deltas.append({'timepos': endpos  , 'delta':-1})
	# Now sort the deltas
	deltas.sort(cmp=lambda a,b: cmp(a['timepos'], b['timepos']))

	# Now run through the deltas, integrating
	count = 0
	curve = {0.0: 0}
	for breakpoint in deltas:
		count += breakpoint['delta']
		curve[breakpoint['timepos']] = count    # note, if multi on same timepos, this deliberately clobbers until all are processed
	return curve


def correlateActivityCurves(c0, c1):
	"""Given two lists of breakpoint-envelope dicts, each of the form {timepos: count}, calculates the correlation along their duration."""
	## first we build a unified 2D breakpoint envelope with segment-duration info
	unikeys = list(set(c0.keys() + c1.keys()))
	unikeys.sort()
	uni = []
	latestcount0 = 0
	latestcount1 = 0
	for whichkey in xrange(len(unikeys)-1):
		key = unikeys[whichkey]
		dur = unikeys[whichkey+1] - unikeys[whichkey]
		latestcount0 = c0.get(key, latestcount0)
		latestcount1 = c1.get(key, latestcount1)
		uni.append( (latestcount0, latestcount1, dur) )
	# Calculate duration-weighted means:
	mean0  = 0.
	mean1  = 0.
	totdur = 0.
	for (count0, count1, dur) in uni:
		mean0 += count0 * dur
		mean1 += count1 * dur
		totdur += dur
	mean0 /= totdur
	mean1 /= totdur
	# Now to calculate Pearson's r on the breakpoints:
	sum01 = 0.
	sum0 = 0.
	sum1 = 0.
	for (count0, count1, dur) in uni:
		dev0 = count0 - mean0
		dev1 = count1 - mean1
		sum01 += dev0 * dev1
		sum0  += dev0 * dev0
		sum1  += dev1 * dev1
	divisor = sqrt(sum0) * sqrt(sum1)
	if divisor == 0.:
		print "Warning: divisor is zero (no variation) in correlateActivityCurves()"
		return 0.
	r = sum01 / divisor
	return r

#####################################################################################
if __name__ == '__main__':
	print "================================================"
	print "                evaluators.py"

	for (name, array) in {
		"Perfect example": [[1,1,1,1], [2,2,2,2,2,2,2,2,2,2,2], [3,3,3,3,3]],
		"Good example": [[1,1,1,1], [2,2,2,2,2,2,2,2,2,2,2,1], [3,3,3,3,3]],
		"Weakly evidenced": [[1,1], [1,2]],
		"Totally scattered": [[1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1]],
		"Totally clumped": [[1,2,3,4,5,6,7,8,9]],
		"Bad example": [[1,2,3,2,1,2,3],[1,2,3,2,1]],
		"Worst possible": [[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]],
			}.iteritems():
		print ''
		print name
		print array
		print " RI: %g" % rand_index(array)
		print "ARI: %g" % adjusted_rand_index(array)
		print "Unc: %g" % uncertainty_coef(array)

