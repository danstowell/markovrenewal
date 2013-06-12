#!/bin/env python

"""This file contains some utility functions taken from Dan's larger file 'chiffchaffch.py'.
Excerpted here are the functions invoked by the main MMRP experiment scripts."""

import os.path

import matplotlib.pyplot as plt

# User parameters:
filteramp = True
filtermingap = True
doplots = True
basepath = os.path.expanduser("~/birdsong/xenocanto_chiffchaff")
syllgapsecs = 0.2  # chiffchaff sylls are typically 0.3 sec separated. So, we suppress chirps within 0.2s of a louder one.
framesize = 1024 #4096
hopsize = 0.5 #0.125

###############################################################################################
def write_the_csv(outpath, peaks):
	fp = open(outpath, "w")
	# make column headings
	fp.write('timepos,from,to,amp\n')
	# write data
	for peak in peaks:
		fp.write('%g,%g,%g,%g\n' % (peak['timepos'], peak['fromto'][0], peak['fromto'][1], peak['mag']))
	fp.close()

def filter_twoway(test, data):
	"Like filter(), but returns the passes AND the fails as two separate lists"
	collected = {True: [], False: []}
	for datum in data:
		collected[test(datum)].append(datum)
	return (collected[True], collected[False])

def filter_amp(peaks, maxamp, threshold=0.15):
	# filter out all atoms with amplitude less than a fraction of the peak
	ampthresh = threshold * maxamp
	(peaks, discarded) = filter_twoway(lambda p: p['mag'] > ampthresh, peaks)
	return (peaks, discarded)

def filter_mingap(peaks, syllgapsecs):
	"filter out all atoms within the syllgap of a more prominent chirp"
	newpeaks = []
	discardedpeaks = []
	for peak in peaks:
		#print "================================================"
		#print "Considering peak at time %g     (amplitude %g)  (so far we have %i peaks)" % (peak['timepos'], peak['mag'], len(newpeaks))
		neighbourhoodedge = peak['timepos'] - syllgapsecs
		#print "Looking as far back as timepos %g" % neighbourhoodedge
		# iterate back to the edge of the neighbourhood
		doneSomething = False
		for otherindex in xrange(-1, -1-len(newpeaks), -1):
			#print "Checking at index %i" % otherindex
			if newpeaks[otherindex]['timepos'] < neighbourhoodedge: # we've reached back older than neighbourhood
				#print "newpeaks[%i] is older than our gap, so we're discarding remainder and adding current" % (otherindex)
				if otherindex < -1:
					newpeaks = newpeaks[:otherindex+1]
				newpeaks.append(peak)
				doneSomething = True
				break
			if newpeaks[otherindex]['salience'] >= peak['salience']:  # we are beaten
				#print "newpeaks[%i] beats this one (%g >= %g) so discarding" % (otherindex, newpeaks[otherindex]['mag'], peak['mag'])
				discardedpeaks.append(peak)
				doneSomething = True
				break
		if not doneSomething:
			#print "Appending frame as only one, since iteration finished. (all must have been quieter and in range)"
			newpeaks = [peak]
	return (newpeaks, discardedpeaks)

def plot_kept_and_discarded(peaks, discardedpeaks, maxamp, framesizesecs, pdfpath, plotlimitsecs=99999):
	filename = os.path.splitext(os.path.split(pdfpath)[1])[0]
	fig = plt.figure()
	#ax = fig.add_subplot(211)
	for peak in discardedpeaks:
		if peak['timepos'] < plotlimitsecs:
			plt.plot([peak['timepos'], peak['timepos'] + framesizesecs], \
				[peak['fromto'][0], peak['fromto'][1]], \
				'#ff9966')  # '#ffff99')   # yellow deactivated cos reviewer
	sortedpeaks = sorted(peaks, key=lambda x: x['mag']) # plot weakest first; then shading makes more visual sense
	for peak in sortedpeaks:
		if peak['timepos'] < plotlimitsecs:
			plt.plot([peak['timepos'], peak['timepos'] + framesizesecs], \
				[peak['fromto'][0], peak['fromto'][1]], \
				str(1 - (peak['mag'] / maxamp)))
	plt.title(filename)
	if plotlimitsecs < 99999:
		plt.xlim(xmin=0, xmax=plotlimitsecs)
	else:
		plt.xlim(xmin=0)
	plt.ylim(ymin=0, ymax=10000)
	plt.xlabel("Time (s)")
	plt.ylabel("Freq (Hz)")
	plt.savefig(pdfpath, papertype='A4', format='pdf')

