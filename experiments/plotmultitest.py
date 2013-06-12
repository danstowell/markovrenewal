#!/bin/env python
# plot results from linloggmm.multitest_abagen_linlog()
# by Dan Stowell, summer 2012

import os.path
import csv
from math import log, exp, pi, sqrt, ceil, floor
from numpy import mean, std
import matplotlib.pyplot as plt
import matplotlib.cm as cm

annotdir = os.path.expanduser("~/svn/stored_docs/python/markovrenewal/output")
plotfontsize = "large" #"xx-small"

def rescale(num):
#	return log((float(num) + 0.005) / (1.005 - num))  # logistic
	return log(1.01 - num)

# NB this array determines the order in which stats are processed too, which gives the order of the legend
#runlabels = [('ip','Ideal recovery, trained on test data'), ('i','Ideal recovery'), ('is','Ideal recovery plus synthetic noise'), ('a','Recovery from audio'), ('ba','Recovery from audio (baseline)')]
knownlabels = {'snrk': "SNR known", 'snru': "SNR unknown", 'snrug': "SNR unknown, greedy inference"}

def fmt_chooser(nmix, known):
#	return {1:'b', 2:'g', 4:'m'}[nmix] + {'snrk':'-.', 'snru':'-', 'snrug': ':'}[known]
	return {1:'b', 2:'b', 4:'b'}[nmix] + {'snrk':'-.', 'snru':'-', 'snrug': ':'}[known]


# load csv into nested dict structure data[whichstat][runtype][known][nmix][snr][]
data = {}
nmixes = []
snrs = []
rdr = csv.DictReader(open("%s/multitest_abagen_linlog.txt" % annotdir, 'rb'))

nruns=0
for row in rdr:
	if nruns==0:  # first row, infer nruns
		while ("val%i" % nruns) in row:
			nruns += 1
	row['runtype'] = 'merged'  # This is a HACK to merge the pdf of 'coh' and 'seg' together
	if row['whichstat'] not in data:
		data[                 row['whichstat']] = {}
	if row['runtype'] not in data[row['whichstat']]:
		data[                 row['whichstat']][row['runtype']] = {}

	row['nmix'] = int(row['nmix'])
	if row['nmix'] not in nmixes:
		nmixes.append(row['nmix'])
	if row['nmix'] not in data[   row['whichstat']][row['runtype']]:
		data[                 row['whichstat']][row['runtype']][row['nmix']] = {}

	if row['known'] not in data[  row['whichstat']][row['runtype']][row['nmix']]:
		data[                 row['whichstat']][row['runtype']][row['nmix']][row['known']] = {}

	row['snr'] = int(row['snr'])
	if row['snr'] not in snrs:
		snrs.append(row['snr'])
	if row['snr'] not in data[    row['whichstat']][row['runtype']][row['nmix']][row['known']]:
		data[                 row['whichstat']][row['runtype']][row['nmix']][row['known']][row['snr']] = []

	for i in xrange(nruns):
		val = float(row['val%i' % i])
		data[                 row['whichstat']][row['runtype']][row['nmix']][row['known']][row['snr']].append(val)

snrs.sort()
snrs.reverse()
snrsrange = (min(snrs)-1, max(snrs)+1)

yticks = [0.3, 0.6, 0.8, 0.9, 0.95, 0.99, 1.0]

# break it down into separate plots:
for whichstat, sdata in data.iteritems():
	for runtype, srdata in sdata.iteritems():
		for nmix, srndata in srdata.iteritems():
			# and in a single plot:
			fig = plt.figure()
			for known, srnkdata in srndata.iteritems():
				linedata = []
				for snr in snrs:
					numlist = srnkdata[snr]

					#numlist = [rescale(num) for num in numlist]  # no, do it after calc'ing stats

					# calc mean and stderr from 'numlist'
					themean = mean(numlist)
					stderr = std(numlist) / sqrt(len(numlist))

					# transform the data for readability:
					themean_l = rescale(themean)
					stderr_l_up = rescale(themean + stderr) - themean_l
					stderr_l_dn = themean_l - rescale(themean - stderr)

					linedata.append({'snr':snr, 'mean': themean_l, 'stderr_up': stderr_l_up, 'stderr_dn': stderr_l_dn})
				# draw a line
				plt.errorbar([x['snr']   for x in linedata], \
					     [x['mean']   for x in linedata], \
					     ([x['stderr_dn'] for x in linedata], [x['stderr_up'] for x in linedata]), \
					label="%i items, %s" % (nmix, knownlabels[known]), fmt=fmt_chooser(nmix, known))

			#plt.title("%s_%s" % (whichstat, runtype), fontsize=plotfontsize)
			plt.xlabel("SNR", fontsize=plotfontsize)
			plt.ylabel("%s" % whichstat, fontsize=plotfontsize)
			plt.xticks(snrs, fontsize=plotfontsize)
			plt.xlim(xmin=snrsrange[1], xmax=snrsrange[0])
			plt.ylim(ymin=rescale(0.3), ymax=rescale(1.001))
			plt.yticks(map(rescale, yticks), yticks, fontsize=plotfontsize)
			#plt.yticks(fontsize=plotfontsize)
			plt.legend(loc=(0.02, 0.05), prop={'size':'medium'})
			plt.savefig("%s/pdf/plot_multitest_%s_%s_%s.pdf" % (annotdir, whichstat, runtype, nmix), papertype='A4', format='pdf')

