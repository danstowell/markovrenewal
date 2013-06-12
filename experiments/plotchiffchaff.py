#!/bin/env python
# plot results from chiffchaff.py
# by Dan Stowell, summer 2012

import os.path
import csv
from math import log, exp, pi, sqrt, ceil, floor
from numpy import mean, std
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#annotdir = os.path.expanduser("~/birdsong/xenocanto_chiffchaff/xcor")
annotdir = os.path.expanduser("~/birdsong/xenocanto_chiffchaff/trimmed/xcor")
#annotdir = os.path.expanduser("~/birdsong/xenocanto_more_chiffchaff/trimmed/xcor")
#annotdir = os.path.expanduser("~/svn/stored_docs/python/markovrenewal/output")
#annotdir = os.path.expanduser("~/birdsong/xenocanto_chiffchaff/psts")
plotfontsize = "large" #"xx-small"

# very manual tweaking
verticalpos = {'Fsn':0.01, 'Ftrans':0.07, 'Fsigtrans':0.28}
ymins = {'Fsn':0.3, 'Ftrans':0.0, 'Fsigtrans':0.0}
horizontalpos = {'Fsn': 0.52}

specgrammodecode = ''  # choose: '', '_sash01', '_sash02'   ######### <== choose

# NB this array determines the order in which stats are processed too, which gives the order of the legend
runlabels = [('ip','Ideal recovery, trained on test data'), ('i','Ideal recovery'), ('is','Ideal recovery plus synthetic noise'), ('af','Recovery from audio (+fwise)'), ('a','Recovery from audio'), ('ag','Recovery from audio (greedy)'), ('ba','Recovery from audio (baseline)')]

runlabels = [rl for rl in runlabels if rl[0] not in ['af']]   # In the FIRST paper we did not explore this; the 'af' mode was only invented during the sash paper
#runlabels = [rl for rl in runlabels if rl[0] not in ['i', 'ip', 'is']]   # this makes the simpler plot without ideals

def fmt_chooser(runtype):
	return {'ip':'m:', 'i':'m--', 'is':'g-.', 'af':'k--', 'a':'k-', 'ag':'k:', 'ba':'b:'}[runtype]

# load csv into nested dict structure data[whichstat][runtype][nmix][]
data = {}
nmixes = []
rdr = csv.DictReader(open("%s/chchstats%s.csv" % (annotdir, specgrammodecode), 'rb'))

# infer how many runs were run
nruns = 0
while ('val%i' % nruns) in rdr.fieldnames:
	nruns += 1

for row in rdr:
	if row['whichstat'] not in data:
		data[row['whichstat']] = {}
	if row['runtype'] not in data[row['whichstat']]:
		data[row['whichstat']][row['runtype']] = {}
	row['nmix'] = int(row['nmix'])
	if row['nmix'] not in nmixes:
		nmixes.append(row['nmix'])
	if row['nmix'] not in data[row['whichstat']][row['runtype']]:
		data[row['whichstat']][row['runtype']][row['nmix']] = []
	for i in xrange(nruns):
		val = float(row['val%i' % i])
		data[row['whichstat']][row['runtype']][row['nmix']].append(val)

nmixrange = (min(nmixes)-1, max(nmixes)+1)

# plot, for each stat type:
for whichstat, sdata in data.iteritems():
	fig = plt.figure()
	# for each runtype:
	for runtype, runlabel in runlabels:
		if runtype not in sdata: continue
		srdata = sdata[runtype]
		linedata = []
		# for each nmix:
		for nmix, numlist in srdata.iteritems():
			# calc mean and stderr from 'numlist'
			themean = mean(numlist)
			stderr = std(numlist) / sqrt(len(numlist))
			linedata.append({'nmix':nmix, 'mean': themean, 'stderr': stderr})
		# draw a line
		plt.errorbar([x['nmix']   for x in linedata], \
		             [x['mean']   for x in linedata], \
		             [x['stderr'] for x in linedata], \
			label=runlabel, fmt=fmt_chooser(runtype))

	#plt.title("", fontsize=plotfontsize)
	plt.xlabel("Number of signals in mixture", fontsize=plotfontsize)
	plt.ylabel("%s" % whichstat, fontsize=plotfontsize)
	plt.xticks(nmixes, fontsize=plotfontsize)
	plt.yticks(fontsize=plotfontsize)
	plt.xlim(xmin=nmixrange[0], xmax=nmixrange[1])
	plt.ylim(ymin=ymins[whichstat], ymax=1.01)
	plt.legend(loc=(horizontalpos.get(whichstat, 0.02), verticalpos.get(whichstat, 0.15)), prop={'size':'small'})
	plt.savefig("%s/pdf/plot_chchstats_%s%s.pdf" % (annotdir, whichstat, specgrammodecode), papertype='A4', format='pdf')

