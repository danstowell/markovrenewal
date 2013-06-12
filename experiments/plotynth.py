#!/bin/env python
# plot results from ynthetictest.py
# by Dan Stowell, spring 2013

import os.path
import csv
from math import log, exp, pi, sqrt, ceil, floor
from numpy import mean, std, shape
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import itertools

#annotdir = os.path.expanduser("~/svn/stored_docs/python/markovrenewal/output")
annotdir = "output"
plotfontsize = "large" #"xx-small"

namelookup = {
	'fsn':'Fsn', 'ftrans':'Ftrans', 'fsigtrans':'Fsigtrans', 'msecs':'Run time (msecs)', \
	'birthdens_mism':'Error in assumed birth density (ratio)',
	'deathprob_mism':'Error in assumed death probability (ratio)',
	'snr_mism':'Error in assumed SNR (dB)',
	'gen_mism':'Proportion of errors in transition probabilities',
	'misseddetectionprob':'Missed detection probability',
	'noisecorr':'Amount of signal correlation imposed on noise',
	'snr':'SNR (dB)',
	'birthdens':'birth intensity',
	#'':'',
	}
def readable_name(name):
	return namelookup.get(name, name)

def fmt_chooser(currentcombi, groupcols, groupingvals):
	fmt = 'k'
	if groupcols[0]=='mmrpmode' and currentcombi[0]=='greedy':
		if (len(groupcols)>1) and groupingvals[groupcols[1]].index(currentcombi[1])>0:
			fmt += ':'
		else:
			fmt += '-.'
	else:
		if (len(groupcols)>1) and groupingvals[groupcols[1]].index(currentcombi[1])>0:
			fmt += '--'
		else:
			fmt += '-'
	return fmt


def ynth_csv_to_ciplot(csvpath, outpath, groupcols, summarycols, filtercols=None, xjitter=0.):
	"""
	groupcols:  used for discrete grouping of data, with the first one becoming the x-axis in a plot, remaining ones as multiple lines;
	summarycols: the name(s) of the columns to be made into y-values. one separate plot will be made for each.
	filtercols: {key->listofallowed...} select rows only where particular STRING values are found. otherwise, summaries are pooled over all values.
	"""
	data = ynth_csv_loaddata(csvpath, groupcols, summarycols, filtercols)
	# data is {'groupingvals':{ col: list  }, 'summarydata':{ tupleofgroupvals: { summarycol:{'mean': _, 'stderr': _} } } }

	csvname = os.path.splitext(os.path.basename(csvpath))[0]

	if isinstance(summarycols, basestring):  summarycols = [summarycols]
	if isinstance(groupcols, basestring):    groupcols   = [groupcols]
	# one plot for each summarycol
	for summarycol in summarycols:
		fig = plt.figure()

		# Now, we're going to use the first grouper as the x-axis.
		# This means we want to iterate over all combinations of the other groupers, drawing a line each time.
		for linegroupcombi in itertools.product(*[data['groupingvals'][col] for col in groupcols[1:]]):
			linedata = []
			for xval in data['groupingvals'][groupcols[0]]:
				fullgroupcombi = (xval,) + tuple(linegroupcombi)
				ourdata = data['summarydata'][fullgroupcombi][summarycol]
				if xjitter != 0:
					xval += random.gauss(0,xjitter)
				linedata.append({'xval':xval, 'mean': ourdata['mean'], 'stderr_up': ourdata['stderr'], 'stderr_dn': ourdata['stderr']})
			# draw a line
			linelabel = ', '.join([linegroupcombi[0]] + ["%s %s" % (readable_name(groupcols[lgi+2]), lg) for lgi, lg in enumerate(linegroupcombi[1:])])
			plt.errorbar([x['xval']   for x in linedata], \
				     [x['mean']   for x in linedata], \
				     ([x['stderr_dn'] for x in linedata], [x['stderr_up'] for x in linedata]), \
				label=linelabel, fmt=fmt_chooser(linegroupcombi, groupcols[1:], data['groupingvals']))


		#plt.title("%s_%s" % (whichstat, runtype), fontsize=plotfontsize)
		plt.xlabel(readable_name(groupcols[0]), fontsize=plotfontsize)
		plt.ylabel(readable_name(summarycol), fontsize=plotfontsize)
		plt.xticks(data['groupingvals'][groupcols[0]], fontsize=plotfontsize)
		xdatamax = max(data['groupingvals'][groupcols[0]])
		xdatamin = min(data['groupingvals'][groupcols[0]])
		plt.xlim(xmin=xdatamin-(xdatamax-xdatamin)*0.05, xmax=xdatamax+(xdatamax-xdatamin)*0.05)
		#yuck if groupcols[0] in ['deathprob_mism', 'birthdens_mism']:
		#yuck 	plt.xscale('log')
		if summarycol in ['msecs']:
			plt.yscale('log')
		else:
			plt.ylim(ymin=0.2, ymax=1) #rescale(0.3), ymax=rescale(1.001))
		#plt.yticks(map(rescale, yticks), yticks, fontsize=plotfontsize)
		plt.yticks(fontsize=plotfontsize)
		plt.legend(loc=(0.02, 0.05), prop={'size':'medium'})
		outfilepath = "%s/%s_%s.pdf" % (outpath, csvname, summarycol)
		plt.savefig(outfilepath, papertype='A4', format='pdf')
		print("Written file %s" % outfilepath)
		# LATER: consider how to avoid filename collisions - just allow user to specify a lbl?

def ynth_csv_to_surfaceplot(csvpath, outpath, groupcols, summarycols, filtercols=None):
	"""
	groupcols:  used for discrete grouping of data, with the first one becoming the x-axis in a plot, second as y-axis;
	summarycols: the name(s) of the columns to be made into y-values. one separate plot will be made for each.
	filtercols: {key->listofallowed...} select rows only where particular STRING values are found. otherwise, summaries are pooled over all values.
	"""
	data = ynth_csv_loaddata(csvpath, groupcols, summarycols, filtercols)
	# data is {'groupingvals':{ col: list  }, 'summarydata':{ tupleofgroupvals: { summarycol:{'mean': _, 'stderr': _} } } }

	csvname = os.path.splitext(os.path.basename(csvpath))[0]

	if isinstance(summarycols, basestring):  summarycols = [summarycols]
	if isinstance(groupcols, basestring):    groupcols   = [groupcols]
	if len(groupcols) != 2: raise ValueError("for surface plot, exactly 2 groupcols must be specified (used as X and Y).")
	# one plot for each summarycol
	for summarycol in summarycols:
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d') # 3D here

		# NOW DO A SURFACE PLOT
		data['groupingvals'][groupcols[0]].sort()
		ydata = map(float, data['groupingvals'][groupcols[1]])
		ydata.sort()
		data['groupingvals'][groupcols[1]].sort(cmp=lambda a,b: cmp(float(a), float(b)))
		z = [[data['summarydata'][(x,y)][summarycol]['mean'] for x in data['groupingvals'][groupcols[0]]] for y in data['groupingvals'][groupcols[1]]]

		ymesh = np.array([data['groupingvals'][groupcols[0]] for _ in range(len(data['groupingvals'][groupcols[1]]))])
		xmesh = np.array([ydata                              for _ in range(len(data['groupingvals'][groupcols[0]]))]).T
		z = np.array(z)
		ax.plot_surface(xmesh, ymesh, z, rstride=1, cstride=1)
		"""
		plt.imshow(z, interpolation='nearest', cmap=cm.binary)
		"""

		"""
		# Now, we're going to use the first grouper as the x-axis.
		# This means we want to iterate over all combinations of the other groupers, drawing a line each time.
		for linegroupcombi in itertools.product(*[data['groupingvals'][col] for col in groupcols[1:]]):
			linedata = []
			for xval in data['groupingvals'][groupcols[0]]:
				fullgroupcombi = (xval,) + tuple(linegroupcombi)
				ourdata = data['summarydata'][fullgroupcombi][summarycol]
				if xjitter != 0:
					xval += random.gauss(0,xjitter)
				linedata.append({'xval':xval, 'mean': ourdata['mean'], 'stderr_up': ourdata['stderr'], 'stderr_dn': ourdata['stderr']})
			# draw a line
			linelabel = ', '.join([linegroupcombi[0]] + ["%s %s" % (readable_name(groupcols[lgi+2]), lg) for lgi, lg in enumerate(linegroupcombi[1:])])
			plt.errorbar([x['xval']   for x in linedata], \
				     [x['mean']   for x in linedata], \
				     ([x['stderr_dn'] for x in linedata], [x['stderr_up'] for x in linedata]), \
				label=linelabel, fmt=fmt_chooser(linegroupcombi, groupcols[1:], data['groupingvals']))
		"""

		#plt.title("%s_%s" % (whichstat, runtype), fontsize=plotfontsize)
		"""
		plt.xlabel(readable_name(groupcols[0]), fontsize=plotfontsize)
		plt.ylabel(readable_name(groupcols[1]), fontsize=plotfontsize)
		plt.title(readable_name(summarycol), fontsize=plotfontsize)
		plt.xticks(range(len(data['groupingvals'][groupcols[0]])), data['groupingvals'][groupcols[0]], fontsize=plotfontsize)
		plt.yticks(range(len(data['groupingvals'][groupcols[1]])), data['groupingvals'][groupcols[1]], fontsize=plotfontsize)
		"""
		"""
		xdatamax = max(data['groupingvals'][groupcols[0]])
		xdatamin = min(data['groupingvals'][groupcols[0]])
		plt.xlim(xmin=xdatamin-(xdatamax-xdatamin)*0.05, xmax=xdatamax+(xdatamax-xdatamin)*0.05)
		ydatamax = max(data['groupingvals'][groupcols[0]])
		ydatamin = min(data['groupingvals'][groupcols[0]])
		plt.ylim(ymin=ydatamin-(ydatamax-ydatamin)*0.05, ymax=ydatamax+(ydatamax-ydatamin)*0.05)
		if summarycol in ['msecs']:
			plt.zscale('log')
		else:
			plt.zlim(ymin=0.2, ymax=1) #rescale(0.3), ymax=rescale(1.001))
		plt.zticks(fontsize=plotfontsize)
		#plt.legend(loc=(0.02, 0.05), prop={'size':'medium'})
		"""
		#can't for 3d:  plt.colorbar()
		outfilepath = "%s/%s_%s_surf.pdf" % (outpath, csvname, summarycol)
		plt.savefig(outfilepath, papertype='A4', format='pdf')
		print("Written file %s" % outfilepath)


def ynth_csv_loaddata(csvpath, groupcols, summarycols, filtercols=None):
	# load the csv data, applying filtering as we load, and floatifying the summarycols and groupcols
	# also build up some lists of the values found in the groupcols
	if isinstance(groupcols, basestring):
		groupcols = [groupcols]
	if isinstance(summarycols, basestring):
		summarycols = [summarycols]
	rdr = csv.DictReader(open(csvpath, 'rb'))
	groupingvals = {col:set() for col in groupcols}
	rawgroupeddata = {} # a dict where a TUPLE of groupedvals maps to a dict containing mean and ci
	for row in rdr:
		# filtering
		skiprow = False
		if filtercols:
			for (filtercol, allowedvals) in filtercols.items():
				if row[filtercol] not in allowedvals:
					skiprow = True
					break
		if skiprow: continue
		# floatify
		# CANNOT (eg for mmrpmode): for col in   groupcols: row[col] = float(row[col])
		row[groupcols[0]] = float(row[groupcols[0]])
		for col in summarycols: row[col] = float(row[col])
		# record the grouping values
		for col in   groupcols: groupingvals[col].add(row[col])
		# and of course store the datum
		groupindex = tuple(row[col] for col in groupcols)
		if groupindex not in rawgroupeddata:
			rawgroupeddata[groupindex] = []
		rawgroupeddata[groupindex].append(row)

	# then construct the summary results: a dict where a TUPLE of groupedvals maps to a dict containing mean and ci
	summarydata = {}
	for groupindex, datalist in rawgroupeddata.items():
		ourstats = {}
		for whichsummarycol in summarycols:
			numlist = [datum[whichsummarycol] for datum in datalist]
			themean = mean(numlist)
			stderr = std(numlist) / sqrt(len(numlist))
			ourstats[whichsummarycol] = {'mean':themean, 'stderr':stderr}
		summarydata[groupindex] = ourstats

	# return the groupcol listing and the big dict of summary data
	for col in groupcols:
		groupingvals[col] = list(groupingvals[col])
		groupingvals[col].sort()
	return {'groupingvals':groupingvals, 'summarydata':summarydata}

################################################################################################################
if __name__ == '__main__':
	# NOTE: filtercols must list string values not floats
	ynth_csv_to_ciplot("%s/ynth_varying1.csv" % annotdir, "%s/pdf" % annotdir, \
		groupcols=['snr', 'mmrpmode', 'birthdens'], summarycols=['fsn', 'fsigtrans', 'msecs'], filtercols=None, xjitter=0.1)
	#ynth_csv_to_ciplot("%s/ynth_varying100.csv" % annotdir, "%s/pdf" % annotdir, \
	#	groupcols=['snr', 'mmrpmode', 'birthdens'], summarycols=['fsn', 'fsigtrans', 'msecs'], filtercols=None, xjitter=1.1)
	ynth_csv_to_ciplot("%s/ynth_sens_snr.csv" % annotdir, "%s/pdf" % annotdir, \
		groupcols=['snr_mism', 'mmrpmode' #, 'snr'
			], summarycols=['fsn', 'fsigtrans'], filtercols=None)
	ynth_csv_to_ciplot("%s/ynth_sens_birth.csv" % annotdir, "%s/pdf" % annotdir, \
		groupcols=['birthdens_mism', 'mmrpmode'], summarycols=['fsn', 'fsigtrans'], filtercols=None)
	ynth_csv_to_ciplot("%s/ynth_sens_death.csv" % annotdir, "%s/pdf" % annotdir, \
		groupcols=['deathprob_mism', 'mmrpmode'], summarycols=['fsn', 'fsigtrans'], filtercols=None)
	ynth_csv_to_ciplot("%s/ynth_sens_noisecorr.csv" % annotdir, "%s/pdf" % annotdir, \
		groupcols=['noisecorr', 'mmrpmode'], summarycols=['fsn', 'fsigtrans', 'msecs'], filtercols=None)  # added msecs to noisecorr since long
	ynth_csv_to_ciplot("%s/ynth_sens_missed.csv" % annotdir, "%s/pdf" % annotdir, \
		groupcols=['misseddetectionprob', 'mmrpmode'], summarycols=['fsn', 'fsigtrans'], filtercols=None)
	ynth_csv_to_ciplot("%s/ynth_sens_tt.csv" % annotdir, "%s/pdf" % annotdir, \
		groupcols=['gen_mism', 'mmrpmode'], summarycols=['fsn', 'fsigtrans'], filtercols=None)

#	ynth_csv_to_surfaceplot("%s/ynth_sens_snr.csv" % annotdir, "%s/pdf" % annotdir, \
#		groupcols=['snr_mism', 'snr'], summarycols=['fsn', 'fsigtrans'], filtercols={'mmrpmode':['full']}) # full inference only

