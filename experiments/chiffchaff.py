#!/bin/env python

# script to analyse mixtures of chiffchaff audios
# by Dan Stowell, summer 2012

from glob import glob
from subprocess import call
import os.path
import csv
from math import log, exp, pi, sqrt, ceil, floor
from numpy import array, mean, cov, linalg, dot, median, std
import numpy as np
import tempfile
import shutil
from operator import itemgetter
from copy import copy, deepcopy
from sklearn.mixture import GMM
import gc
from random import uniform, shuffle
import time, datetime

from markovrenewal import mrp_autochunk_and_getclusters  # MRPGraph
from evaluators import *

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from colorsys import hsv_to_rgb
import random
import os, sys

#################################################################################################
# USER SETTINGS

#analysistype = "chrm"  # Ring-modulation chirplet analysis, facilitated by "chiffchaffch.py" and the other files in my chirpletringmod folder
#analysistype = "chmp"  # MPTK chirplet analysis, facilitated by "likechrm.py" and the other files in my py4mptk folder
analysistype = "xcor"  # Spectrotemporal cross-correlation template-matching, carried out by "xcordetect.py"
#analysistype = "psts"  # Uses another MRP underneath, to stitch peaks into tweets, then models each with a polynomial fit

usetrimmed = True
datadir = os.path.expanduser("~/birdsong/xenocanto_chiffchaff")
if usetrimmed: datadir += "/trimmed"
mp3dir = datadir+"/mp3"
wavdir = datadir+"/wav"
annotdir = datadir+"/"+analysistype
csvdir = datadir+"/csv"

maxtestsetsize = 5

boutgapsecs = 1 #2    # chiffchaff bouts are typically at least 3 sec separated. this value is used to segment bouts. (since syll gap is typically 0.35 sec, this allows one or even two dropped sylls to remain in-bout.)
maxsyllgapsecs = 1   # this is passed to the MRP as the longest it needs to consider

fastdev = False # set this to True to miss out lots of tests, for rapid-tweaking purposes (doesn't make same plots)
if fastdev:
	froz_nmix = [2,4]
	froz_permute = [0,10,15]
else:
	froz_nmix = [1,2,3,4,5]
	froz_permute = None
fewerruntypes = fastdev   # For the paper with Sash, we don't need the extra run types, so hard code this to True

#######FOR DEVELOPMENT ONLY: use a frozen pre-made MPTK analysis
#frozpath = os.path.expanduser("~/birdsong/xenocanto_chiffchaff/chmp/frozen_24_tmpgOtHD__chiffchaff")
frozpath = None

#################################################################################################
# loading modules depending on user settings (inc localpath ones)

if analysistype=='xcor':
	import xcordetect as xcor
	specgrammode = xcor.default_specgrammode
	if specgrammode==None:
		specgrammodecode = ''
	else:
		specgrammodecode = '_%s' % specgrammode
	print "Building xcor template"
	xcor_gridinfo = xcor.get_gridinfoGMM_cacheable(datadir, specgrammode)

if analysistype[:3]=='pst':
	import peakstitcher

if analysistype=='chrm':
	cmd_folder = os.path.realpath(os.path.abspath("../chirpletringmod/"))
	if cmd_folder not in sys.path:
		sys.path.insert(0, cmd_folder)
	import fileutils as chf
if analysistype=='chmp':
	cmd_folder = os.path.realpath(os.path.abspath("../py4mptk/"))
	if cmd_folder not in sys.path:
		sys.path.insert(0, cmd_folder)
	import likechrm

starttime = time.time()

#################################################################################################
# PREPROCESSING

wavpaths = glob(wavdir + "/XC*.wav")
# Check the WAVs have been made from the MP3s
if len(wavpaths) == 0:
	raise ValueError("Found no WAVs - please run the script to automatically convert the MP3s (chiffchaff_mp3wav.bash)")
if len(wavpaths) <= (maxtestsetsize * 2):
	raise ValueError("Only found %i WAVs, whereas test set is expected to be of size %i and we need a training set too" % (len(wavpaths), maxtestsetsize))

# Load the data:
gtboutindex = 0
def makeItemObject(wavpath):
	"Defines a simple struct data layout, also checking that the annotations exist and loading them, chunked into bouts."
	global gtboutindex

	basename = os.path.splitext(os.path.split(wavpath)[1])[0]
	annotpath   = "%s/%s%s.csv"       % (annotdir, basename, specgrammodecode)
	annotpath_d = "%s/%s%s_noise.csv" % (annotdir, basename, specgrammodecode)
	if not os.path.exists(annotpath):
		raise ValueError("Failed to find expected annotation file %s -- you may need to rerun %s" % (annotpath, { \
			'chrm':'../chirpletringmod/chiffchaffch.py', \
			'xcor':'xcordetect.py',\
			'psts':'peakstitcher.py',\
			}[analysistype]))

	# load the annotations; and SEPARATE THE DATA INTO BOUTS using boutgapsecs
	rdr = csv.DictReader(open(annotpath, 'rb'))

	csvdata = [{key:float(row[key]) for key in row} for row in rdr]
	if analysistype=='xcor':
		xcor.add_fwisebin_data_to_csvdata(csvdata, "%s/%s%s.fwisebins.csv" % (annotdir, basename, specgrammodecode))
	csvdata.sort(key=itemgetter('timepos'))   # TBH I think it's usually sorted already, but belt-and-braces is OK

	bouts = [[]]
	prevrow = None
	gtboutindex += 1   # ensure consecutive files don't share index
	for row in csvdata:
		if ('amp' in row) and not ('mag' in row):   # harmonise terminology
			row['mag'] = row['amp']
			del row['amp']
		if prevrow != None:
			if (row['timepos'] - prevrow['timepos']) > boutgapsecs:
				if len(bouts[-1])==0:
					print wavpath
					raise ValueError(str(row))
				bouts.append([]) # begin a new bout
				gtboutindex += 1
		row['gtboutindex'] = gtboutindex
		row['fromto'] = (row['from'], row['to']) # bleh
		bouts[-1].append(row)
		prevrow = row

	# Now load the noise. This is simpler because we don't care about bouts or gaps.
	rdr = csv.DictReader(open(annotpath_d, 'rb'))
	noise = []
	for row in rdr:
		row = {key:float(row[key]) for key in row}
		if ('amp' in row) and not ('mag' in row):   # harmonise terminology
			row['mag'] = row['amp']
			del row['amp']
		row['fromto'] = (row['from'], row['to']) # bleh
		noise.append(row)
	if analysistype=='xcor':
		xcor.add_fwisebin_data_to_csvdata(noise, "%s/%s%s_noise.fwisebins.csv" % (annotdir, basename, specgrammodecode))

	return {'annotpath':annotpath, 'basename':basename, 'wavpath':wavpath, 'bouts': bouts, 'noise': noise}

items = map(makeItemObject, wavpaths)

#################################################################################################
# TRAINING

def fitGmm(anarray):
	"This ALSO normalises and returns the normalisation vectors"
	themean = mean(anarray, 0)
	theinvstd = std(anarray, 0)
	for i,val in enumerate(theinvstd):
		if val == 0.0:
			theinvstd[i] = 1.0
		else:
			theinvstd[i] = 1.0 / val		
	print "theinvstd: ", theinvstd
	thegmm = GMM(n_components=10, cvtype='full')

	# DEACTIVATED STD STANDARDISATION
	theinvstd = array([1.0 for _ in xrange(len(anarray[0]))])

	if len(anarray)<10:
		anarray = np.vstack((anarray, anarray)) # because scipy refuses if <10
	thegmm.fit((anarray - themean) * theinvstd)   # with standn
	return {'gmm':thegmm, 'mean':themean, 'invstd':theinvstd}

def unigramvectorfromsyll(syll, vecextramode=None):
	"Returns a list of unigram data (to be concatenated if making bigram data)"
	# NB do not return numpy array, return standard list (because of concatenation!)
	if vecextramode=='fwise':
		# NOTE: this xcor function deliberately limits the number of frames used, for manageable data size
		return map(log, xcor.fwisebindata_as_vector(syll['fwise']))
	else:
		return map(log, [syll['fromto'][0], syll['fromto'][1]])

def bigramvectorfromsyll(frm, too, vecextramode=None):
	timedelta = too['timepos'] - frm['timepos']
	magratio  = too['mag']     / frm['mag']
	return unigramvectorfromsyll(frm, vecextramode) + map(log, [timedelta, magratio]) + unigramvectorfromsyll(too, vecextramode)

def trainModel(items, plot=False, vecextramode=None):
	"Supply some subset of 'items' and this will train a Gaussian from the log-time-deltas and log-freqs and log-mag-ratios"
	trainingdata = [] # pairwise
	marginaltrainingdata = [] # unigramwise
	noisetrainingdata = [] # unigramwise

	for item in items:
		for bout in item['bouts']:
			for i in xrange(len(bout)-1):
				timedelta = bout[i+1]['timepos'] - bout[i]['timepos']
				magratio  = bout[i+1]['mag']     / bout[i]['mag']
				vector = bigramvectorfromsyll(bout[i], bout[i+1], vecextramode)
				trainingdata.append(vector)
			for datum in bout:
				vector = unigramvectorfromsyll(datum, vecextramode)
				marginaltrainingdata.append(vector)
		for datum in item['noise']:
			vector = unigramvectorfromsyll(datum, vecextramode)
			noisetrainingdata.append(vector)
	trainingdata = array(trainingdata)

	avgpathlen = mean([mean([len(bout) for bout in item['bouts']]) for item in items])

	thegmm      = fitGmm(trainingdata)   # p(X & Y)
	marginalgmm = fitGmm(array(marginaltrainingdata))   # p(X)

	# noise model is similar to the marginal, in that it's just 2D [from, to]
	while len(noisetrainingdata) < 10: noisetrainingdata += noisetrainingdata   # a bit hacky - it's cos GMM refuses to fit to few datapoints; not too crucial here
	noisegmm = fitGmm(array(noisetrainingdata))

	model = {'gmm':thegmm, 'avgpathlen':avgpathlen, 'marginalgmm': marginalgmm, 'noisegmm': noisegmm}
	return model

def plottimedeltas(items):
	"Plots a histo of the timedeltas for each separate training file"
	fig = plt.figure()
	deltas = []
	mintime=0
	maxtime=0
	for item in items:
		deltas.append({'label': item['basename'], 'vals':[]})
		for bout in item['bouts']:
			for i in xrange(len(bout)-1):
				timedelta = bout[i+1]['timepos'] - bout[i]['timepos']
				if timedelta < mintime:
					mintime = timedelta
				if timedelta > maxtime:
					maxtime = timedelta
				deltas[-1]['vals'].append(timedelta)

	for whichplot, anitem in enumerate(deltas):
		ax = fig.add_subplot(len(deltas), 1, 1 + whichplot)
		if whichplot==0:
			plt.title("Time deltas in each training file")
		plt.hist(anitem['vals'], 40, range=(mintime, maxtime))
		plt.yticks([0], [anitem['label']], fontsize='xx-small')
		if whichplot != len(deltas)-1:
			plt.xticks([])
	plt.savefig("%s/pdf/plot_timedeltahistos.pdf" % annotdir, papertype='A4', format='pdf')
	plt.clf() # must do this - helps prevent memory leaks with thousands of Bboxes

def standardise_vector(vector, submodel):
	mean   = submodel['mean']
	invstd = submodel['invstd']
	return [(x - mean[i]) * invstd[i] for i,x in enumerate(vector)]

def likelihood_signal_model(model, frm, too, vecextramode=None):
	"Evaluates the conditional likelihood of a transition represented by 'vector' under model 'model'"
	fullprob = model['gmm']        ['gmm'].eval([   standardise_vector(bigramvectorfromsyll(frm, too, vecextramode=vecextramode)    , model['gmm'        ])   ])[0][0]  # log(P(A AND B))
	fromprob = model['marginalgmm']['gmm'].eval([   standardise_vector(unigramvectorfromsyll(frm, vecextramode=vecextramode), model['marginalgmm'])   ])[0][0]  # log(P(A))
	return exp(fullprob - fromprob)   # P(B | A)

def likelihood_marginal_model(model, syll, vecextramode=None):
	"Evaluates the likelihood of a single datum represented by 'vector' under model 'model', IGNORING TRANSITIONS ETC"
	aprob = model['marginalgmm']['gmm'].eval([   standardise_vector(unigramvectorfromsyll(syll, vecextramode=vecextramode), model['marginalgmm'])   ])[0][0]  # log(P(A))
	return exp(aprob)   # P(A)

def likelihood_noise_model(model, syll, vecextramode=None):
	"Evaluates the likelihood of a datum represented by 'vector' under the NOISE model in 'model'"
	aprob = model['noisegmm']['gmm'].eval([   standardise_vector(unigramvectorfromsyll(syll, vecextramode=vecextramode), model['noisegmm'])   ])[0][0]  # log(P(N))
	return exp(aprob)

def sequenceloglikelihood(sequence, model, vecextramode=None):
	"For a single cluster, finds the log-likelihood of the entire transition sequence (WITHOUT including birth/death)"
	ll = 0.0
	for i in xrange(len(sequence)-1):
		a = sequence[i]
		b = sequence[i+1]
		prob = likelihood_signal_model(model, a, b, vecextramode=vecextramode)
		ll += log(prob)
	return ll

def chmodelMRPGraph_andgetclusters(data, model, estsnr=200, greedynotfull=False, vecextramode=None):
	"""Factory method for using MRPGraph with chiffchaff signal model and fixed probys derived from estimated lengths and SNRs.
	'expectedpathlen' is the expected number of emissions in a MRP sequence - if deathprob is fixed then it's exponential decay over sequence index.
	'estsnr' is an estimate of signal-to-noise ratio (NOT in dB, as a ratio) - e.g. "2" means you estimate two-thirds of the datapoints to be non-clutter.
	"""
	deathprob = (1./model['avgpathlen'])      # exponential decay relationship
	birthprob =   estsnr / ((1. + estsnr) * model['avgpathlen'])
	clutterprob = 1.     / (1.+estsnr)
	print "Probabilities derived: birth %g, death %g, clutter %g." % (birthprob, deathprob, clutterprob)
	def transprobcallback(a,b):
		#prob = likelihood_signal_model(model, [a['fromto'][0], a['fromto'][1], b['timepos']-a['timepos'], b['mag']/a['mag'], b['fromto'][0], b['fromto'][1]])
		prob = likelihood_signal_model(model, a, b, vecextramode=vecextramode)
		# sparsify the graph to make inference more efficient - when prob==0, arcs are not created
		if prob < 1e-22:
			return 0.
		else:
			return prob
	birthprobcallback   = lambda a: birthprob
	deathprobcallback   = lambda a: deathprob
	def clutterprobcallback(a):
		return clutterprob   *   likelihood_noise_model(model, a, vecextramode=vecextramode)
	cl = mrp_autochunk_and_getclusters(
			# MRP ctor args:
			data, transprobcallback, birthprobcallback, deathprobcallback, clutterprobcallback, maxtimedelta=maxsyllgapsecs,
			# cluster-getting args:
			numcutoff=90, greedynotfull=greedynotfull
			)
	return cl

def chmodel_baseline_andgetclusters(data, model, vecextramode=None):
	"""Baseline system - simple signal/noise likelihood test for each datum, and clustering
	based on a time-gap threshold. Does not use any transition information.
	returns, like the MRP_agc does, {'other': [theclutter], 'clusters': [[val, [thecluster]], ...]}"""
	data.sort(key=itemgetter('timepos'))
	clusters = []
	noise = []
	prevtimepos = -9e99
	for datum in data:
		sig_like   = likelihood_marginal_model(model, datum, vecextramode=vecextramode)
		noise_like = likelihood_noise_model(   model, datum, vecextramode=vecextramode)
		if sig_like >= noise_like:
			if (datum['timepos'] - prevtimepos) > 0.7:
				# start a new cluster
				clusters.append([0., []])
			clusters[-1][0] += log(sig_like / noise_like)
			clusters[-1][1].append(datum)
			prevtimepos = datum['timepos']
		else:
			noise.append(datum)
	return {'clusters':clusters, 'other':noise}

trainModel(items, True)
plottimedeltas(items)

#################################################################################################
# TESTING

def analysemixedwav(mixedwav, frozpath=None):
	"'frozpath' is, for development purposes, a path to a 'frozen' copy of the analysis data, so 'chmp' mode won't run..."
	unframed = []
	if analysistype == "chrm":
		framesize = 1024 #4096   ### Make sure this matches what chiffchaffch is using
		hopsize   = 0.5   # 0.125  ### Make sure this matches what chiffchaffch is using
		chfanalysis = chf.analysefile(mixedwav, numtop=1, framesize=framesize, hopsize=hopsize)
		hopsecs = float(chfanalysis['framesize'] * chfanalysis['hopsize']) / chfanalysis['srate']
		framesizesecs = float(chfanalysis['framesize']) / chfanalysis['srate']
		# we liberate peaks from their frames
		for framepos, frame in enumerate(chfanalysis['frames']):
			for peak in frame['peaks']:
				peak['timepos'] = framepos * hopsecs
				if peak['salience'] > 0:    # filter to just downwards
					unframed.append(peak)
	elif analysistype == "chmp":
		skipprocessing = (frozpath != None)
		pass_in_tmpdir = frozpath or tmpdir
		(rawdata, _) = likechrm.mptk_wav2csv_likechrm_one(mixedwav, tmpdir=pass_in_tmpdir, \
				filtermingap=False, filteramp=True, ampthresh=0.001, snr=16, skipprocessing=skipprocessing)
		hopsecs = likechrm.hopsecs
		framesizesecs = likechrm.framesizesecs
		for peak in rawdata:
			if peak['salience'] > 0:    # filter to just downwards
				unframed.append(peak)
	elif analysistype == "xcor":
		# ampthreshes here are done to match what's done in the orig analysis
		ampthresh = {None: 0.8, 'sash01': 0.2, 'sash02': 0.05}[specgrammode]
		(unframed, _) = xcor.xcor_likechrm_one(xcor_gridinfo, datadir, mixedwav, filteramp=True, filtermingap=False, plot=True, ampthresh=ampthresh,
					specgrammode=specgrammode,
					plotlimitsecs=6
						)
		framesizesecs = float(xcor.framelen)/xcor.fs
		hopsecs = framesizesecs * xcor.hop
	elif analysistype[:3] == "pst":
		print "* * * peakstitcher * * *"
		(unframed, _) = peakstitcher.pst_likechrm_one(datadir, mixedwav, analysistype[3], filteramp=True, filtermingap=False, plot=True, ampthresh=0.1)
		for datum in unframed: datum['salience'] = datum['mag']
		framesizesecs, hopsecs = peakstitcher.framesizesecs_hopsecs()
	else:
		raise ValueError("Unknown analysis type %s" % analysistype)

	print "---------------chfanalysis unframed----------------------"
	saliences = [x['salience'] for x in unframed]
	print "%i saliences: min, mean, median, max: %g, %g, %g, %g" % (len(saliences), min(saliences), mean(saliences), median(saliences), max(saliences))
	# Filter out the weak ones:
	thresh = max(saliences) * 0.1
	unframed = filter(lambda x: x['salience'] > thresh, unframed)
	saliences = [x['salience'] for x in unframed]
	unframed.sort(key=itemgetter('timepos'))
	print "%i saliences: min, mean, median, max: %g, %g, %g, %g" % (len(saliences), min(saliences), mean(saliences), median(saliences), max(saliences))
	return (hopsecs, framesizesecs, unframed)

def plotactivitycurve(curve, label="0"):
	pdfpath = "%s/pdf/activity_%s.pdf" % (annotdir, str(label))
	#print "Will write to %s" % pdfpath

	sortedtimes = sorted(curve.keys())
	sortedvals  = [curve[key] for key in sortedtimes]
	plotfontsize = "xx-small"
	fig = plt.figure()
	plt.title(label, fontsize=plotfontsize)
	plt.plot(sortedtimes, sortedvals, drawstyle='steps-post')
	plt.ylabel("Num active bouts", fontsize=plotfontsize)
	plt.xticks(fontsize=plotfontsize)
	plt.yticks(fontsize=plotfontsize)
	plt.xlabel("Time (s)", fontsize=plotfontsize)
	plt.savefig(pdfpath, papertype='A4', format='pdf')
	plt.clf() # must do this - helps prevent memory leaks with thousands of Bboxes

def plotunframed(framesizesecs, unframed, label="0", numsplit=4, colormapper=None):
	pdfpath = "%s/pdf/mixture_%s.pdf" % (annotdir, str(label))
	#print "Will write %i peaks to %s" % (len(unframed), pdfpath)
	if len(unframed)==0:
		print "Warning: plotunframed() got zero peaks; won't plot %s" % pdfpath
		return
	maxmag = max([x['mag'] for x in unframed])
	plotfontsize = "xx-small"
	fig = plt.figure()
	chunkdur = (max([x['timepos'] for x in unframed]) + 0.01) / float(numsplit)
	chunkclumps = [[] for _ in xrange(numsplit)]
	for peak in unframed:
		chunkclumps[int(floor((peak['timepos']/chunkdur)))].append(peak)
	peaksplotted = 0
	for whichsplit, chunk in enumerate(chunkclumps):
		ax = fig.add_subplot(numsplit,1,whichsplit+1)
		if len(chunk)!=0 and 'nn_gtsourceindex' in chunk[0]:
			# sort so that clutter is plotted first and goes to the bottom
			chunk.sort(key=itemgetter('nn_gtsourceindex'))
		if whichsplit == 0:
			plt.title(label, fontsize=plotfontsize)
		for peak in chunk:
			alpha = 1 - (peak['mag'] / maxmag)
			if colormapper==None:
				col = [0,0,0]
			else:
				col = colormapper(peak)
			plt.plot([peak['timepos'], peak['timepos'] + framesizesecs], \
			        [peak['fromto'][0], peak['fromto'][1]], \
			        color=col, alpha=alpha)
			plt.xlim(            xmin=chunkdur * (whichsplit),       xmax=chunkdur * (whichsplit+1))
			plt.xticks(range(int(ceil(chunkdur * (whichsplit))), int(ceil(chunkdur * (whichsplit+1)))), fontsize=plotfontsize)
			plt.ylim(2000, 9000)
			peaksplotted += 1
		plt.ylabel("Freq (Hz)", fontsize=plotfontsize)
		plt.yticks(fontsize=plotfontsize)
	print "plotunframed(%s) plotted %i peaks" % (pdfpath, peaksplotted)
	plt.xlabel("Time (s)", fontsize=plotfontsize)
	plt.savefig(pdfpath, papertype='A4', format='pdf')
	plt.clf() # must do this - helps prevent memory leaks with thousands of Bboxes

def plot_gtindex_vs_time(clusteredindicesforplot, label="0"):
	"data supplied is clustered lists of items in the form {timepos, gtboutindex_f, mag} with _f meaning mildly fuzzed"
	pdfpath = "%s/pdf/gtindex_%s.pdf" % (annotdir, str(label))
	if sum(len(cl) for cl in clusteredindicesforplot)==0:
		print "Warning: plot_gtindex_vs_time() got zero peaks; won't plot %s" % pdfpath
		return
	maxmag = max([max([peak['mag'] for peak in cl]) for cl in clusteredindicesforplot])
	plotfontsize = "xx-small"
	fig = plt.figure()

	for cl in clusteredindicesforplot:
		xdata = [peak['timepos']            for peak in cl]
		ydata = [peak['gtboutindex_f']      for peak in cl]
		alpha = [1 - (peak['mag'] / maxmag) for peak in cl]
		plt.plot(xdata, ydata, 'x-', alpha=sum(alpha) / float(len(alpha)))

	plt.title(label, fontsize=plotfontsize)
	plt.ylabel("gt cluster index", fontsize=plotfontsize)
	plt.xticks(fontsize=plotfontsize)
	plt.yticks(fontsize=plotfontsize)
	plt.xlabel("Time (s)", fontsize=plotfontsize)
	plt.savefig(pdfpath, papertype='A4', format='pdf')
	plt.clf() # must do this - helps prevent memory leaks with thousands of Bboxes

def rainbowcolour(index, length):
	return hsv_to_rgb(float(index)/length, 1.0, 0.9)

def plottimelimit(data):
	"Handy data truncation for scrying plots"
	return filter(lambda x: x['timepos']< 309999, data) # 12

def plotunframed_rainbow(clusters, framesizesecs, label):
	mrpcolours = {-1: (0.85,0.85,0.85,0.5)}
	for clustindex in xrange(len(clusters['clusters'])):
		mrpcolours[clustindex] = rainbowcolour(clustindex, len(clusters['clusters']))
	plotunframed(framesizesecs, plottimelimit(decluster_remember_indices(clusters)), label=label, numsplit=4, \
		colormapper=lambda x: mrpcolours[x['clustid']])

def decluster_remember_indices(clusters):
	declustered = []
	for clid, cl in enumerate(clusters['clusters']):
		for x in cl[1]:
			x['clustid'] = clid
		declustered.extend(cl[1])
	for x in clusters['other']:
		x['clustid'] = -1
	declustered.extend(clusters['other'])
	return declustered

if frozpath == None:
	tmpdir = tempfile.mkdtemp('_chiffchaff')
else:
	tmpdir = "NONE"
print "================================================================"
print "Testing phase. (tmpdir is %s)" % tmpdir

mixestotest = froz_nmix  or  range(2, maxtestsetsize+1)
# NB: "runtypes" list actually determines which tests will be run (...and so how long it will take...)
if fewerruntypes:
	runtypes = ['ba',       'af', 'a',       'i']
else:
	runtypes = ['ba', 'ag', 'af', 'a', 'is', 'i', 'ip']

results = {nmix:{runtype:[] for runtype in runtypes} for nmix in mixestotest}

firstrun = True
for permuteoffset in (froz_permute or xrange(0, len(items), maxtestsetsize)):

	indices = range(len(items))
	permutedlist = indices[permuteoffset:] + indices[:permuteoffset]
	trainindices = permutedlist[maxtestsetsize:]
	testindices  = permutedlist[:maxtestsetsize]
	print "trainindices: ", trainindices
	print "testindices: ", testindices
	trainset = [items[i] for i in trainindices]
	testset  = [items[i] for i in testindices ]

	model = trainModel(trainset)
	model_fwise = trainModel(trainset, vecextramode='fwise')

	for nmix in mixestotest:
		mixset = testset[:nmix]
		mixedwav = "%s/mixedfile_%i_%i.wav" % (annotdir, permuteoffset, nmix)  # NB mixedwav used to be written to tempdir - now in annotdir so is longterm cached
		# calling a subprocess seems to inflate memory usage - avoid it in the sashmodes where the results aren't used
		# http://stackoverflow.com/questions/1367373/python-subprocess-popen-oserror-errno-12-cannot-allocate-memory
		if (frozpath == None) and (specgrammode != 'sash01') and (specgrammode != 'sash02'):
			if os.path.exists(mixedwav):
				print("Loading existing %s" % mixedwav)
			else:
				# sum together the audio from $nmix different wavs (different lengths no problem; though should we randomly offset?)
				if nmix==1:
					soxcmd = ['sox', '-m'] + [x['wavpath'] for x in mixset] + [x['wavpath'] for x in mixset] + [mixedwav]
				else:
					soxcmd = ['sox', '-m'] + [x['wavpath'] for x in mixset] + [mixedwav]
				print soxcmd
				call(soxcmd)

		truebouts = []
		for item in mixset:
			print "item %s -- bout lengths are %s -- timespans %s" % (item['basename'], \
				', '.join([str(len(bout)) for bout in item['bouts']]), \
				', '.join([  "[%.1f--%.1f]" %  ((min(peak['timepos'] for peak in bout),
				              max(peak['timepos'] for peak in bout)))       for bout in item['bouts']]))
			truebouts.extend(item['bouts'])
		activitycurve_t = calcactivitycurve(truebouts)
		if firstrun: plotactivitycurve(activitycurve_t, label="true_mixedfile_%i_%i" % (permuteoffset, nmix))

		# run chirplet analysis on the mixture
		(hopsecs, framesizesecs, unframed) = analysemixedwav(mixedwav, frozpath)
		numpeaksfromoriganalysis = len(unframed)
		if firstrun: plotunframed(framesizesecs, unframed, "mixedfile_%i_%i" % (permuteoffset, nmix))
		maxtimepos = max(map(lambda x: x['timepos'], unframed))

		# use all-NN to assign chirplets to their "ground-truth" source file
		# -- note that this relies on an assumption about how similar the recovered chirps are in the mix and the orig CSVs
		print ">allNN"
		candidateNNs = []
		sourcecolours = {-1: (0.85,0.85,0.85,0.5)}  # for diagnostic plotting
		for sourceindex, item in enumerate(mixset):
			sourcecolours[sourceindex] = rainbowcolour(sourceindex, len(mixset))
			for about in item['bouts']:
				for peak in about:
					peak['sourceindex'] = sourceindex
				candidateNNs.extend(about)
		for datum in unframed:
			datum['nn_dist'] = 9e99
			for candindex, cand in enumerate(candidateNNs):
				# NB scaling is manually estimated:
				# small diff in timepos is on the order of 0.05
				# small diff in freq is on the order of 100
				# freq is double-counted because there are 2 freqs, so if anything we'd prefer to over-weight the time
				dist = (((cand['from']   - datum['fromto'][0])/100.)**2) \
				     + (((cand['to']     - datum['fromto'][1])/100.)**2) \
				     + (((cand['timepos']- datum['timepos']  )/0.001)**2)
				if dist < datum['nn_dist']:
					datum['nn_dist'] = dist
					datum['nn_gtboutindex'] = cand['gtboutindex']
					datum['nn_gtsourceindex'] = cand['sourceindex']
					datum['datumindex'] = candindex
		# now each datum should have ['nn_gtboutindex'] which we can use for evaluation
		#  but some of them might be noise - we assume the nearest to any particular GT is the true, and the others are noise...
		nearest_dist_to_each_cand = {}
		for datum in unframed:
			if (datum['datumindex'] not in nearest_dist_to_each_cand) or \
					(nearest_dist_to_each_cand[datum['datumindex']] > datum['nn_dist']):
				nearest_dist_to_each_cand[datum['datumindex']] = datum['nn_dist']
		# having found the nearest distances for each candidate, we can now kick out any who are further
		for datum in unframed:
			if datum['nn_dist'] > nearest_dist_to_each_cand[datum['datumindex']]:
				datum['datumindex'] = -1
				datum['nn_gtboutindex'] = -1
				datum['nn_gtsourceindex'] = -1
		print "<allNN"
		thedistances = [sqrt(datum['nn_dist']) for datum in unframed]
		print "allNN distances: range [%g, %g], mean %g" % (min(thedistances), max(thedistances), float(sum(thedistances))/len(thedistances))

		# This little iteration may seem weird - storing a 'datumindex' inside data that actually ARE the groundtruth.
		# The reason is so we can treat groundtruth and audio-analysed symmetrically when we do evaluation.
		for gtindex, gtdatum in enumerate(candidateNNs):
			gtdatum['datumindex'] = gtindex

		if firstrun: plotunframed(framesizesecs, candidateNNs, "mixedfile_groundtruth_%i_%i" % (permuteoffset, nmix),
			colormapper=lambda x: sourcecolours[x['sourceindex']])

		#############################################################################
		# run MRP inference on the output
		actualsnr = float(len(candidateNNs))/max(1, numpeaksfromoriganalysis - len(candidateNNs))
		print "actual SNR is %g  (gt has %i peaks, analysis has %i peaks)" % (actualsnr, len(candidateNNs), numpeaksfromoriganalysis)
		clusters = chmodelMRPGraph_andgetclusters(unframed, model, 1)   # snr estimate fixed at reasonable default

		mrpcolours = {-1: (0.85,0.85,0.85,0.5)}  # for diagnostic plotting
		for clustindex in xrange(len(clusters['clusters'])):
			mrpcolours[clustindex] = rainbowcolour(clustindex, len(clusters['clusters']))

		activitycurve_e = calcactivitycurve([cl[1] for cl in clusters['clusters']])

		# plot, coloured in by FILE of origin -- i.e. groundtruth -- and also by estimated clustering
		declustered = decluster_remember_indices(clusters)
		if firstrun:
			print "Plotting sourcecoloured"
			plotunframed(framesizesecs, plottimelimit(declustered), label="sourcecolouredall", numsplit=4, \
				colormapper=lambda x: sourcecolours[x['nn_gtsourceindex']])
			plotunframed(framesizesecs, plottimelimit(filter(lambda x: x['clustid']!=-1, declustered)), label="sourcecoloured", numsplit=4, \
				colormapper=lambda x: sourcecolours[x['nn_gtsourceindex']])
			plotunframed(framesizesecs, plottimelimit(declustered), label="mrpcolouredall", numsplit=4, \
				colormapper=lambda x: mrpcolours[x['clustid']])

			plotactivitycurve(activitycurve_e, label="est_mixedfile_%i_%i" % (permuteoffset, nmix))

		print "Numbers of peaks: in unframed %i, in declustered %i, in clutter %i" % (len(unframed), len(declustered), len(clusters['other']))

		# compare the results of inference against the groundtruth
		print "Groundtruth   has %i bouts    (mean len %g), %i items" % \
			(len(truebouts), mean([len(x) for x in truebouts]), len(candidateNNs))
		print "Recovered set has %i clusters (mean len %g), %i items (plus %i clutter)" % \
			(len(clusters['clusters']), mean([len(x) for x in clusters['clusters']]), sum([len(ci) for ci in clusters['clusters']]), len(clusters['other']))
		print "num peaks from orig analysis: %i" % numpeaksfromoriganalysis

		# plot connected lines of clusters, on a gtboutindex-vs-time plot
		if firstrun:
			clusteredindicesforplot = [[{'timepos': hit['timepos'], 'gtboutindex_f': hit['nn_gtboutindex'] + ((clindex * 0.04) % 0.5), 'mag': hit['mag']} \
				for hit in cl[1]] for clindex, cl in enumerate(clusters['clusters'])]
			plot_gtindex_vs_time(clusteredindicesforplot, "mixedfile_%i_%i" % (permuteoffset, nmix))

		# Add the results to our collections
		results[nmix]['a' ].append(cluster_many_eval_stats(mixset, clusters, activitycurve_t, printindices=True))

		######################################################################################################
		# Now let's try other setups than the standard audio-analysis one (ideal-recovery case, baseline, etc)
		if 'af' in runtypes:
			print "======================================================"
			print "Checking fwise case..."
			clusters_af = chmodelMRPGraph_andgetclusters(unframed, model_fwise, 1, vecextramode='fwise')
			results[nmix]['af'].append(cluster_many_eval_stats(mixset, clusters_af, activitycurve_t, printindices=False))
		if 'ag' in runtypes:
			print "======================================================"
			print "Checking greedy case..."
			clusters_ag = chmodelMRPGraph_andgetclusters(unframed, model, 1, greedynotfull=True)
			results[nmix]['ag'].append(cluster_many_eval_stats(mixset, clusters_ag, activitycurve_t, printindices=False))
		if 'ba' in runtypes:
			print "======================================================"
			print "Checking baseline audio case..."
			clusters_ba = chmodel_baseline_andgetclusters(unframed, model)
			results[nmix]['ba'].append(cluster_many_eval_stats(mixset, clusters_ba, activitycurve_t, printindices=False))
		if ('ip' in runtypes) or ('i' in runtypes) or ('is' in runtypes):
			# this is needed for all 'i*' run types
			idealcasepeaks = []
			newboutindex = 0
			for source in mixset:
				for bout in source['bouts']:
					newboutindex += 1
					for peak in bout:
						peak = copy(peak)
						peak['nn_gtboutindex'] = newboutindex
						idealcasepeaks.append(peak)
			idealcasepeaks.sort(key=itemgetter('timepos'))
		if 'ip' in runtypes:
			# ideal-case analysis: use the "mixset"'s precalculated chirps rather than reanalysing audio - should upper-bound real performance
			print "======================================================"
			print "Checking ideal-recovery-and-peeking-training case..."
			peekingmodel = trainModel(mixset)
			clusters_ip = chmodelMRPGraph_andgetclusters(idealcasepeaks, peekingmodel, 200)
			results[nmix]['ip'].append(cluster_many_eval_stats(mixset, clusters_ip, activitycurve_t, printindices=False))
		if 'i' in runtypes:
			print "======================================================"
			print "Checking ideal-recovery case..."
			clusters_i = chmodelMRPGraph_andgetclusters(idealcasepeaks, model, 200)
			results[nmix]['i' ].append(cluster_many_eval_stats(mixset, clusters_i , activitycurve_t, printindices=False))
		if 'is' in runtypes:
			print "======================================================"
			print "Checking ideal-recovery-scramblenoise case..."
			# Construct "ideal-plus-scramble" dataset - ideal, plus a duplicate marked as clutter and with timeposses shuffled
			scrambled_ideal_peaks = deepcopy(idealcasepeaks)
			timerange = (scrambled_ideal_peaks[0]['timepos'], scrambled_ideal_peaks[-1]['timepos'])
			for i, peak in enumerate(scrambled_ideal_peaks):
				peak['nn_gtboutindex'] = -1
				peak['timepos'] = uniform(*timerange)
			peaks_is = idealcasepeaks + scrambled_ideal_peaks
			shuffle(peaks_is) # ensure order-of-presentation cannot bias results
			peaks_is.sort(key=itemgetter('timepos'))
			clusters_is = chmodelMRPGraph_andgetclusters(peaks_is, model, 1)
			results[nmix]['is'].append(cluster_many_eval_stats(mixset, clusters_is, activitycurve_t, printindices=False))

		if firstrun: 
			if 'i' in runtypes:
				plotunframed_rainbow(clusters_i , framesizesecs, "mrpcoloured_ideal")
			if 'ip' in runtypes:
				plotunframed_rainbow(clusters_ip, framesizesecs, "mrpcoloured_idealpeek")
			if 'is' in runtypes:
				plotunframed_rainbow(clusters_is, framesizesecs, "mrpcoloured_idealscramble")

		firstrun = False
		plt.close('all')  # does this help prevent memleaks with thousands of Bbox etc objects kept?
		print("gc...")
		print gc.collect()

statsfile = open("%s/chchstats%s.csv" % (annotdir, specgrammodecode), 'w')
statsfile.write('nmix,runtype,whichstat')
for i in xrange(len(results[mixestotest[0]]['a'])):
	statsfile.write(',val%i' % i)
statsfile.write("\n")

statstolist = ["Fsn", "Ftrans", "Fsigtrans"]
for nmix in mixestotest:
	print "-------------------------------------------"
	print "Overall results for nmix %d (%d-fold xv, %s)" % (nmix, len(results[nmix]['a']), analysistype)
	for runtype in runtypes:
		print "[%2s]   " % (runtype),
		# results[nmix][runtype] is a list of dictionaries
		for whichstat in statstolist:
			alist = [adict[whichstat] for adict in results[nmix][runtype]]
			print "%s: %-6s   " % (whichstat, "%.3g" % mean(alist)),
			statsfile.write("%i,%s,%s,%s\n" % (nmix, runtype, whichstat, ','.join(map(str, alist)) ))
		print
statsfile.close()
#shutil.rmtree(tmpdir)

endtime = time.time()
timetaken = datetime.timedelta(seconds=endtime-starttime)
print("Finished. Time taken: %s" % (str(timetaken)))

