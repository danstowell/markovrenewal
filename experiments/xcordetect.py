#!/bin/env python

"""
Cross-correlation detection for use with MRP.
By Dan Stowell August 2012.
"""

import numpy as np
import numpy.linalg as npla
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os.path
from scikits.audiolab import Sndfile
from scikits.audiolab import Format
from operator import itemgetter
import copy
import csv
from glob import iglob
import os, errno
import cPickle as pickle

import chiffchaffch_excerpted as chiffchaffch

default_specgrammode = None   # None or 'sash01' or 'sash02'

####################################################################################
# this stuff is about building a GMM and evaluting it to create the template

class GaussianComponent:
	"""Represents a single Gaussian component, 
	with a float weight, vector location, matrix covariance."""
	def __init__(self, weight, loc, cov):
		self.weight = float(weight)
		self.loc    = np.array(loc, dtype=float, ndmin=2)
		self.cov    = np.array(cov, dtype=float, ndmin=2)
		self.loc    = np.reshape(self.loc, (np.size(self.loc), 1)) # enforce column vec
		self.cov    = np.reshape(self.cov, (np.size(self.loc), np.size(self.loc))) # ensure shape matches loc shape
		# precalculated values for evaluating gaussian:
		k = len(self.loc)
		self.part1 = (2.0 * np.pi) ** (-k * 0.5)
		self.part2 = np.power(npla.det(self.cov), -0.5)
		self.invcov = np.linalg.inv(self.cov)

	def __str__(self):
		return "GaussianComponent(%g, %s, %s)" % (self.weight, str(self.loc), str(self.cov))

	def valueat(self, x):
		x = np.array(x, dtype=float)
		x = np.reshape(x, (np.size(self.loc), 1)) # enforce column vec
		dev = x - self.loc
		#print "dev is ", dev
		part3 = np.exp(-0.5 * np.dot(np.dot(dev.T, self.invcov), dev))
		return self.part1 * self.part2 * part3 * self.weight

# freqfrm was originally 4500 for a while, then 3100 while sashhacking...
tplfreqrange = (3100, 7500)  #(4500, 7500)   # overall freq range for the grid-evaluated gmm
tplknee = (0.12, 4900)    # the location of the beginning of the hoizontal bar in the template
chirprange = (-160000.0, 160000.0)  # the range of chirp values considered
chirpstep = 4000  # the quantisation of chirp values

def makeGMM(numsper = [20, 20], chirped=False):
	"chirp=False (default) gives 2D GMM, otherwise 3D."
	comps = []
	eachweight = 1. / sum(numsper)
	# freq std was 100 for both, and timestd 0.01
	for freqfrm,     freqtoo,    timefrm,    timetoo, numper,     freqstd, timestd, chirpmean, chirpstd in [
	    [7000,       5000,       0.085,      0.11,    numsper[0], 100,     0.01,    -80000,    24000], 
	    [tplknee[1], tplknee[1], tplknee[0], 0.19,    numsper[1], 100,     0.01,    0,         16000]]:  # to check: flat chirp std ok? was 4k. 20k (1/4 of slopesslope) may be more forgiving?
		for i in xrange(numper):
			b = float(i)/numper
			a = 1. - b
			freq = freqfrm * a + freqtoo * b
			time = timefrm * a + timetoo * b
			if chirped:
				g = GaussianComponent(eachweight, [freq, time, chirpmean], [[freqstd ** 2, 0, 0], [0, timestd ** 2, 0], [0, 0, chirpstd ** 2]])
			else:
				g = GaussianComponent(eachweight, [freq, time           ], [[freqstd ** 2, 0],    [0, timestd ** 2]])
			#print g
			comps.append(g)
	return comps

def gridevalGMM(gmm, timefrm=0.05, timetoo=0.2, freqstep=None, timestep=None, norm=True, chirped=False):
	freqstep = freqstep or (fs/fftsize)
	timestep = timestep or (hop*framelen/fs)
	freqs = np.arange(tplfreqrange[0], tplfreqrange[1], freqstep)
	times = np.arange(timefrm, timetoo, timestep)
	chirpvals = np.arange(chirprange[0], chirprange[1]+1, chirpstep)

	if chirped:
		vals = np.zeros((len(freqs), len(times), len(chirpvals)))
	else:
		vals = np.zeros((len(freqs), len(times)))
	tot = 0.
	if chirped:
		for fi, freq in enumerate(freqs):
			for ti, time in enumerate(times):
				for ci, chirpval in enumerate(chirpvals):
					for g in gmm:
						val= g.valueat([freq, time, chirpval])
						tot += val
						vals[fi,ti,ci] += val
	else:
		for fi, freq in enumerate(freqs):
			for ti, time in enumerate(times):
				for g in gmm:
					val= g.valueat([freq, time])
					tot += val
					vals[fi,ti] += val
	if norm:
		vals *= (1./tot)
	return (vals, freqs, times)

def chirpval_to_bin(chirpval):
	return max(0, int(round(((
			min(chirprange[1]-(chirpstep/2), chirpval)
		 - chirprange[0]) / chirpstep))))

def template_get_chirpedbins_for_seg(templategrid, specseg_chirpbins):
	"Given a 3D template grid and a 2D segment of chirp BIN values, returns the appropriate 2D reduction of the template"
	try:
		#this line was an attempted efficiency improvement, not sure if it's sane OR fast...		return templategrid[:, :, specseg_chirpbins]
		return np.array([[templategrid[xi, yi, chirpbin] \
			for yi, chirpbin in enumerate(chirprow)] \
			for xi, chirprow in enumerate(specseg_chirpbins)])
	except:
		print "np.shape(templategrid):"
		print np.shape(templategrid)
		print "np.shape(specseg_chirpvals):"
		print np.shape(specseg_chirpvals)
		print "xi:"
		print xi
		print "yi:"
		print yi
		print "chirpval:"
		print chirpval
		print "indices into templategrid:"
		print [xi, yi, chirpval_to_bin(chirpval)]
		raise

#######################################################
# signal analysis

fs=44100.0
#hop=0.125
#framelen=1024
hop=0.5
framelen=512
fftsize=framelen          # !!! CHECK this is the size you want -- for sash02 data I need 1039, for 'normal' I just use framelen. (Code below throws error if mismatched.)
specfreqrange=(1000, 8500)   #(3000, 8000)
specfreqbinrange = [int(float(f * fftsize)/fs) for f in specfreqrange]
print "specfreqbinrange: ", specfreqbinrange
bintofreq = fs/fftsize
bintotime = (framelen * hop)/fs

def stft(x):
    hopsamp = int(hop*framelen)
    w = np.hamming(framelen)
    res = np.array([np.fft.fft(w*x[i:i+framelen]) 
                     for i in xrange(0, len(x)-framelen, hopsamp)])
    return res

def grok_sash_datum(datum):
	if datum=='NaN': return 1e-12
	else: return max(1e-12, abs(float(datum)))

def file_to_specgram(path, specgrammode=None):
	if specgrammode==None: # default is to do a "normal" spectrogram right here
		if fftsize != framelen: raise ValueError("this mode requires normal fftsize")
		if not os.path.isfile(path):
			raise ValueError("path %s not found" % path)
		sf = Sndfile(path, "r")
		if sf.channels != 1:
			raise Error("ERROR in xcordetect: sound file has multiple channels (%i) - mono audio required." % sf.channels)
		if sf.samplerate != fs:
			raise Error("ERROR in xcordetect: wanted srate %g - got %g." % (fs, sf.samplerate))
		chunksize = 4096
		pcm = np.array([])
		while(True):
			try:
				chunk = sf.read_frames(chunksize, dtype=np.float32)
				pcm = np.hstack((pcm, chunk))
			except RuntimeError:
				break
		spec = stft(pcm).T
	elif specgrammode=='sash01':
		if fftsize == framelen: raise ValueError("this mode requires abnormal fftsize")
		sashbasepath = os.path.expanduser("~/Documents/Dropbox/bird")
		sashfpath = "%s/%s_full_reass_spect_ddm_dgr_2_N_512_hop_256_overlap_50_fft_1039_tresh_0db.csv" % (sashbasepath, os.path.basename(path).rsplit('.', 1)[0])
		rdr = csv.reader(open(sashfpath, 'rb'))
		spec = []
		for row in rdr:
			row = map(grok_sash_datum, row)
			spec.append(row)
		spec = np.array(spec).T
		spec = spec / np.max(spec) # normalise
	elif specgrammode=='sash02':
		if fftsize == framelen: raise ValueError("this mode requires abnormal fftsize")
		sashbasepath = os.path.expanduser("~/Documents/Dropbox/bird")
		sashfpath = "%s/%s_full_reass_%%s_ddm_dgr_2_N_512_hop_256_overlap_50_fft_%%i_tresh_0db.csv" % (sashbasepath, os.path.basename(path).rsplit('.', 1)[0])
		# load mag info:
		rdr = csv.reader(open(sashfpath % ('spect', 1039), 'rb'))
		spec = []
		for row in rdr:
			row = map(grok_sash_datum, row)
			spec.append(row)
		spec = np.array(spec).T
		spec = spec / np.max(spec) # normalise
		# load chirp info:
		rdr = csv.reader(open(sashfpath % ('fm1', 1039), 'rb'))
		fm1 = []
		for row in rdr:
			row = map(float, row)  # do not use grok_sash_datum!
			fm1.append(row)
		fm1 = np.array(fm1).T
		if np.shape(fm1) != np.shape(spec):
			print sashfpath
			raise ValueError("mismatched array shapes for magnitudes and chirpvals: %s, %s" % (str(np.shape(fm1)), str(np.shape(spec))))
	else:
		raise ValueError("specgrammode not recognised: %s" % specgrammode)
	spec = spec[specfreqbinrange[0]:specfreqbinrange[1],:]
	spec = abs(spec)
	if specgrammode==None:
		spec = np.log(spec)
	if specgrammode=='sash02':
		fm1 = fm1[specfreqbinrange[0]:specfreqbinrange[1],:]
		return (spec, fm1)
	else:
		return spec

def freq_of_xcorspecgrambin(index):
	return index*bintofreq + specfreqrange[0]

def xcor2d(template, spec, chirpspec=None):
	"if chirpspec is not None, it should be same shape as 'spec' but holding chirprates (in Hz/s)"
	specmin = min(map(min,spec))
	specmax = max(map(max,spec))
	spec = (spec - specmin)/(specmax-specmin)
	fsize = len(template)
	tsize = len(template[0])
	if chirpspec==None:
		return np.array([[
			xcor2d_domultiply(template, xcor2d_getseg(template, spec, foff, toff))
				for toff in xrange(len(spec[0])-tsize)] for foff in xrange(len(spec)-fsize)])
	else:
		# Here we pre-convert chirp vals to bin indices
		chirpbinspec = np.array([[chirpval_to_bin(x) for x in csrow] for csrow in chirpspec])
		return np.array([[
			xcor2d_domultiply(
				template_get_chirpedbins_for_seg(template, xcor2d_getseg(template, chirpbinspec, foff, toff)), 
				xcor2d_getseg(template, spec, foff, toff))
				for toff in xrange(len(spec[0])-tsize)] for foff in xrange(len(spec)-fsize)])

def xcor2d_getseg(template, spec, foff, toff, binsbelow=0):
	"Given a template, a (larger) specgram, and INTEGER freq and time offsets, gets the correct specgram lump to match against the template"
	return spec[foff-binsbelow:foff+len(template), toff:toff+len(template[0])]

def xcor2d_domultiply(template, specseg):
	#over_segmax = 1. / max(map(max, specseg)) # normalising within each seg might be ok, seems to give noisier result though
	#return sum(sum(template * specseg))
	#return sum((template * specseg).flat)
	# Note: profiling finds that np.correlate was faster than sumsum and sumflat.
	return np.correlate(template.flat, specseg.flat)[0]

def findpeaks(arr):
	"Just returns 1 where it's a local peak and 0 where it's not"
	ret = np.zeros(np.shape(arr),dtype=float)
	for x, row in enumerate(arr):
		for y, datum in enumerate(row):
			if (x==0 or datum > arr[x-1,y]) \
				and (y==0 or datum > arr[x,y-1]) \
				and (x==(len(arr)-1) or datum > arr[x+1,y]) \
				and (y==(len(arr[0])-1) or datum > arr[x,y+1]):
				ret[x,y] = 1
	return ret

def xco2peaks(xco, ampthresh=0.2, mingap=0.0, timeoffset=0, freqoffset=0):
	"Provide a 2D array; returns a filtered list of peaks, each of the form {timepos, freq, mag}"
	xcomax = max(map(max, xco))
	absthresh = xcomax * ampthresh
	xco_peaks = findpeaks(xco) * xco
	peaks = []
	for foff, row in enumerate(xco_peaks):
		for toff, datum in enumerate(row):
			if datum > absthresh:
				# NB: the small additions to time and freq here are to move the location from the "bottom-left" of the template 
				#  to a more meaningful location, here the location of the template's "elbow".
				peaks.append({'timepos': toff*bintotime + timeoffset, 'freq': freq_of_xcorspecgrambin(foff) + tplknee[1] - freqoffset, 'mag': datum, 'foff': foff, 'toff': toff, 'salience':datum})
	peaks.sort(key=itemgetter('timepos'))
	(peaks, discarded) = chiffchaffch.filter_mingap(peaks, mingap)
	return (peaks, discarded)

########################################################################################################
# functions related to framewise peaks-per-bin (details extractable after you've picked your xcor peaks)

def onesyll_getframewisepeakbins(syll, syllid, templategrid, spec, ignorefirstsecs=0.03, ignorelastsecs=0.2, binsbelow=16):
	"""Once you've got some peaks (using xco2peaks), this function can be used on each one to re-inspect the spectrogram
	and find, for each relevant frame, what the freq of the peak bin is.
	It WRITES ITS RESULTS INTO THE 'fwise' DICT ENTRY rather than returning them."""
	ignorebinsbefore = ignorefirstsecs / bintotime
	ignorebinsafter  = ignorelastsecs  / bintotime
	ret = []
	binsbelow = min(binsbelow, syll['foff'])
	for fwisepeak in xcor2d_findpeakforeachtimeslice(templategrid, xcor2d_getseg(templategrid, spec, syll['foff'], syll['toff'], binsbelow)):
		if (fwisepeak['toff'] >= ignorebinsbefore) and (fwisepeak['toff'] <= ignorebinsafter):
			ret.append({'freq': freq_of_xcorspecgrambin(syll['foff']+fwisepeak['foff']-binsbelow),
				    'time': bintotime            * (syll['toff']+fwisepeak['toff']),
				    'mag' : fwisepeak['mag'],
				    'syllid': syllid})
	syll['fwise'] = ret

def xcor2d_findpeakforeachtimeslice(templategrid, specseg):
	"Once you've found a general xcor2d match, use this to find out exactly which bins were the strongest in their time-slice"
	#prod = (templategrid * specseg).T
	prod = specseg.T    # NOT weighted by template, just inspecting the whole rectangle, since tweet shape can deviate from template a lot
	peakbins = [{'foff':np.argmax(aslice), 'toff':index, 'mag':np.max(aslice)} for index, aslice in enumerate(prod)]
	return peakbins

def write_csv_fwisebins(csvpath, peaks, grid, spec):
	"Write a CSV file of the detailed framewise data, the 'fine detail' of each detection"
	fp = open(csvpath, 'w')
	fp.write("syllid,time,freq,mag\n")

	for syllid, syll in enumerate(peaks):
		for datum in syll['fwise']:
			fp.write("%i,%g,%g,%g\n" % (datum['syllid'], datum['time'], datum['freq'], datum['mag']))
	fp.close()

def add_fwisebin_data_to_csvdata(maindata, fwisefpath, keytoadd='fwise'):
	"IN-PLACE operation - reads a CSV file and modifies the corresponding entries in maindata adding an 'fwise' subsection"
	rdr = csv.DictReader(open(fwisefpath, 'rb'))
	clumped = {}
	for row in rdr:
		syllid = int(row['syllid'])
		row = {key:float(row[key]) for key in row}
		if syllid not in clumped: clumped[syllid] = []
		clumped[syllid].append(row)
	if len(clumped) != len(maindata):
		raise ValueError("len(clumped) != len(maindata) ... %i != %i " % (len(clumped), len(maindata)))
	for i, aclumped in clumped.iteritems():
		maindata[i][keytoadd] = aclumped

def fwisebindata_as_vector(fwise):
	"""Converts fwise data to a standard vector format for use in GMMs etc."""
	indices = [0, 3, 7, 11, 15]
	vec = [fwise[index]['freq'] for index in indices] # shortening
	#vec = [datum['freq'] for datum in fwise]  # notshortening
	return vec

############################################
# the functions acting like chiffchaffch.py:

def xcor_likechrm_one(gridinfo, basepath, wavpath, filteramp=True, filtermingap=True, syllgapsecs=0.2, plot=False, ampthresh=0.2, plotlimitsecs=99999, specgrammode=None):
	"Returns data in same format as chiffchaffch."
	(grid, freqs, times) = gridinfo
	spec = file_to_specgram(wavpath, specgrammode=specgrammode)
	if specgrammode=='sash02':
		(spec, chirpspec) = spec  # unpack added chirpdata
	specmin = min(map(min,spec))
	specmax = max(map(max,spec))
	print "specmax, specmin: ", (specmax, specmin)
	spec = (spec - specmin)/(specmax-specmin)

	if specgrammode=='sash02':
		xco = xcor2d(grid, spec, chirpspec)
	else:
		xco = xcor2d(grid, spec)

	if (specgrammode=='sash01') or (specgrammode=='sash02'):
		####################################################
		# debug: plotting sash's specgram
		plotmaxdur = 2.5
		sashbasepath = os.path.expanduser("~/Documents/Dropbox/bird")
		examplefontsize='large'
		#### for straight spectral data, we can just image it
		for (thearray, fnamepostfix, doyticks) in [
					(       (spec * (specmax-specmin) + specmin), '',     True), 
					(xco,                                        '_xco', False),
					]:
			plt.figure()
			#plt.imshow(xco, origin='lower', aspect='auto', interpolation='nearest')   #, cmap=cm.binary)
			plt.imshow(thearray, origin='lower', aspect='auto', interpolation='nearest', cmap=cm.binary)
			plt.xlabel("Time", fontsize=examplefontsize)
			plt.ylabel("Freq", fontsize=examplefontsize)
			plt.xlim(0, plotmaxdur / bintotime)
			plt.xticks([0, plotmaxdur / bintotime], [0, plotmaxdur])
			if doyticks:
				plt.yticks([0, np.shape(spec)[0]-1], specfreqrange)
			plt.savefig("%s/replotted/%s%s.pdf" % (sashbasepath, os.path.basename(wavpath).rsplit('.', 1)[0], fnamepostfix), papertype='A4', format='pdf')
			plt.clf() # does this help prevent memory leaks with thousands of Bboxes?
		if specgrammode=='sash02':
			#### for chirp data, let's just pick the peak in each frame and plot it
			plt.figure()
			chirpstoplot = []   # timestart, timeend, freqstart, freqend, mag
			print "Plotting loaded sash chirp data - %i frames to plot" % len(spec.T)
			print "np.shape(spec):"
			print np.shape(spec)
			print "np.shape(chirpspec):"
			print np.shape(chirpspec)
			for whichframe, aframe in enumerate(spec.T):
				whichmax = np.argmax(aframe)
				thebinfreq = freq_of_xcorspecgrambin(whichmax)  # NB adds lower bound back on, since spec is trimmed
				chirpdelta = chirpspec[whichmax,whichframe] * (float(framelen)/fs) * 0.5 * 0.5
				if (aframe[whichmax] > 0.0) and (thebinfreq > 0.0):
					chirpstoplot.append({'timestart': whichframe * bintotime, 'timeend': whichframe * bintotime + (float(framelen)/fs) * 0.5,
						'freqstart': thebinfreq - chirpdelta, 'freqend': thebinfreq + chirpdelta, 'mag': aframe[whichmax], 
						'midfreq': thebinfreq, 'whichframe_index': whichframe, 'whichbin_index': whichmax})
			maxplotmag = max(x['mag'] for x in chirpstoplot[:200])
			for peak in chirpstoplot:
				peak['alpha'] = 1# - (peak['mag'] / maxplotmag)
				plt.plot([peak['timestart'], peak['timeend']], \
					 [peak['freqstart'], peak['freqend']], \
					alpha=peak['alpha'], color=[0,0,0])
			plt.xlim(0, plotmaxdur)
			plt.ylim(*specfreqrange)
			plt.xlabel("Time", fontsize=examplefontsize)
			plt.ylabel("Freq", fontsize=examplefontsize)
			plt.savefig("%s/replotted/%s_peakchirp.pdf" % (sashbasepath, os.path.basename(wavpath).rsplit('.', 1)[0]), papertype='A4', format='pdf')
			plt.clf() # must do this - helps prevent memory leaks with thousands of Bboxes
		####################################################

	if filteramp:
		passampthresh = ampthresh
	else:
		passampthresh = -1.
	if filtermingap:
		passmingap = syllgapsecs
	else:
		passmingap = 0.0

	peaks, discardedpeaks = xco2peaks(xco, ampthresh=passampthresh, mingap=passmingap, timeoffset=times[0], freqoffset=freqs[0])

	print "xcor produced %i peaks (and %i discarded)" % (len(peaks), len(discardedpeaks))

	# NB: since "syllid" is re-used for the kept and the discarded, it does not uniquely identify a syll if you combine them.
	# It corresponds to index in the separated lists
	for syllid, peak in enumerate(peaks):
		peak['fromto'] = (peak['freq']+2000, peak['freq'])
		onesyll_getframewisepeakbins(peak, syllid, grid, spec)
	for syllid, peak in enumerate(discardedpeaks):
		peak['fromto'] = (peak['freq']+2000, peak['freq'])
		onesyll_getframewisepeakbins(peak, syllid, grid, spec)

	# Now write out a CSV file in the 'chirp-like' format
	subfolder = 'xcor'
	filename = os.path.splitext(os.path.split(wavpath)[1])[0]
	csvpath_base = "%s/%s/%s" % (basepath, subfolder, filename)
	if specgrammode==None:
		specgrammodecode = ''
	else:
		specgrammodecode = '_%s' % specgrammode
	chiffchaffch.write_the_csv("%s%s.csv"          % (csvpath_base, specgrammodecode), peaks)
	chiffchaffch.write_the_csv("%s%s_noise.csv"    % (csvpath_base, specgrammodecode), discardedpeaks)
	# Here we write out auxiliary CSV whose purpose is to be freq-time locations for sash to use as reassignment starting-points
	write_csv_fwisebins("%s%s.fwisebins.csv"       % (csvpath_base, specgrammodecode), peaks, grid, spec)
	write_csv_fwisebins("%s%s_noise.fwisebins.csv" % (csvpath_base, specgrammodecode), discardedpeaks, grid, spec)

	if plot:
		tmp_maxamp = max(peak['mag'] for peak in peaks)
		tmp_framesizesecs = 0.01   # dummy value
		chiffchaffch.plot_kept_and_discarded(peaks, discardedpeaks, tmp_maxamp, tmp_framesizesecs, "%s/%s/pdf/%s%s.xcor.pdf" % (basepath, subfolder, filename, specgrammodecode), plotlimitsecs)
		plt.clf() # does this help prevent memory leaks with thousands of Bboxes?

	return (peaks, discardedpeaks)

def xcor_likechrm_batch(basepath, filteramp=True, filtermingap=True, syllgapsecs=0.2, doplots=True, ampthresh=0.05, plotlimitsecs=99999, specgrammode=None):
	"Analyses a batch of audio files"
	pattern = "%s/wav/XC*.wav" % basepath
#	pattern = "%s/wav/XC46524*.wav" % basepath   # use this line for quick test on only one
	print pattern
	gridinfo = get_gridinfoGMM_cacheable(basepath, specgrammode)
	for wavpath in iglob(pattern):
		print wavpath
		xcor_likechrm_one(gridinfo, basepath, wavpath, filteramp=filteramp, filtermingap=filtermingap, syllgapsecs=syllgapsecs, \
			plot=doplots, ampthresh=ampthresh, plotlimitsecs=plotlimitsecs, specgrammode=specgrammode)

def get_gridinfoGMM_cacheable(basepath, specgrammode):
	"If a cached template matching our settings is available, loads it; otherwise, builds it"
	chirped = (specgrammode=='sash02')
	picklepath = "%s/xcor/templategrid_fft%i_frange%i-%i_ch%s.pickle" % (basepath, fftsize, tplfreqrange[0], tplfreqrange[1], str(chirped))

	if os.path.isfile(picklepath):
		print "Reloading cached template from %s" % picklepath
		gridinfo = pickle.load(open(picklepath, 'rb'))
	else:
		print "Building template (chirped=%s)" % str(chirped)
		gmm = makeGMM(chirped=chirped)
		gridinfo = gridevalGMM(gmm, chirped=chirped)
		pickle.dump(gridinfo, open(picklepath, 'wb'), -1)
		print "Written cached template to %s" % picklepath
	return gridinfo

################################################################################################
# Ensure we have folders to write our results to
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST:
            pass
        else: raise

if __name__ == '__main__':
	specgrammode=default_specgrammode
	chirped = (specgrammode=='sash02')

	print "Building template (non-chirped) to plot it"
	gmm = makeGMM()
	(grid, freqs, times) = gridevalGMM(gmm, freqstep=fs/fftsize, timestep=hop*framelen/fs)
	plt.figure()
	plt.imshow(grid, origin='lower', aspect='auto', interpolation='nearest', cmap=cm.binary)
	ticksubsample = 10
	examplefontsize='large'
	plt.xticks(range(len(times))[::ticksubsample], [round(t, 2) for t in times[::ticksubsample]], fontsize=examplefontsize)
	plt.yticks(range(len(freqs))[::ticksubsample], [int(round(f, -2)) for f in freqs[::ticksubsample]], fontsize=examplefontsize)
	plt.xlabel("Time (s)", fontsize=examplefontsize)
	plt.ylabel("Freq (Hz)", fontsize=examplefontsize)
	#plt.title("Spectro-temporal template", fontsize='x-small')
	plt.savefig("output/pdf/plot_xcor_grid.pdf", papertype='A4', format='pdf')
	plt.clf() # must do this - helps prevent memory leaks with thousands of Bboxes
	print "Finished plotting grid. Press enter to continue, and analyse the dataset."
	raw_input()

	"""
	print "Loading example file"
	spec = file_to_specgram("/home/dan/birdsong/xenocanto/chiffchaff_XC48101_for_devt/XC48101.mp3.wav")
	print "specgram shape: %s" % (str(np.shape(spec)))
	spec = spec[:,:1200]  ########### TEMPORARY

	specmin = max(map(min,spec))
	specmax = max(map(max,spec))
	#print (specmin, specmax)
	spec = (spec - specmin)/(specmax-specmin)

	specmerge = copy.deepcopy(spec)
	scaleup = max(map(max,spec)) / max(map(max,grid))
	for rowi, row in enumerate(grid):
		for coli, datum in enumerate(row):
			if rowi < len(specmerge) and coli < len(specmerge[0]):
				specmerge[rowi, coli] += datum * scaleup
	plt.figure()
	plt.imshow(specmerge, origin='lower', aspect='auto', interpolation='nearest', cmap=cm.binary)
	plt.title("Spectrogram of example, with template added", fontsize='xx-small')
	plt.savefig("plot_xcor_example.pdf", papertype='A4', format='pdf')

	print "Doing cross-correlation analysis"
	xco = xcor2d(grid, spec)
	xcomax = max(map(max, xco))
	plt.figure()
	plt.imshow(xco, origin='lower', aspect='auto', cmap=cm.binary)
	plt.title("xco", fontsize='xx-small')
	plt.savefig("plot_xcor_xco.pdf", papertype='A4', format='pdf')
	xco_peaks = findpeaks(xco) * xco * (xco > (xcomax * 0.2))
	plt.figure()
	plt.imshow(xco_peaks, origin='lower', aspect='auto', cmap=cm.binary)
	plt.title("xco_peaks", fontsize='xx-small')
	plt.savefig("plot_xcor_xco_peaks.pdf", papertype='A4', format='pdf')

	peaks, discarded = xco2peaks(xco)
	#for peak in peaks: print peak

	peaksasarr = np.zeros(np.shape(xco_peaks))
	for peak in peaks:
		peaksasarr[peak['fbin'], peak['tbin']] = peak['mag']

	plt.figure()
	plt.imshow(peaksasarr, origin='lower', aspect='auto', cmap=cm.binary)
	plt.title("xco_peaks_filt", fontsize='xx-small')
	plt.savefig("plot_xcor_xco_peaks_filt.pdf", papertype='A4', format='pdf')
	"""

	############################################################################
	################# Now let's do the XC chch data

	basepath = os.path.expanduser("~/birdsong/xenocanto_chiffchaff")
	usetrimmed = False # True
	if usetrimmed: basepath += "/trimmed"

	mkdir_p("%s/xcor/pdf" % basepath)

	# ampthresh 0.8 was good before taking log away, 0.1 or 0.2 was nonlog thresh for sash01, but 0.2 too restrictive for sash02 trying 0.1
	ampthresh = {None: 0.8, 'sash01': 0.2, 'sash02': 0.05}[specgrammode]
	xcor_likechrm_batch(basepath=basepath, filteramp=True, filtermingap=True, syllgapsecs=chiffchaffch.syllgapsecs, doplots=True, ampthresh=ampthresh, plotlimitsecs=25,
		specgrammode=specgrammode
		)

	## Here's how to analyse just one:
	#gmm = makeGMM()
	#gridinfo = gridevalGMM(gmm)
	#xcor_likechrm_one(gridinfo, basepath=basepath, wavpath="/home/dan/birdsong/xenocanto_chiffchaff/xcor/froz/mix_XC46524-nl_XC35097-es.wav", filteramp=True, filtermingap=False, syllgapsecs=chiffchaffch.syllgapsecs, plot=True, ampthresh=0.8, plotlimitsecs=30)

