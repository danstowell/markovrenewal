#!/usr/bin/env python

# takes a set of audio files, trims them to the length of the shortest, using the largest-power chunk from each file.

import os, os.path, errno
from glob import iglob, glob
from scikits.audiolab import Sndfile
from scikits.audiolab import Format
import numpy as np
from math import floor

datadir = os.path.expanduser("~/birdsong/xenocanto_more_chiffchaff")
wavdir = datadir+"/wav"
outdir = datadir+"/trimmed/wav"

fs = 44100.0

####################################################################
# load all audios   (error if srate mismatch etc)
wavpaths = glob(wavdir + "/XC*.wav")
wavaudio = {}
fmt = None
for wavpath in wavpaths:
	print wavpath
	if not os.path.isfile(wavpath):
		raise ValueError("path %s not found" % wavpath)
	sf = Sndfile(wavpath, "r")
	if sf.channels != 1:
		raise Error("ERROR: sound file has multiple channels (%i) - mono audio required." % sf.channels)
	if sf.samplerate != fs:
		raise Error("ERROR: wanted srate %g - got %g." % (fs, sf.samplerate))
	if not(fmt):
		fmt = sf.format
	chunksize = 4096
	pcm = np.array([])
	while(True):
		try:
			chunk = sf.read_frames(chunksize, dtype=np.float32)
			pcm = np.hstack((pcm, chunk))
		except RuntimeError:
			break
	wavaudio[os.path.basename(wavpath)] = pcm
	sf.close()
#print wavaudio.keys()

# determine shortest duration
tgtdur = min([len(data) for data in wavaudio.values()])
print "Will use %i samples from each file" % tgtdur

# foreach file, determine strongest segment (use hopping window), write it out as audio, make note of location in editlist.txt
for (filename, data) in wavaudio.items():
	print "Studying file %s" % filename
	bestpower = 0
	bestoffset = 0
	framelen = 4096

	framepowers = [sum(data[offset:offset+framelen] ** 2) for offset in range(0, len(data), framelen)]

	tgtdur_frames = int(floor(float(tgtdur)/framelen))

	for frmoffset in range(0, len(framepowers)-tgtdur_frames):
		power = sum(framepowers[frmoffset:frmoffset+tgtdur_frames])
		if power > bestpower:
			bestpower  = power
			bestoffset = frmoffset * framelen
	print "   selected offset %i, power %g" % (bestoffset, bestpower)
	outpath = "%s/%s" % (outdir, filename)
	print "Will write %s" % outpath
	outsf = Sndfile(outpath, "w", fmt, 1, fs)
	outsf.write_frames(data[bestoffset:bestoffset+tgtdur])
	outsf.close()

