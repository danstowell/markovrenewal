#!/bin/env python

# synthetic test of MMRP inference, using a simple discrete state and continuous time with log-gaussian transitions
# (c) 2013 Dan Stowell and Queen Mary University of London

from math import sqrt, log, exp, pi, erf
from operator import itemgetter
import random
import itertools
import copy
import os.path
import gc
import copy

import matplotlib.pyplot as plt

from markovrenewal import MRPGraph
import evaluators
import timer

myfloat = float

########################################################################################
class Generator:
	"""Represents a scenario in which multiple Markov Renewal Processes generate observations.
	The MRPs have common params: a fixed alphabet size, and a particular transition table, generated parametrically here.
	It can then generate observation sequences with clutter noise etc.
	"""
	def __init__(self, alphabetsize=10, hopdurdiversity=1, ttsparsity=0.5):
		self.alphabetsize = alphabetsize
		self.logmeanhopdur = log(1.0)  # the "parent" mean used for selecting the means in the TT entries
		self.hopdurdiversity = hopdurdiversity  # the stdev (in log domain) used for generating means in the TT entries
		self.ttsparsity = ttsparsity
		outarcseach = int(ttsparsity * alphabetsize)  # the number of outarcs that will be generated for each state
		# create the TT
		outarcs = []
		for fromstate in range(alphabetsize):
			# randomly choose which outarcs we will create
			tostates = range(alphabetsize)
			random.shuffle(tostates)
			tostates = tostates[:outarcseach]
			(outprobs, outprobs_cumul) = random_multinomial(len(tostates))
			# sample logmean hopsize for each, sampled from a gaussian on the parent params
			logmeanhop = [random.gauss(self.logmeanhopdur, self.hopdurdiversity) for _ in range(len(tostates))]
			logstdhop  = [0.1 for _ in range(len(tostates))]  # LATER look at different settings
			# NB do not sort the "outprobs" etc etc -- the cumulative proby etc all need to be in the same order
			outarcs.append([
				{'to':tostates[i], 'outprob':outprobs[i], 'outprob_cumul':outprobs_cumul[i], 'logmeanhop': logmeanhop[i], 'logstdhop': logstdhop[i], 'inv_logstdhop': 1./logstdhop[i]}
				for i in range(len(tostates))])
		self.tt = outarcs
		self.cache_trans_likelihood = {}

	def generate(self, snr=0, noisecorr=0, birthdens=1, deathprob=0.1, obsdur=40.):
		"Generate a set of signal and noise observations"
		noiseratio = snrdb_to_ratio(snr)

		# birth density is births-per-second, so we expect to see (obsdur/birthdens) births. sampling from poisson uses exponential time gaps.
		birthposses = []
		curpos = 0.0
		while True:
			curpos += random.expovariate(birthdens)
			if curpos > obsdur:
				break
			birthposses.append(curpos)
		print "G.generate(): generated %i births (expected val %g)" % (len(birthposses), (obsdur * birthdens))

		polyphony = 0.
		obsns = []
		# foreach birth, generate a sequence of events with timestamps, gt_ids
		for gtid, birthpos in enumerate(birthposses):

			curstate = random.randrange(self.alphabetsize)  # NB here we have uniform birth state proby, for simplicity
			curpos   = birthpos
			obsseq = []
			while True:
				obsseq.append({'timepos':curpos, 'state':curstate, 'gtid':gtid})
				#print obsseq[-1]
				if random.random() < deathprob: # maybe die
					finalpos = curpos
					break
				# update to next state by sampling from TT
				tt_from = self.tt[curstate]
				tt_index_f = random.random()
				tt_index_i = 0
				while tt_from[tt_index_i]['outprob_cumul'] < tt_index_f:
					tt_index_i += 1
				tt_to = tt_from[tt_index_i]
				curstate = tt_to['to']
				curpos += exp(random.gauss(tt_to['logmeanhop'], tt_to['logstdhop']))

				if curpos > obsdur: # maybe fall off end
					finalpos = obsdur
					break
			# build up the mean polyphony count
			polyphony += (finalpos - birthpos) / obsdur
			obsns.extend(obsseq)
		print "  G.generate(): generated %i emissions (mean polyphony %g)" % (len(obsns), polyphony)

		# now generate noise (with gtid -1)
		numnoise = int(len(obsns) * noiseratio)
		noiseobsns = []
		numnoise_correlated = 0
		for _ in range(numnoise):
			curpos = random.uniform(0, obsdur)
			curstate = random.randrange(self.alphabetsize)
			# implement time and state correlation by picking a random obsn and imposing correlation with it
			corr = False  # 'corr' just tmp
			if noisecorr != 0:
				if random.random() < noisecorr:
					correlator = random.choice(obsns)
					curstate = correlator['state']
					curpos   = correlator['timepos'] # + random.gauss(0,0.1)
					numnoise_correlated += 1
					corr = True
			noiseobsns.append({'timepos':curpos, 'state':curstate, 'gtid':-1, 'corr': corr})  # 'corr' just tmp
		print "  G.generate(): plus %i noise emissions (SNR as ratio: %g)" % (len(noiseobsns), noiseratio)
		if len(noiseobsns) != 0:
			print "      correlated %i of %i noise obsns (%g %%)" % (numnoise_correlated, len(noiseobsns), 100. * numnoise_correlated / len(noiseobsns))

		# finally combine all the obsns and sort by time
		obsns.extend(noiseobsns)
		random.shuffle(obsns) # shuffle to ensure no information in whether noise or obs comes first
		obsns.sort(key=itemgetter('timepos'))
		for datumindex, obsn in enumerate(obsns):
			obsn['datumindex'] = datumindex   # and assign a unique id to each datum
			print obsn

		# return the labelled observations, and also the measured mean polyphony
		return {'obsns':obsns, 'polyphony': polyphony}

	def tt_get_fromto(self, state0, state1):
		"Gets entry from the TT corresponding to a state pair. state0 and state1 must be integer indices"
		tt_fromto = [item for item in self.tt[state0] if item['to']==state1]
		if len(tt_fromto)==0:
			return None
		elif len(tt_fromto)==1:
			return tt_fromto[0]
		else:
			print self.tt
			raise Exception("tt_get_fromto(tt, %i, %i) error, found multiple matching options (multiple outarcs to same place?)" % (state0, state1))

	def trans_likelihood(self, state0, state1, timedelta):
		"""Returns the likelihood of the state & time transition. Caches the calculations since it's a profiling hotspot."""
		cachekey = (state0, state1, timedelta)
		if cachekey in self.cache_trans_likelihood:
			return self.cache_trans_likelihood[cachekey]
		fromto = self.tt_get_fromto(state0, state1)
		if fromto == None:
			return 0.
		prob_timedelta = evaluatenormal(fromto['logmeanhop'], fromto['inv_logstdhop'], log(timedelta))
		#timedelta_x0 = max(1e-9, timedelta - (0.5 * timeepsilon))
		#timedelta_x1 =           timedelta + (0.5 * timeepsilon)
		#prob_timetrans  = gaussian_prob_integral(tt_fromto['logmeanhop'], tt_fromto['logstdhop'], log(timedelta_x0), log(timedelta_x1))
		prob_statetrans = fromto['outprob']
		prob = prob_timedelta * prob_statetrans
		self.cache_trans_likelihood[cachekey] = prob
		return prob

	def get_degraded_tt(self, degradeamount=0.5):
		"""To represent inference with mismatched TT, this makes a copy of the Generator,
		finds the nonzero TT entries, resamples some portion of their log-gaussians and some portion of their outarc probys."""
		# LATER
		gen = copy.deepcopy(self)
		for fromindex, fromdata in enumerate(gen.tt):
			if random.random() < degradeamount:
				(outprobs, outprobs_cumul) = random_multinomial(len(gen.tt))
				for toindex, todata in enumerate(fromdata):
					todata['outprob'      ] = outprobs[toindex]
					todata['outprob_cumul'] = outprobs_cumul[toindex]
			for toindex, todata in enumerate(fromdata):
				if random.random() < degradeamount:
					todata['logmeanhop'] = random.gauss(gen.logmeanhopdur, gen.hopdurdiversity)
		return gen

########################################################################################
# utility stuff

SQRTTWOPI = myfloat(sqrt(2. * pi))
ONEOVER_SQRTTWOPI = myfloat(1. / SQRTTWOPI)
def evaluatenormal(mean, inv_stdev, location):
	"Returns value of univariate standard normal at a given location"
	deviance = (location - mean) * inv_stdev
	return (ONEOVER_SQRTTWOPI * inv_stdev) * exp(-0.5 * deviance * deviance)

def snrdb_to_ratio(snrval):
	return 10 ** (-snrval * 0.05)

def gaussian_prob_integral(mean, std, x0, x1):
	"Evaluate the probability of a region, gaussian distribution. x1 must be higher than x0."
	divisor = sqrt(2 * std * std)
	cdf0 = 0.5 * (1 + erf((x0 - mean)/divisor))
	cdf1 = 0.5 * (1 + erf((x1 - mean)/divisor))
	return cdf1 - cdf0

def random_multinomial(numstates):
	"sample a multinomial distribution on outprobs (this is equiv to sampling from a symmetric dirichlet with alpha=1)"
	outprobs = [random.random() for _ in range(numstates)]
	outprobs = [aprob / sum(outprobs) for aprob in outprobs]
	outprobs_cumul = [outprobs[0]]
	for aprob in outprobs[1:]:
		outprobs_cumul.append(outprobs_cumul[-1] + aprob) # build cumulative
	return (outprobs, outprobs_cumul)

########################################################################################

def infer_and_evaluate(obsseq, birthdens_told, deathprob_told, gen_told, snr_told, misseddetectionprob=0):
	"Runs both greedy and full inference, evaluating the results and also timing the things"

	# The probability callbacks
	timeepsilon = 1e-4  # NB we will use a common but arbitrary time resolution in here, so as to be working with probabilities evaluated over intervals
	def transprobcallback(a,b):
		timedelta = abs(b['timepos']-a['timepos'])

		prob = gen_told.trans_likelihood(a['state'], b['state'], timedelta)

		# sparsify the graph to make inference more efficient - when prob==0, arcs are not created
		if prob < 1e-22:
			return 0.
		else:
			return prob
	birthprobcallback   = lambda a: birthdens_told * timeepsilon   # integral of uniform density over the quantum of time
	deathprobcallback   = lambda a: deathprob_told

	# The clutter noise is generated such that the actual density we expect is a ratio of the number of signal obsns (not the number of tracks) we expect.
	# the constant deathprob means track length is geometrically distributed:
	expected_density_sigobs = birthdens_told / deathprob_told   # mean of geometric distrib is 1/p
	noiseratio_told = snrdb_to_ratio(snr_told)
	clutterprobcallback = lambda a: expected_density_sigobs * noiseratio_told * timeepsilon
	print "                          clutterprob: %g" % (expected_density_sigobs * noiseratio_told * timeepsilon)

	if misseddetectionprob == 0:
		obsseq_maybemissed = obsseq
	else:
		obsseq_maybemissed = copy.copy(obsseq)
		random.shuffle(obsseq_maybemissed)
		obsseq_maybemissed = obsseq_maybemissed[:int(len(obsseq_maybemissed)*(1.-misseddetectionprob))]
		obsseq_maybemissed.sort(key=itemgetter('timepos'))
		print "Missed detections: %i -> %i total events" % (len(obsseq), len(obsseq_maybemissed))

	results = {'greedy':{}, 'full':{}}
	for mmrpmode in results.keys():
		greedynotfull = (mmrpmode=='greedy')
		results[mmrpmode]['mmrpmode'] = mmrpmode
		with timer.Timer() as t:
			g = MRPGraph(obsseq_maybemissed, transprobcallback, birthprobcallback, deathprobcallback, clutterprobcallback, maxtimedelta=10.)
			mcf = g.getMinCostFlow(numcutoff=90, greedynotfull=greedynotfull)
			cl = g.getClustersFromMinCostFlow(mcf)
		results[mmrpmode]['msecs'] = t.msecs
		#print "Time taken: %g ms" % t.msecs
		found_sig = [item[1] for item in cl['clusters']]
		found_noi = cl['other'   ]

		# to measure F_{sn} we extract the groundtruth-cluster-IDs from what we found
		found_sig_gt_clustids = [[datum['gtid'] for datum in clust] for clust in found_sig]
		found_noi_gt_clustids =  [datum['gtid'] for datum in found_noi]
		print "true cluster ids, grouped according to the inferred clusters:"
		for alist in found_sig_gt_clustids: print alist
		print "%s <- clutter" % str(found_noi_gt_clustids)
		print "TP: %i" % sum([len( filter(lambda x: x!=-1, alist) ) for alist in found_sig_gt_clustids])
		print "FP: %i" % sum([len( filter(lambda x: x==-1, alist) ) for alist in found_sig_gt_clustids])
		print "TN: %i" % len(filter(lambda x: x!=-1, found_noi_gt_clustids))
		print "FN: %i" % len(filter(lambda x: x!=-1, found_noi_gt_clustids))
		# first arg is a list-of-lists, merely of the CLUSTER NUMBERS, and second arg is also that -- i.e. we use gtid
		results[mmrpmode]['fsn'] = evaluators.fmeasure(found_sig_gt_clustids, found_noi_gt_clustids)

		# to measure F_{trans} we extract the lists-of-datum-IDs from the groundtruth and from the inferred
		found_sig_idseqs = [[datum['datumindex'] for datum in clust] for clust in found_sig]
		truth_sig_idseqs = []
		if len(obsseq) != 0:
			for gtid in range(1+max(o['gtid'] for o in obsseq)):
				truth_sig_idseqs.append([o['datumindex'] for o in obsseq if o['gtid']==gtid])
		# each arg must be a list-of-lists, merely of the ID NUMBERS of the events
		results[mmrpmode]['ftrans'   ] = evaluators.fmeasure_transitions(truth_sig_idseqs, found_sig_idseqs)
		results[mmrpmode]['fsigtrans'] = evaluators.fmeasure_transitions(truth_sig_idseqs, found_sig_idseqs, sigtransonly=True)

	return results



def multi_generate_infer_and_evaluate(alphabetsize=10, hopdurdiversity=1, ttsparsity=0.5, snr=0, noisecorr=0, birthdens=1, deathprob=0.1, numruns=10, \
			birthdens_mism=0, deathprob_mism=0, gen_mism=0, snr_mism=0, misseddetectionprob=0):
	"""Generates 'numruns' independent observation sets and evaluates performance on them.
	Returns a list, with each run's data as a dict."""
	snr_told       = snr + snr_mism
	birthdens_told = birthdens * birthdens_mism
	deathprob_told = deathprob * deathprob_mism
	results = []
	for whichrun in range(numruns):
		generator = Generator(alphabetsize=alphabetsize, hopdurdiversity=hopdurdiversity, ttsparsity=ttsparsity)
		if gen_mism == 0:
			gen_told = generator
		else:
			gen_told = generator.get_degraded_tt(degradeamount=gen_mism)
		generated = generator.generate(snr=snr, noisecorr=noisecorr, birthdens=birthdens, deathprob=deathprob)
		aresult = infer_and_evaluate(generated['obsns'], birthdens_told, deathprob_told, gen_told, snr_told=snr_told, misseddetectionprob=misseddetectionprob)
		for mmrpmoderesult in aresult.values():
			mmrpmoderesult['polyphony'] = generated['polyphony']
			print mmrpmoderesult
			results.append(mmrpmoderesult)
		del generated
		del generator
		del gen_told
	print("gc...")
	print gc.collect()
	return results


def iterate_settings_run(paramslists, numruns=10, csvpath=None):
	"Iterates over the various combinations of settings passed in. Results returned as a list of dicts, and optionally written out as CSV too."
	if csvpath:
		columnslist = ['alphabetsize', 'hopdurdiversity', 'ttsparsity', \
			'snr', 'snr_mism', 'noisecorr', \
			'birthdens', 'birthdens_mism', 'deathprob', 'deathprob_mism', 'gen_mism', \
			'misseddetectionprob', 'mmrpmode', \
			'polyphony', \
			# results...
			'fsn', 'ftrans', 'fsigtrans', 'msecs']
		fp = open(csvpath, 'wb', 1)
		fp.write(','.join(columnslist) + "\n")
	results = []
	totnumcombis = reduce(lambda x,y: x*y, [len(alist) for alist in paramslists.values()])
	curcombinum = 0
	for curvals in itertools.product(*paramslists.values()):
		combi = dict(zip(paramslists.keys(), curvals))
		curcombinum += 1
		print("===================================================================================")
		print "iterate_settings_run(%s) combi %i of %i: %s" % (os.path.basename(csvpath) if csvpath else '', curcombinum, totnumcombis, str(combi))

		result = multi_generate_infer_and_evaluate(alphabetsize=combi['alphabetsize'], hopdurdiversity=combi['hopdurdiversity'], ttsparsity=combi['ttsparsity'], \
			snr=combi['snr'], noisecorr=combi['noisecorr'], birthdens=combi['birthdens'], deathprob=combi['deathprob'], \
			numruns=numruns, \
			birthdens_mism=combi['birthdens_mism'], deathprob_mism=combi['deathprob_mism'], gen_mism=combi['gen_mism'], snr_mism=combi['snr_mism'], \
			misseddetectionprob=combi['misseddetectionprob'])
		for oneresult in result:
			oneresult.update(combi) # parameters and results smooshed into same dict
		results.extend(result)
		# write csv too - combi settings, followed by evaluation & runtime stats etc
		if csvpath:
			for oneresult in result:
				fp.write(','.join(map(str, [oneresult[key] for key in columnslist])) + "\n")
	if csvpath: fp.close()
	return results

########################################################################################
# python ynthetictest.py && xdg-open output/ynth_varying1.csv
if __name__ == '__main__':

	outfolder = 'output'
	#outfolder = 'output_deleteme'

	#########
	print "Sampling one generator, to visualise"
	gen = Generator(alphabetsize=10)
	obsseq = gen.generate(snr=0, birthdens=1, deathprob=0.1, obsdur=20)
	for o in obsseq['obsns']:
		print "    %g\t%i\t%i" % (o['timepos'], o['state'], o['gtid'])
	plt.figure()
	plotem = [[o['timepos'], o['state']] for o in obsseq['obsns'] if o['gtid']==-1]
	plt.plot([pl[0] for pl in plotem], [pl[1] for pl in plotem], 'kx')
	plt.hold(True)
	for gtid in range(1+max(o['gtid'] for o in obsseq['obsns'])):
		plotem = [[o['timepos'], o['state']] for o in obsseq['obsns'] if o['gtid']==gtid]
		plt.plot([pl[0] for pl in plotem], [pl[1] for pl in plotem], '>-')
	plt.ylim(-0.5, gen.alphabetsize-0.5)
	plt.yticks(range(gen.alphabetsize))
	plt.ylabel('State index')
	plt.xlabel('Time (arbitrary units)')
	plt.savefig(outfolder + '/ynth_genex.pdf', format='pdf')
	plt.hold(False)

	#########################################################
	# Edit this list to control which iterative tests to run:
	iteratetypes = [
		'varying1',
		#'varying100',
		'sens_snr',
		'sens_birth',
		'sens_death',
		'sens_noisecorr',
		'sens_missed',
		'sens_tt',
	]

	if 'varying1' in iteratetypes:
		print "Running main test of performance with varying conditions"
		paramslists = {
			'alphabetsize': [10], # [10, 100],
			'hopdurdiversity': [1], # [0.1, 1, 10],
			'snr': [-12, 0, 12, 24], #[0, 6, 12, 18, 24], #[12, -12], #[24, 12, 0, -12, -24],
			'birthdens': [0.1, 0.2],  #[1] seems pretty good, but takes a while to run
			'deathprob': [0.01, 0.1, 0.25], # [0.01, 0.1, 0.25],
			'ttsparsity': [0.1, 0.5],
			'noisecorr': [0],
			'misseddetectionprob': [0],
			'birthdens_mism':[1], 'deathprob_mism':[1], 'gen_mism':[0], 'snr_mism':[0],
		}
		iterate_settings_run(paramslists, numruns=10, csvpath=outfolder + '/ynth_varying1.csv')
	if 'varying100' in iteratetypes:
		print "Running main test of performance with varying conditions, alphabet size 100"
		paramslists = {
			'alphabetsize': [100], # [10, 100],
			'hopdurdiversity': [1], # [0.1, 1, 10],
			'snr': [-12, 0, 12, 24], #[0, 6, 12, 18, 24], #[12, -12], #[24, 12, 0, -12, -24],
			'birthdens': [0.1, 0.2],  #[1] seems pretty good, but takes a while to run
			'deathprob': [0.01, 0.1, 0.25], # [0.01, 0.1, 0.25],
			'ttsparsity': [0.1, 0.5],
			'noisecorr': [0],
			'misseddetectionprob': [0],
			'birthdens_mism':[1], 'deathprob_mism':[1], 'gen_mism':[0], 'snr_mism':[0],
		}
		iterate_settings_run(paramslists, numruns=10, csvpath=outfolder + '/ynth_varying100.csv')
	#########
	print "Running sensitivity tests - mismatched parameters"

	# for each of these we want to choose a SMALL combination of true parameters, and iterate over each mismatch parameter separately
	if 'sens_snr' in iteratetypes:
		paramslists = {
			'alphabetsize': [10],
			'hopdurdiversity': [1],
			'snr': [0], #[0, 6, 12],
			'birthdens': [0.2],
			'deathprob': [0.1],
			'ttsparsity': [0.5],
			'noisecorr': [0],
			'misseddetectionprob': [0],
			'birthdens_mism':[1], 'deathprob_mism':[1], 'gen_mism':[0], 'snr_mism':[-24, -12, -6, 0, 6, 12, 24],
		}
		iterate_settings_run(paramslists, numruns=20, csvpath=outfolder + '/ynth_sens_snr.csv')

	if 'sens_birth' in iteratetypes:
		paramslists = {
			'alphabetsize': [10],
			'hopdurdiversity': [1],
			'snr': [0],
			'birthdens': [0.2],
			'deathprob': [0.1],
			'ttsparsity': [0.5],
			'noisecorr': [0],
			'misseddetectionprob': [0],
			'birthdens_mism':[0.25, 0.5, 1, 2, 4], 'deathprob_mism':[1], 'gen_mism':[0], 'snr_mism':[0],
		}
		iterate_settings_run(paramslists, numruns=20, csvpath=outfolder + '/ynth_sens_birth.csv')

	if 'sens_death' in iteratetypes:
		paramslists = {
			'alphabetsize': [10],
			'hopdurdiversity': [1],
			'snr': [0],
			'birthdens': [0.2],
			'deathprob': [0.1],
			'ttsparsity': [0.5],
			'noisecorr': [0],
			'misseddetectionprob': [0],
			'birthdens_mism':[1], 'deathprob_mism':[0.25, 0.5, 1, 2, 4], 'gen_mism':[0], 'snr_mism':[0],
		}
		iterate_settings_run(paramslists, numruns=20, csvpath=outfolder + '/ynth_sens_death.csv')

	if 'sens_noisecorr' in iteratetypes:
		paramslists = {
			'alphabetsize': [10],
			'hopdurdiversity': [1],
			'snr': [0], ######[12],
			'birthdens': [0.2],
			'deathprob': [0.1],
			'ttsparsity': [0.5],
			'noisecorr': [0, 0.25, 0.5, 1], #[0, 0.25, 0.5, 0.9],
			'misseddetectionprob': [0],
			'birthdens_mism':[1], 'deathprob_mism':[1], 'gen_mism':[0], 'snr_mism':[0],
		}
		iterate_settings_run(paramslists, numruns=20, csvpath=outfolder + '/ynth_sens_noisecorr.csv')

	if 'sens_missed' in iteratetypes:
		paramslists = {
			'alphabetsize': [10],
			'hopdurdiversity': [1],
			'snr': [0],
			'birthdens': [0.2],
			'deathprob': [0.1],
			'ttsparsity': [0.5],
			'noisecorr': [0],
			'misseddetectionprob': [0, 0.1, 0.25, 0.5, 0.75],
			'birthdens_mism':[1], 'deathprob_mism':[1], 'gen_mism':[0], 'snr_mism':[0],
		}
		iterate_settings_run(paramslists, numruns=20, csvpath=outfolder + '/ynth_sens_missed.csv')

	if 'sens_tt' in iteratetypes:
		paramslists = {
			'alphabetsize': [10],
			'hopdurdiversity': [1],
			'snr': [0],
			'birthdens': [0.2],
			'deathprob': [0.1],
			'ttsparsity': [0.5],
			'noisecorr': [0],
			'misseddetectionprob': [0],
			'birthdens_mism':[1], 'deathprob_mism':[1], 'gen_mism':[0, 0.1, 0.25, 0.5, 0.9], 'snr_mism':[0],
		}
		iterate_settings_run(paramslists, numruns=20, csvpath=outfolder + '/ynth_sens_tt.csv')


