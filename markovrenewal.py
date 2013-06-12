#!/bin/env python

# markovrenewal.py - multiple Markov Renewal Process mixture inference
# Written by Dan Stowell May--Dec 2012.
# (c) 2012 Dan Stowell and Queen Mary University of London.
# All rights reserved.

# Requires Python 2.7 or later (small reason: 2.6 doesn't have dict comprehensions)

# NOTE: no numpy to be used in this core code
from math import log, exp
import copy
import tempfile
import os.path
import sys
from operator import itemgetter
import subprocess

max_possible_pathlen = 4000  # path-length-searches must have an upper limit to prevent recursionlimit/outofmemory errors
sys.setrecursionlimit(max_possible_pathlen + 500)

myfloat = float
INF = myfloat('Inf')

##############################################################################
# auxiliary classes
class Vertex:
	def __init__(self, matrixindex):
		"""Args:
		'matrixindex' integer row/col location in the cost matrix, must be unique;
		'vpair' the VertexPair object which is the 'home' of this vertex - only None for source and sink.
		Fill in the inarcs and outarcs and vpair after creation."""
		self.index = matrixindex
		# the user of this class (MRP*Graph) must manually fill in the following data after creation:
		self.vpair = None
		self.inarcs  = []   # Vertex[]
		self.outarcs = []   # Vertex[]
		self.outarccosts = {}    # maps from  int index -> float cost. Should have entries for the "true" outarcs as well as the negative-cost reversed arcs that are found when we create residual networks
	def __repr__(self):
		return "Vertex(%i -%i +%i)" % (self.index, len(self.inarcs), len(self.outarcs))
	def sortoutarcs(self):
		"""Sorts the outarcs into lowest-cost first. This is so that search algorithms can do lowest-cost first iterations"""
		self.outarcs.sort(cmp=self.__sortoutarcs_cmp)
	def __sortoutarcs_cmp(self, a, b):
		return cmp(self.outarccosts[a.index], self.outarccosts[b.index])

class VertexPair:
	"""A pair of vertices representing a single datum in the input data. This exists cos of the
	"vertex expansion" carried out to turn vertex costs into arc costs and simplify path search."""
	def __init__(self, vin, vout, origdatum):
		self.vin  = vin   # Vertex
		self.vout = vout  # Vertex
		self.origdatum = origdatum

class FlowPath:
	def __init__(self, vertexlist, cost, checkStartEnd=True):
		if checkStartEnd and ((vertexlist[0].index != 0) or (vertexlist[-1].index != 1)):
			raise ValueError("Expected vertexlist to start at source (0) and end at sink (1); got %i, %i" % (vertexlist[0].index, vertexlist[-1].index))
		self.vertexlist = vertexlist
		self.cost = cost

	def verify(self, mrpabstractgraph):
		"No-op. Edit code to name this 'verify_NOTNOOP' (and rename the other), if you want verification to be used."
		pass
	def verify_DEACTIVATED(self, mrpabstractgraph):
		"Verifies the sequence of vertices and the cost"
		cost = myfloat(0)
		for index in xrange(len(self.vertexlist)-1):
			frm = self.vertexlist[index]
			too = self.vertexlist[index+1]
			if frm==too:
				raise ValueError("FlowPath.verify: self transition for vertex %i" % (frm.index))
			if not frm in too.inarcs:
				raise ValueError("FlowPath.verify: %i->%i in sequence but arc-end knows of know such arc" % (frm.index, too.index))
			if not too in frm.outarcs:
				raise ValueError("FlowPath.verify: %i->%i in sequence but arc-start knows of know such arc" % (frm.index, too.index))
			cost += mrpabstractgraph.getArcCost(frm, too)
		if abs(cost - self.cost) > 1e-11: # small delta allowed re floating-point accuracy
			raise ValueError("FlowPath.verify: recalc'ed cost as %g, but was told it was %g (diff %g)" % (cost, self.cost, cost - self.cost))

	def getorigdatalist(self):
		"""traverses and returns list of "origdata" items. Note that we 
		DO NOT YET avoid duplication (inevitable cos of vertex expansion).
		Probably will implement, but there's no guarantee the path always hits 
		pairs, except once the flows are added together in which case a 
		feasible flow must always hit pairs.
		TODO: de-duplicate in the MRPGraph class, after summing."""
		return [v.vpair.origdatum for v in self.vertexlist[1:-1]]

	def __repr__(self):
		return "FlowPath[%.3g: %s]" % (self.cost, ','.join([str(v.index) for v in self.vertexlist]))

	def __eq__(self, other):
		if isinstance(other, self.__class__):
			return (self.cost == other.cost) and (self.vertexlist == other.vertexlist)
		else:
			return False
	def __ne__(self, other):
		return not self.__eq__(other)

def _logval(innum):
	if innum == 0.: return -INF
	else:           return log(innum)

def lessthanzero(val):
	return val < 0

##############################################################################
# convenience methods

def mrp_autochunk_and_getclusters(
			# MRP ctor args:
			data, transprobcallback, birthprobcallback, deathprobcallback, clutterprobcallback, maxtimedelta,
			# cluster-getting args:
			numcutoff, greedynotfull
			):
	"If the data neatly separates with time gaps greater than maxtimedelta, the problem can be easily divided into smaller ones, which should run faster in practice."
	data.sort(key=itemgetter('timepos'))
	#print "mrp_autochunk_and_getclusters: tot len %i" % len(data)
	# Now we go through the data, and for each separable chunk we run MRP. Then we need to merge the clusters and the noise back together	
	chunkstart = 0
	prevtimepos = data[0]['timepos']
	allresults = {'clusters':[], 'other':[]}
	for i in xrange(1,len(data)):
		delta = data[i]['timepos'] - prevtimepos
		if (delta > maxtimedelta) or (i == len(data)-1):
			#print "  processing chunk [%i,%i), length %i" % (chunkstart, i, i-chunkstart)
			g = MRPGraph(data[chunkstart:i], transprobcallback, birthprobcallback, deathprobcallback, clutterprobcallback, maxtimedelta)
			mcf = g.getMinCostFlow(numcutoff=numcutoff, greedynotfull=greedynotfull)
			cl = g.getClustersFromMinCostFlow(mcf)
			#print "  got %i clusters" % (len(cl['clusters']))
			allresults['clusters'].extend(cl['clusters'])
			allresults['other'   ].extend(cl['other'   ])
			chunkstart = i
			#print gc.collect() # now is a good time to gc
		prevtimepos = data[i]['timepos']
	return allresults

##############################################################################
# main classes

class MRPAbstractGraph:
	"""An abstract superclass having common functionality for both:
	 - MRPGraph which represents the network flow setup;
	 - MRPResidual which is a residual network from that."""
	def __init__(self):
		raise NotImplementedError   # we are abstract

	def getArcCost(self, vfrom, vto):
		return vfrom.outarccosts[vto.index]

	def getMinCostPath(self, maxcostthreshold=INF, greedynotfull=False):
		"""Shortest-path algorithm (breadth-first).
		Returns a FlowPath, or None if none exists (when source has 0 outdegree).
		If maxcostthreshold is set (e.g. to 0) then the search is optimised to ignore any possible paths 
		  greater than this value. May return None even if a path (more expensive than the threshold) exists.
		Note the value "max_possible_pathlen" (in code) is used to limit the max path length, 
		  primarily to limit search time in large graphs and avoid running out of memory."""
		# Initialise tmp vals used during search algorithm.
		searchdata = { \
			'incurrentpath': [False for _ in xrange(len(self.orderedvertexlist))], \
			# Best cost from source--->this
			'bestcosts':     [INF   for _ in xrange(len(self.orderedvertexlist))], \
			# Immediate parent in best path
			'bestparents':   [None  for _ in xrange(len(self.orderedvertexlist))], \
		}
		searchdata['bestcosts'][0] = myfloat(0) # source-> source cost is special, our starting position

		# iterate the graph, adding up all NEGATIVE costs that are actually active. (this will help us trim down some fruitless search branches)
		totnegcostsavailable = sum([ \
				sum(filter(lessthanzero, [self.getArcCost(v, kid) for kid in v.outarcs])) \
			for v in self.orderedvertexlist])

		self.__getMinCostPath_examinesingletons(searchdata)
		try:
			self.__getMinCostPath_depthfirst(self.source, totnegcostsavailable, max_possible_pathlen, 
						maxcostthreshold, searchdata, greedynotfull=greedynotfull)
		except RuntimeError:
			print "===================================================================="
			print "MRPAbstractGraph:depthfirst RuntimeError."
			print "Graph has %i nodes total." % len(self.orderedvertexlist)
			whotrue = [v.index for v in filter(lambda v: searchdata['incurrentpath'][v.index], self.orderedvertexlist)]
			print "incurrentpath is true for %i nodes:" % len(whotrue)
			print whotrue
			print "the outward arcs of the sink lead to:"
			print [v.index for v in self.sink.outarcs]
			#raw_input()
			#raise
			return None

		# We should have examined everything - if we haven't reached the sink then no more feasible paths exist.
		if searchdata['bestparents'][1] == None:
			return None
		
		vertexlist = []
		ascender = self.sink
		vertexlist.append(ascender)
		while searchdata['bestparents'][ascender.index] != None:
			ascender = searchdata['bestparents'][ascender.index]
			vertexlist.append(ascender)
		vertexlist.reverse()
		if vertexlist[0] != self.source:
			print "ValueError: shortestpath found a result... source should be first but isn't."
			print [v.index for v in vertexlist]
		fp = FlowPath(vertexlist, searchdata['bestcosts'][self.sink.index])
		fp.verify(self)   # could be deactivated later for speed
		return fp

	def __getMinCostPath_examinesingletons(self, searchdata):
		"Explicitly examines singleton forward paths, in the hope of helping to lower-bound various path costs"
		for index in xrange(2, len(self.orderedvertexlist), 2):
			vin  = self.orderedvertexlist[index]
			vout = self.orderedvertexlist[index+1]
			if len(vin.outarcs)==1 and vin.outarcs[0]==vout:  # if is fwd
				pathcost = 0.
				frm = self.source
				tooi = vin.index
				pathcost += frm.outarccosts[tooi]
				if pathcost < searchdata['bestcosts'][tooi]:
					searchdata['bestcosts'][tooi] = pathcost
					searchdata['bestparents'][tooi] = frm
				frm = vin
				tooi = vout.index
				pathcost += frm.outarccosts[tooi]
				if pathcost < searchdata['bestcosts'][tooi]:
					searchdata['bestcosts'][tooi] = pathcost
					searchdata['bestparents'][tooi] = frm
				frm = vout
				tooi = self.sink.index
				pathcost += frm.outarccosts[tooi]
				if pathcost < searchdata['bestcosts'][tooi]:
					searchdata['bestcosts'][tooi] = pathcost
					searchdata['bestparents'][tooi] = frm

	def __getMinCostPath_depthfirst(self, v, negcostsavailable, lengthremaining, maxcostthreshold, searchdata, greedynotfull):
		"""depth-first search recursion used by the getMinCostPath() method"""
		if searchdata['incurrentpath'][v.index]:  # TEMPORARY -- validation -- remove this once code is stable
			raise RuntimeError("exploring node %i that is already in current path" % v.index)
		searchdata['incurrentpath'][v.index] = True
		# examine all children who are not already in the path.
		# Note: outarcs is sorted into min-cost-first (by the constructor etc), 
		#  so that means we're doing min-cost-first here & hopefully cutting off some branches of futile exploration.
		for kid in v.outarcs:
			if searchdata['incurrentpath'][kid.index]:
				continue
			if (greedynotfull and (kid.index!=1 and kid.index<v.index)):
				continue
			arccost = self.getArcCost(v, kid)
			pathcost = searchdata['bestcosts'][v.index] + arccost
			# NOTE: the following test changed from "<" to "<=" because of my pre-evaluation of singletons filling in the exact value
			if pathcost <= searchdata['bestcosts'][kid.index]:   # if this path is fruitful, remember it
				searchdata['bestcosts'][kid.index] = pathcost
				searchdata['bestparents'][kid.index] = v
				# if we're a good alternative (and not yet reached the sink) recurse
				if kid.index != 1:
					if (pathcost + negcostsavailable) > searchdata['bestcosts'][1]: # i.e. no chance to get a better route to the sink
						pass
					elif (pathcost + negcostsavailable) > maxcostthreshold: # i.e. no chance to get a route we'd want to keep
						pass  # this one is rarely invoked since maxcostthreshold defaults to inf. can be used to truncate.
					elif lengthremaining==0:
						print "Max path-length reached."
						pass
					else:
						# The kid will need to know what negative costs are available once we've added the arc
						kid_negcostsavailable = negcostsavailable - min(arccost, 0.)
						self.__getMinCostPath_depthfirst(kid, kid_negcostsavailable, lengthremaining,
									maxcostthreshold, searchdata, greedynotfull=greedynotfull)
		# Now drop back, taking ourselves out of the current path
		searchdata['incurrentpath'][v.index] = False

	def _getMinCostFlowResidual(self, pathcostcutoff=0, numcutoff=INF, vizpath=None, recurcount=0, greedynotfull=False):
		"""Does most of the recursive work of getting a min-cost flow, but doesn't recover the flow,
		leaves it implicit in the residual network which is returned. Might not even return a true
		'residual' since if there is no best path, it just returns the original network.
		This is the Edmonds-Karp algorithm approach."""
		bestpath = self.getMinCostPath(pathcostcutoff, greedynotfull=greedynotfull)
		if (bestpath == None) or (bestpath.cost > pathcostcutoff) or (numcutoff==0):
			return self
		bestpath.verify(self)   # could be deactivated later for speed
		if vizpath != None:
			self.graphviz([bestpath], "%s_r%i" % (vizpath, recurcount))
		residual = MRPResidual(self, bestpath)
		finalresidual = residual._getMinCostFlowResidual(pathcostcutoff, numcutoff-1, vizpath, recurcount+1, greedynotfull=greedynotfull)  # RECURSION
		return finalresidual

	def findNegativeCostCycle(self):
		"""Uses Bellman-Ford to find a negative cost cycle.
		NOTE that this adds an implicit arc from sink back to source (zero cost inf cap), which in practice allows it to modify the flow value,
		so we use it to search for the generally min-cost flow at any flow value.
		A 'true' MRPGraph can have no negative cycles so the ones formed using the implicit arc are the 
		only ones that can be found; an MRPResidual may very well have other cycles.
		Returns None if there is no cycle, or a FlowPath representing the cycle."""
		#print ">findNegativeCostCycle"
		### Bellman-Ford-Moore:
		# Initialise source-to-each-vertex distances, and parents
		bestfroms = [None] * len(self.orderedvertexlist)
		bestdists = [INF]  * len(self.orderedvertexlist)
		bestdists[0] = 0
		# Do N-1 times
		for _ in xrange(len(self.orderedvertexlist)-1):
			# For each arc
			for frm in self.orderedvertexlist:
				for too in frm.outarcs:
					# relax the distance
					newdist = bestdists[frm.index] + frm.outarccosts[too.index]
					if newdist < bestdists[too.index]:
						bestdists[too.index] = newdist
						bestfroms[too.index] = frm
			# Here we handle the implicit zero-cost arc from sink to source
			if bestdists[1] < bestdists[0]:
				bestdists[0] = bestdists[1]
				bestfroms[0] = self.sink
		#print " findNegativeCostCycle - done the push"
		# Now look for the most negative-weight cycle:
		cyc = None
		cyccost = 0.
		for frm in self.orderedvertexlist:
			for too in frm.outarcs:
				newdist = bestdists[frm.index] + frm.outarccosts[too.index]
				if newdist < bestdists[too.index]:
					#print "Found a negative-cost cycle, while inspecting %i->%i (cost %g)" % (frm.index, too.index, bestdists[too.index])
					(newcyc, newcyccost) = self.__findNegativeCostCycle_reconstruct(bestfroms, frm)
					#print [v.index for v in newcyc]
					#print "cost is %g" % newcyccost
					if newcyccost == 0.:
						print "NOTE: cycle of zero cost discovered (indicates multiple equivalent solutions available)"
					if newcyccost < cyccost:
						#print "this is best so far (cost %g); keeping" % newcyccost
						cyc = newcyc
						cyccost = newcyccost
						break   # I originally thought we'd not break, but look for the most-neg cycle; not sure if that actually makes sense
			if cyc != None: break  # if python could break out of both loops at once I wouldn't need this
		# Here we handle the implicit zero-cost arc from sink to source
		if bestdists[1] < bestdists[0]:
			#print "Found a negative-cost cycle, while inspecting %i->%i (cost %g)" % (1, 0, bestdists[0])
			(newcyc, newcyccost) = self.__findNegativeCostCycle_reconstruct(bestfroms, self.sink)
			#print [v.index for v in newcyc]
			#print "cost is %g" % newcyccost
			if newcyccost == 0.:
				print "NOTE: cycle of zero cost discovered (indicates multiple equivalent solutions available)"
			if newcyccost < cyccost:
				#print "this is best so far (cost %g); keeping" % newcyccost
				cyc = newcyc
				cyccost = newcyccost
		#print "<findNegativeCostCycle"
		if cyc == None:
			return None
		else:
			return FlowPath(cyc, cyccost, checkStartEnd=False)

	def __findNegativeCostCycle_reconstruct(self, bestfroms, startnode):
		cyclist = []
		cyccost = myfloat(0.)
		frm = startnode
		too = bestfroms[frm.index]
		while True:
			# update
			cyclist.append(too)
			cyccost += frm.outarccosts[too.index]
			# ascend
			frm = too
			too = bestfroms[frm.index]
			if len(cyclist)>1 and (frm.index == cyclist[0].index):
				break
			# THIS IS FOR DEBUG PURPOSES -- IF WE CAN FIX THE PROBLEM OF INFINITE LOOPS EVER OCCURRING, WE DON'T NEED IT
			if len(cyclist)>1000:
				print "over-long cyclist detected:"
				print [x.index for x in cyclist]
				raise RuntimeError("over-long cyclist detected")				
		cyclist.reverse()
		# If the cycle includes the sink->source or source->sink arcs explicitly, we must re-glue it so that it's implicit.
		for i in xrange(len(cyclist)-1):
			if (cyclist[i]==self.sink and cyclist[i+1]==self.source) or \
			   (cyclist[i]==self.source and cyclist[i+1]==self.sink):
				print "Reglueing:"
				print [x.index for x in cyclist]
				cyclist = cyclist[i+1:] + cyclist[1:i+1]
				print [x.index for x in cyclist]
				break
		cyccost = -cyccost
		return (cyclist, cyccost)

	def graphviz(self, flow=None, path=None, filetype="pdf"):
		"""Returns graphviz code representing the network. If 'flow' is given it is an array of FlowPaths
		to be highlighted. If 'path' is given then the code is also written to file and a PDF is rendered."""
		# this array increases thickness to show flow
		arrowthickness = [[1 for _ in xrange(len(self.orderedvertexlist))] for _ in xrange(len(self.orderedvertexlist))]
		if flow != None:
			for fp in flow:
				fp.verify(self)   # could be deactivated later for speed
				for fpindex in xrange(len(fp.vertexlist)-1):
					frm = fp.vertexlist[fpindex].index
					too = fp.vertexlist[fpindex+1].index
					arrowthickness[frm][too] += 3
		code = "digraph G {\nrankdir = LR;\n\n"
		# Write out source and sink specially (special style and perhaps location, but ID consistent with numbering)
		code += "v0 [style=\"filled\", penwidth=3, label=\"s\", rank=\"source\"];\n"
		code += "v1 [style=\"filled\", penwidth=3, label=\"t\", rank=\"sink\"];\n"
		code += "\n\n"
		# Write out each vertexpair (as a little box round the two).
		# Since we can't truly access vertexpair (residual doesn't have them), 
		# we build these by making assumptions about the indexing (numbering goes [source, sink, in, out, in, out, in, out...])
		numdatums = len(self.source.outarcs) + len(self.source.inarcs)
		for vpindex in xrange(numdatums):
			code += "subgraph cluster%i { v%i; v%i; }\n" % (vpindex, 2 + (vpindex * 2), 3 + (vpindex * 2))
		code += "\n\n"
		# Write out each arc
		def writeanarc(frmi, tooi, cost):
			# reversed arcs (from sink, or to source, or from higher index to lower one) are drawn as reverse-reverse
			if (frmi==1) or (tooi==0) or ((tooi!=1) and (frmi > tooi)):
				return "v%i -> v%i [label=\"%.3g\", penwidth=%i%s];\n" % (tooi, frmi, cost, arrowthickness[frmi][tooi], \
							", dir=\"back\", style=\"dashed\"")
			else:
				return "v%i -> v%i [label=\"%.3g\", penwidth=%i];  \n" % (frmi, tooi, cost, arrowthickness[frmi][tooi])
		# We do a each-node-once traversal to write all the arcs
		indicesdone = []
		queue = [self.source, self.sink]
		while len(queue) > 0:
			v = queue[0]
			queue = queue[1:]
			if v.index in indicesdone:
				continue
			indicesdone.append(v.index)
			for kid in v.outarcs:
				code += writeanarc(v.index, kid.index, self.getArcCost(v, kid))
				if not kid.index in indicesdone:
					queue.append(kid)
		# finish off syntax
		code += "\n}\n"
		if path != None:
			dotpath = tempfile.gettempdir() + '/' + os.path.basename(path) + ".dot"
			f = open(dotpath, 'w')
			f.write(code)
			f.close()
			subprocess.call(["dot", dotpath, "-T%s" % filetype, "-o%s.%s" % (path, filetype)])
		return code

	@classmethod
	def arcisreverse(cls, frm, too):
		"Returns true if arc is a reverse one (judged by the numbering scheme)"
		return (too.index != 1) and (too.index < frm.index)

class MRPGraph(MRPAbstractGraph):
	"""A network flow graph specialised for the Markov Renewal Process inference.
	This means that various assumptions are baked in: unit capacities, integer flow,
	graph is 'simple' (no vertex has >1 indegree AND >1 outdegree), sink and source
	are superconnected, and origdata contains 'timepos' so that arcs will be made 
	for every pair of data where the delta < maxtimedelta."""
	def __init__(self, origdata, transprobcallback, birthprobcallback, deathprobcallback, clutterprobcallback, maxtimedelta=5):
		"""'origdata' must be a list of datums each having 'timepos' (used to decide whether to arc).
		A shallow copy of origdata is made so that we can sort it and guarantee timepos-order.
		The callbacks will all be called by the constructor lots of times, as it precalculates the cost matrix."""
		# Vertices (no arcs at this point!)
		self.source = Vertex(0)  #  source must go in list at index 0
		self.sink   = Vertex(1)  #  sink   must go in list at index 1
		# Note: do NOT change the numbering - many bits of code assume [source, sink, in, out, in, out...]
		self.orderedvertexlist = [self.source, self.sink]
		self.vpairs = []
		# There are no direct arcs between source and sink; 
		# however when using cycle-cancelling for minCostFlow we use a 'pretend' arc to complete the loop, 
		# which is when these two arc-costs become relevant:
		self.source.outarccosts[self.sink.index] = 0.
		self.sink.outarccosts[self.source.index] = 0.
		# Now add all the stuff
		self.transprobcallback = transprobcallback
		self.birthprobcallback = birthprobcallback
		self.deathprobcallback = deathprobcallback
		self.clutterprobcallback = clutterprobcallback
		self.__appendData(origdata, maxtimedelta)

	def __appendData(self, newdata, maxtimedelta=5):
		"""Used by __init__ and by update: assuming we have a list of nodes containing AT LEAST [source, sink], this appends
		the supplied data points, building up arcs and calculating costs.
		IMPORTANT: The supplied data points must have a time position NO LOWER THAN that of any points already added.
		This is because the old and new data are sorted separately, THEN concatenated and iterated, and must be in correct order."""
		if len(newdata)==0: return # can crash, if we don't test for this
		# Create shallow copy of the data and ensure it's ok for our purposes
		newdata = copy.copy(newdata)
		newdata.sort(key=itemgetter('timepos'))
		if len(self.vpairs) > 0 and self.vpairs[-1].origdatum['timepos'] > newdata[0]['timepos']:
			raise ValueError("New input data contains timepos %g earlier than already-used timepos %g" \
					% (newdata[0]['timepos'], self.vpairs[-1].origdatum['timepos']))
		# find the oldest neighbour that we need to consider as a potential predecessor to any of our new data
		pasthorizon = newdata[0]['timepos'] - maxtimedelta
		pasthorizonindex = len(self.vpairs)
		while (pasthorizonindex>0) and (self.vpairs[pasthorizonindex-1].origdatum['timepos'] >= pasthorizon):
			pasthorizonindex -= 1

		autoinc = len(self.orderedvertexlist)
		for datum in newdata:
			# Each item in newdata becomes a VertexPair
			vin  = Vertex(autoinc)
			vout = Vertex(autoinc+1)
			self.orderedvertexlist.append(vin)
			self.orderedvertexlist.append(vout)
			vp = VertexPair(vin, vout, datum)
			vin.vpair = vp
			vout.vpair = vp
			vpindex = len(self.vpairs)
			self.vpairs.append(vp)
			autoinc += 2
			# Calc costs and create arcs:
			# source->vin
			self.__makearc(self.source, vp.vin, -_logval(self.birthprobcallback(vp.origdatum)))
			# vin->vout
			self.__makearc(vp.vin, vp.vout, _logval(self.clutterprobcallback(vp.origdatum))) # NB "log" not "-log" for clutter only
			# vout->sink
			deathprob = self.deathprobcallback(vp.origdatum)
			self.__makearc(vp.vout, self.sink, -_logval(deathprob))
			# and between neigbouring vpairs
			for earliervpindex in xrange(pasthorizonindex, vpindex): # triangular iteration over potential predecessors
				earliervp = self.vpairs[earliervpindex]
				survprob = 1 - self.deathprobcallback(earliervp.origdatum)
				timedelta = vp.origdatum['timepos'] - earliervp.origdatum['timepos']
				# NB timedelta here must be >0, not >=0; transitions to same-time are banned as could cause infinite loops
				if (timedelta > 0) and (timedelta <= maxtimedelta):
					# NB: multiplying by survprob here -- we assume the transition probys normalise to 1,
					#  i.e. P(state | prevstate && notdead) - we multiply by P(notdead) to give P(state | prevstate)
					self.__makearc(earliervp.vout, vp.vin, -_logval(survprob * self.transprobcallback(earliervp.origdatum, vp.origdatum)))
		# Now pre-sort outarcs by weight - do this for every vertex we have added or might have touched
		for vindex in xrange(self.vpairs[pasthorizonindex].vout.index, len(self.orderedvertexlist), 2):
			self.orderedvertexlist[vindex].sortoutarcs()

	def __makearc(self, v1, v2, cost):
		"weighted arc is created, represented both as a pair of costs in the costmatrix, and as pointers in the Vertex objects"
		if cost != INF:
			v1.outarcs.append(v2)
			v2.inarcs.append(v1)
			v1.outarccosts[v2.index] =  cost
			v2.outarccosts[v1.index] = -cost

	def getMinCostFlow(self, pathcostcutoff=0, numcutoff=INF, vizpath=None, greedynotfull=False):
		"""returns a list of mutually orthogonal FlowPaths. With default args this should
		be a minimum-cost flow (having maximum flow among all minimum-cost flows). The args
		can be used to truncate search at a particular pathcost or flow cardinality.
		This is the Edmonds-Karp algorithm approach."""
		residual = self._getMinCostFlowResidual(pathcostcutoff, numcutoff, vizpath, greedynotfull=greedynotfull)
		flowpaths = self.getFlowFromResidual(residual)
		if vizpath != None:
			self.graphviz(flowpaths, "%s_final" % (vizpath))
		return flowpaths

	def updateMinCostFlow(self, prevflow=None, newdata=None, vizpath=None):
		"""Given an MRPGraph, a flow in that graph, and some new data to be added, 
		this uses the cycle-cancelling algorithm to update to a new optimal flow.
		NOTE: the new data must be later in time than the old data; we do not check this.
		Returns a list of mutually orthogonal FlowPaths. With default args this should
		be a minimum-cost flow (having maximum flow among all minimum-cost flows). The args
		can be used to truncate search at a particular pathcost or flow cardinality."""
		if prevflow==None:
			prevflow = []
		else:
			prevflow = copy.copy(prevflow)    # shallow copy - we won't change any entries, but might add new ones

		# add the new data to our graph, inc all the new arcs, costs etc
		ouroldsize = len(self.orderedvertexlist)
		if newdata != None:
			self.__appendData(newdata, maxtimedelta=5)

		for ini in xrange(ouroldsize, len(self.orderedvertexlist), 2):
			# for each added datum, test whether the singleton flow is worth doing - if so, add it to prevflow
			# (cycle-cancelling may completely change these later, but they're a good first pass)
			nodeA = self.orderedvertexlist[ini]
			nodeB = self.orderedvertexlist[ini+1]
			cost = self.source.outarccosts[ini] + nodeA.outarccosts[ini+1] + nodeB.outarccosts[1]
			if cost < 0.:
				print "Newly-added datum at index %i has good singleton cost (%g); adding to draft flow" % (ini, cost)
				prevflow.append(FlowPath([self.source, nodeA, nodeB, self.sink], cost))
		
		# create a residual using all the "draft" flowpaths
		residual = self
		for flowpath in prevflow:
			residual = MRPResidual(residual, flowpath)

		###### Cycle-cancelling algorithm for min cost flow:
		while True:
			cycle = residual.findNegativeCostCycle()
			if cycle == None:
				print "No cycle found"
				break
			residual = MRPResidual(residual, cycle)
		flowpaths = self.getFlowFromResidual(residual)
		for fp in flowpaths: print fp
		return flowpaths

	def getFlowFromResidual(self, residual):
		"""Takes an MRPGraph and a residual, reconstructs a flow (i.e. an array of FlowPath objects).
		This lookup-builder assumes a forwards-flow which is why it's only defined for true MRPGraphs."""
		lookup = dict((v.index,v) for v in [self.source, self.sink] + self.source.outarcs + [vv.outarcs[0] for vv in self.source.outarcs])
		flowpaths = []
		# Every outarc from the SINK represents a unit flow
		for ascender in residual.sink.outarcs:
			cost = 0.
			nums = [1]
			frm = residual.sink
			while True:
				# We track the reverse-order numbers (and the costs)
				nums.append(ascender.index)
				cost += self.getArcCost(ascender, frm)
				if ascender.index == 0:
					break
				# Now we update "ascender" and "frm" to represent the next arc in our journey
				frm = ascender
				# The index-ordering means we're always looking for the lower-index outarc, until we hit source.
				reversearcs = filter(lambda v: MRPGraph.arcisreverse(ascender, v), ascender.outarcs)
				if len(reversearcs) != 1:
					raise ValueError("getMinCostFlow expected only one reverse arc from vertex %i" % (ascender.index))
				ascender = reversearcs[0]
			nums.reverse()
			# and then to recover the Vertex objects we ask self.source for its kids, converted into a number lookup.
			fp = FlowPath([lookup[index] for index in nums], cost)
			fp.verify(self)   # could be deactivated later for speed
			flowpaths.append(fp)
		return flowpaths

	def getClustersFromMinCostFlow(self, mcf, plotpath=None, statekey='statepos', plotunderway=False, plotfontsize="small", plotclutter=True):
		"""Returns: {'clusters': list of lists of (cost, [origdata]), 'other': list of nonincluded origdata points}.
		Supply plotpath if you want a plot written to a file; OR set plotunderway=True to assume a plot is underway (e.g. a subplot)."""
		# Since every FlowPath starts at 0 and must by definition go through both parts of a vertexpair, 
		#  we hop through entries [1,3,5,7...] to gather them up
		def collectdatums(fp):
			return [fp.vertexlist[index].vpair.origdatum for index in xrange(1, len(fp.vertexlist)-1, 2)]
		clusters = [(fp.cost, collectdatums(fp)) for fp in mcf]
		# To collect the leftovers, we find all of the children of the source whose indices are not known to the flowpaths
		indicesknown = []
		for fp in mcf:
			indicesknown.extend([v.index for v in fp.vertexlist])
		other = [v.vpair.origdatum for v in filter(lambda vv: not vv.index in indicesknown, self.source.outarcs)]
		if (plotpath != None) or plotunderway:
			import matplotlib.pyplot as plt
			if not plotunderway:
				fig = plt.figure()
			plt.hold(True)
			for clust in clusters:
				x = [datum['timepos'] for datum in clust[1]]
				y = [datum[statekey ] for datum in clust[1]]
				plt.plot(x, y, 'b-', alpha=0.25)
				plt.plot(x, y, 'b+')
				if not plotunderway:
					plt.text(x[-1], y[-1], "%1.3g" % exp(-clust[0]), fontsize=plotfontsize) # signal-and-noise vs noise likelihood ratio
			if len(clusters) > 0:
				plt.title("LR %1.3g" % (exp(sum([-clust[0] for clust in clusters]))), fontsize=plotfontsize)
			if plotclutter:
				x = [datum['timepos'] for datum in other]
				y = [datum[statekey ] for datum in other]
				if plotunderway:
					plt.plot(x, y, ',', color="0.2")
				else:
					plt.plot(x, y, '.', color="0.5")
			plt.xticks(fontsize=plotfontsize)
			plt.yticks(fontsize=plotfontsize)
			if not plotunderway:
				plt.xlabel('Time (s)')
				plt.ylabel('State')
				plt.savefig("%s.pdf" % plotpath, papertype='A4', format='pdf')
			plt.hold(False)
		return {'clusters': clusters, 'other': other}

	def getVpairForDatum(self, datum):
		"ONLY FOR DEBUG (in unit test) - for a datum, return the Vpair."
		for vp in self.vpairs:
			if vp.origdatum == datum:
				return vp
		raise ValueError("not found")

	def getMinCostPath_BellmanFord(self):
		"""Uses Bellman-Ford to find the single shortest path.
		This is ONLY applicable to the original graph NOT the residual, since the latter is
		not guaranteed to be negative-cycle-free.
		Returns the FlowPath shortest path, or throws an error if there was a negative cycle.
		PLEASE NOTE: UNIT TESTING INDICATES THIS IS SLOWER THAN THE BREADTH-FIRST ALGO. Test speed on your own data.
		"""
		### Bellman-Ford-Moore:
		# Initialise source-to-each-vertex distances, and parents
		bestfroms = [None] * len(self.orderedvertexlist)
		bestdists = [INF]  * len(self.orderedvertexlist)
		bestdists[0] = 0
		# Do N-1 times
		for _ in xrange(len(self.orderedvertexlist)-1):
			# For each arc
			for frm in self.orderedvertexlist:
				for too in frm.outarcs:
					# relax the distance
					newdist = bestdists[frm.index] + frm.outarccosts[too.index]
					if newdist < bestdists[too.index]:
						bestdists[too.index] = newdist
						bestfroms[too.index] = frm
		# Check for negative-cost cycles
		for frm in self.orderedvertexlist:
			for too in frm.outarcs:
				newdist = bestdists[frm.index] + frm.outarccosts[too.index]
				if newdist < bestdists[too.index]:
					raise RuntimeError("Found a negative-cost cycle, while inspecting %i->%i (cost %g)" % (frm.index, too.index, bestdists[too.index]))
		# Reconstruct the path
		vertexlist = []
		ascender = self.sink
		vertexlist.append(ascender)
		while bestfroms[ascender.index] != None:
			ascender = bestfroms[ascender.index]
			vertexlist.append(ascender)
		vertexlist.reverse()
		if vertexlist[0] != self.source:
			print "ValueError: shortestpath found a result... source should be first but isn't."
			print [v.index for v in vertexlist]
		fp = FlowPath(vertexlist, bestdists[self.sink.index])
		fp.verify(self)   # could be deactivated later for speed
		return fp

class MRPResidual(MRPAbstractGraph):
	"""A residual network, created from a MRPGraph (or MRPResidual) and a flowpath.
	It DOES share some data structure with the parent, namely the cost matrix,
	but creates a copy set of vertices and arcs, so that it can change them."""
	def __init__(self, mrpparent, flowpath):
		# The main job is to create a copy of every vertex, and recreate the arcs (before applying the flowpath change)
		# Start by scraping the newly-minted vertices into an array in same order as in the cost matrix, with same arcs
		self.orderedvertexlist = [Vertex(index) for index in xrange(len(mrpparent.orderedvertexlist))]
		self.source = self.orderedvertexlist[0]
		self.sink   = self.orderedvertexlist[1]
		for v in mrpparent.orderedvertexlist:
			# recreate arcs, ensuring to connect to new vertices rather than to parent's ones
			self.orderedvertexlist[v.index].inarcs  = [self.orderedvertexlist[other.index] for other in v.inarcs ]
			self.orderedvertexlist[v.index].outarcs = [self.orderedvertexlist[other.index] for other in v.outarcs]
			self.orderedvertexlist[v.index].outarccosts = v.outarccosts  # mapping is immutable index->cost so we can reuse the cost list directly, no copy
		# Then for each Vertex->Vertex arc in flowpath, we need to reverse the arc - i.e.:
		#	to the vin: remove vout reference from outarcs, add it to inarcs
		#	to the vout: remove vout reference from outarcs, add it to inarcs
		flowpathindices = [v.index for v in flowpath.vertexlist]
		for i in xrange(len(flowpathindices)-1):
			frmi = flowpathindices[i]
			tooi = flowpathindices[i+1]
			# Validation that there's a well-formed arc there for us to remove
			if len(  filter(lambda v: v.index == tooi, self.orderedvertexlist[frmi].outarcs)  ) != 1:
				print "%i.inarcs: %s" % (frmi, [one.index for one in self.orderedvertexlist[frmi].inarcs])
				print "%i.outarcs: %s" % (frmi, [one.index for one in self.orderedvertexlist[frmi].outarcs])
				print "%i.inarcs: %s" % (tooi, [one.index for one in self.orderedvertexlist[tooi].inarcs])
				print "%i.outarcs: %s" % (tooi, [one.index for one in self.orderedvertexlist[tooi].outarcs])
				raise ValueError("flowpath led us to expect arc %i->%i, but %i has no such outarc" % (frmi, tooi, frmi))
			if len(  filter(lambda v: v.index == frmi, self.orderedvertexlist[tooi].inarcs )  ) != 1:
				print "%i.inarcs: %s" % (frmi, [one.index for one in self.orderedvertexlist[frmi].inarcs])
				print "%i.outarcs: %s" % (frmi, [one.index for one in self.orderedvertexlist[frmi].outarcs])
				print "%i.inarcs: %s" % (tooi, [one.index for one in self.orderedvertexlist[tooi].inarcs])
				print "%i.outarcs: %s" % (tooi, [one.index for one in self.orderedvertexlist[tooi].outarcs])
				raise ValueError("flowpath led us to expect arc %i->%i, but %i has no such inarc" % (frmi, tooi, tooi))
			# use 'filter' to drop the existing arc
			self.orderedvertexlist[frmi].outarcs = filter(lambda v: v.index != tooi, self.orderedvertexlist[frmi].outarcs)
			self.orderedvertexlist[tooi].inarcs  = filter(lambda v: v.index != frmi, self.orderedvertexlist[tooi].inarcs )
			# and create the reverse arc
			self.orderedvertexlist[frmi].inarcs.append( self.orderedvertexlist[tooi])
			self.orderedvertexlist[tooi].outarcs.append(self.orderedvertexlist[frmi])
			# then update the sorting, for faster search
			self.orderedvertexlist[tooi].sortoutarcs()

