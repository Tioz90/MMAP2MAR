import math


def entropy(pmf, base = 2):
	e = 0.0
	if max(pmf) < 1.0:
		for p in pmf:
			if p > 0:
				e -= p*math.log(p, base)
	return e

def mutual_information():
	
	
	return


def mutual_information( self, X, Y, evidence ):
	# if one of the variables is already in the evidence set then return because it makes no sense to calculate
	if X in evidence or Y in evidence:
		return -1
	
	# set up inference using variable elimination algorithm
	from pgmpy.inference import VariableElimination
	
	model_infer = VariableElimination( self.model_pgmpy )
	
	# calculate joint distribution
	joint = model_infer.query( variables=[ X, Y ], evidence=evidence, joint=True )
	
	# calculate marginals from joint
	Y_mar = joint.marginalize( [ X ], inplace=False ).values
	X_mar = joint.marginalize( [ Y ], inplace=False ).values
	
	# sometimes order of joint table is inverted, I want to guarantee Y on rows
	if joint.variables[ 0 ] != Y:
		XY_joint = np.transpose( joint.values )
	else:
		XY_joint = joint.values
	
	from math import log
	mutual_info = 0
	for i in range( len( Y_mar ) ):
		for j in range( len( X_mar ) ):
			try:
				mutual_info += XY_joint[ i, j ] * log( XY_joint[ i, j ] / (Y_mar[ i ] * X_mar[ j ]) )
			except ValueError:
				# in information theory 0*log(0)=0 so I can skip the value
				mutual_info = mutual_info
	
	return mutual_info