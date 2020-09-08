from network import Network

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--iterations", default="1")
parser.add_argument("--input_file")
parser.add_argument("--evidence_size")
parser.add_argument("--entropy_threshold")
parser.add_argument("--evidence_file")
parser.add_argument("--files_folder", default="../uai/")
parser.add_argument("--merlin_folder", default="../merlin")
args = parser.parse_args()


UAI_FILE = args.input_file

# hyperparameters
ITERATIONS = 1 if args.evidence_file else int(args.iterations)
EVIDENCE_SIZE = int(args.evidence_size)
ACCEPT_ENTROPY = float(args.entropy_threshold)

distances = []
matches = []
merlin_times = []
approx_times = []
explained_variables = []
observed_entropies = []
for i in range(ITERATIONS):
	approx_time_sum = 0
	observed_entropies.append([])
	
	net = Network(UAI_FILE, args.evidence_file, args.files_folder, args.merlin_folder)
	
	net.read()  # Read the uai file to parse cardinatilies of the variables
	if not args.evidence_file:
		# EXPLANATION_LENGTH = net.size - EVIDENCE_SIZE
		net.random_evidence(EVIDENCE_SIZE)  # Create a random evidence of size EVIDENCE_SIZE and write it to an .evid file
		net.write_evi_file()
		
	# assert EVIDENCE_SIZE + EXPLANATION_LENGTH < len(net.cardinalities), 'cannot explain more'
	
	for k in range(net.size):
		import timeit, functools
		
		approx_time_sum += timeit.timeit( functools.partial( net.compute_marginals, ACCEPT_ENTROPY ), number=1 )
		if not net.new_observed_variables:
			break
		# print('Var promoted to evidence', net.new_observed_variables)
		net.observed_variables.extend(net.new_observed_variables)
		net.observed_states.extend(net.new_observed_states)
		net.write_evi_file()
		#observed_entropies[i].extend( [ el for i, el in enumerate(net.entropies) if i in net.new_observed_variables ] )
		
		pass
		
	net.find_variables_to_explain()
	
	approx_observed_variables = net.observed_variables
	approx_observed_states = net.observed_states
	net.observed_variables = net.original_observed_variables
	net.observed_states = net.original_observed_states
	net.write_evi_file()
	
	
	approx_times.append(approx_time_sum)
	merlin_times.append( timeit.timeit( functools.partial( net.mmap_query ), number=1 ) )
	exact_expl = net.mmap_expl
	approx_expl = []
	for a, b in zip(approx_observed_variables, approx_observed_states):
		if a in net.to_explain:
			approx_expl.append(b)
			

	# print("Exact solution:", exact_expl, "Approximate solution", approx_expl)
	
	from scipy.spatial.distance import hamming
	import numpy as np
	import math
	# print("Hamming distance", hamming(exact_expl, approx_expl))
	distances.append( 1.0 - hamming(exact_expl, approx_expl) )
	if np.array_equal( exact_expl, approx_expl ): matches.append(1)
	explained_variables.append( len(net.to_explain) )

num_ignored = len(distances)
distances = [ el for i, el in enumerate(distances) if not math.isnan(el) and explained_variables[i] > 1 ]
num_ignored -= len(distances)
matches = [ el for i, el in enumerate(matches) if explained_variables[i] > 1 ]
approx_times = [ el for i, el in enumerate(approx_times) if explained_variables[i] > 1 ]
merlin_times = [ el for i, el in enumerate(merlin_times) if explained_variables[i] > 1 ]
print("iterations", ITERATIONS)
print(explained_variables)
if max(explained_variables) > 0:
	print("Mean number of explained variables", np.mean(explained_variables))
	print("Mean similarity:", np.mean(distances))
	print("Number of matches:", np.sum(matches))
	print("Mean approximate time:", np.mean(approx_times))
	print("Mean exact time:", np.mean(merlin_times))
	print("Observed entropies over iterations:", observed_entropies)
	print("Number of ignored solutions:",num_ignored)
 