from utilities import entropy
import numpy as np
import random
import os


class Network:
	
	def __init__(self, filename, initial_evidence, files_folder, merlin_path):
		self.file = filename
		self.size = -1
		self.cardinalities = []
		self.observed_variables = []
		self.observed_states = []
		self.original_observed_variables = []
		self.original_observed_states = []
		self.mpe_file = filename + '2.mmap'
		self.new_observed_states = -1
		self.new_observed_variables = -1
		self.marginals = []
		self.entropies = []
		self.most_probable_states = []
		self.to_explain = []
		self.mmap_expl = None
		
		self.files_folder = files_folder
		self.merlin_path = merlin_path
		self.evidence_file = filename + '_work.evid'
		if initial_evidence:
			from shutil import copyfile
			
			copyfile(self.files_folder + initial_evidence, self.files_folder + self.evidence_file)
	
	def read(self):
		with open(self.files_folder + self.file) as fp:
			line = fp.readline()
			cnt = 1
			while line:
				line = fp.readline()
				if cnt == 2:
					self.cardinalities = [int(_) for _ in line.strip().split(' ')]
				cnt += 1
		self.size = len(self.cardinalities)
	
	def random_evidence(self, d):
		np.random.seed(42)
		permutation = np.random.permutation(self.size)
		observed_variables = permutation[:d]
		for observed_variable in observed_variables:
			cardinality = self.cardinalities[observed_variable]
			observed_state = random.randint(0, cardinality - 1)
			self.observed_variables.append(observed_variable)
			self.observed_states.append(observed_state)
			self.original_observed_variables.append(observed_variable)
			self.original_observed_states.append(observed_state)
	
	def write_evi_file(self):
		f = open(self.files_folder + self.evidence_file, "w")
		evidence_string = '%d ' % (len(self.observed_variables))
		for variable, state in zip(self.observed_variables, self.observed_states):
			evidence_string += "%d %d " % (variable, state)
		f.write(evidence_string)
		f.close()
	
	def compute_marginals(self, entropy_threshold):
		self.marginals = []
		self.entropies = []
		self.most_probable_states = []
		os.system(self.merlin_path + '/merlin -f ' + self.files_folder + self.file + ' -t MAR -a bte -e ' + self.files_folder + self.evidence_file + ' -o ' + self.files_folder + self.file+' > ' + self.merlin_path + '/merlin_mar')
		with open(self.files_folder + self.file + '.MAR') as fp:
			line = fp.readline()
			cnt = 1
			while line:
				line = fp.readline()
				if cnt == 5:
					marginals = line.strip().split(' ')
				cnt += 1
		step = 2
		
		for variable in range(self.size):
			k = self.cardinalities[variable]
			mass_function = [1.0 / k for _ in range(self.cardinalities[variable])]
			if variable not in self.observed_variables:
				for state in range(self.cardinalities[variable]):
					mass_function[state] = float(marginals[step + state])
			step += self.cardinalities[variable] + 1
			self.marginals.append(mass_function)
			self.entropies.append(entropy(mass_function, self.cardinalities[variable]))
			self.most_probable_states.append(mass_function.index(max(mass_function)))
			
		self.new_observed_variables = [i for i, ent in enumerate(self.entropies) if ent < entropy_threshold]
		self.new_observed_states = [self.most_probable_states[i] for i in self.new_observed_variables]
		
		return

	def find_variables_to_explain(self):
		self.to_explain = [_ for _ in self.observed_variables if _ not in self.original_observed_variables]

	def mmap_query(self):
		mpe_string = '%d ' % len(self.to_explain)
		# Run over the difference between observed variables and original observed variables
		for variable in self.to_explain:
			mpe_string += str(variable)+' '
			
		f = open(self.files_folder + self.file + '.mmapq', "w")
		f.write(mpe_string)
		f.close()
		
		os.system(self.merlin_path + '/merlin -v 3 -f ' + self.files_folder + self.file + ' -t MMAP -a bte -q ' + self.files_folder + self.file + '.mmapq -e ' + self.files_folder + self.evidence_file + ' -o ' + self.files_folder + self.file + '.mmap'+' > ' + self.merlin_path + '/merlin_mmap')
		with open(self.files_folder + self.file + '.mmap.MMAP') as fp:
			line = fp.readline()
			cnt = 1
			while line:
				line = fp.readline()
				if cnt == 1:
					mmap_expl = line.strip().split(' ')
				cnt += 1
				
		self.mmap_expl = [int(_) for _ in mmap_expl[1:]]
		
		
