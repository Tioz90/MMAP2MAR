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
	 
	def compute_joint( self, variable,folder, file ):
		filename=folder+file
		with open( filename, 'r' ) as f:
			data = f.readlines()
			
			# increment variables
			num_variables = int( data[ 1 ] )
			data[ 1 ] = str( num_variables + 1 ) + "\n"
			
			# add a new dummy variable of cardinality 2
			cardinalities = data[ 2 ].split()
			data[ 2 ] = " ".join( cardinalities )
			data[ 2 ] += " 2\n"
			
			# increment fields
			num_fields = int( data[ 3 ] )
			data[ 3 ] = str( num_fields + 1 ) + "\n"
			
			# declare new field for dummy variable
			data.insert(num_fields+4,"dummy field")
			data.append("dummy field")
		
		marginals = {}
		for v in range(num_variables):
			if v!=variable:
				# fill in new field for dummy variable
				data[num_fields+4]= "3 "+str(num_variables)+" "+str(variable)+" "+str(v)+"\n"
				
				# add the field cardinality for the dummy variable
				data[-1]= "\n"+str( 2 * int( cardinalities[ variable ] ) * int( cardinalities[ v ] ) )
				
				# write a UAI file for each combination of states in the couple of variables

				for i in range(int(cardinalities[variable])):
					for j in range(int(cardinalities[v])):
						joint_filename = file +"_"+ "_".join([str(variable),str(v),str(i),str(j)])
						os.makedirs( os.path.dirname( folder + "joints/" + joint_filename ), exist_ok=True )
						
						with open( folder + "joints/" + joint_filename, 'w' ) as f:
							f.writelines( data )
							
						with open( folder + "joints/" + joint_filename, 'a' ) as f:
							field = np.zeros( int( cardinalities[variable ] ) * int( cardinalities[v ] ) )
							field[ i *int(cardinalities[v]) + j ]=1
							field_inverted = 1 - field
							f.write("\n")
							f.write(" ".join("".join(item) for item in field.astype( str ) ) + " "+ " ".join("".join(item) for item in field_inverted.astype( str ) ) )
						
						# calculate marginals for network with dummy variable
						os.system(self.merlin_path + '/merlin -f ' + folder + "joints/" + joint_filename +' -t MAR -a bte' +' -e ' + self.files_folder + self.evidence_file +' -o ' + self.files_folder + self.file +' > ' + self.merlin_path + '/merlin_mar' )
						with open( self.files_folder + self.file + '.MAR' ) as f:
							data_marginals = f.readlines()
							marginal = data_marginals[5].split()[-2]
		return
	
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
			
			self.compute_joint(variable=variable,folder=self.files_folder ,file=self.file,)
			
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
		
		
