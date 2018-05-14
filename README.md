Necessary folders:
- Data
- Results
- Models

# Generate training data 
python "Dataset_Generators/generators__manifold-learning/swiss-roll.py" -n 1000 -noise 0.1 -seed 1 -output "Data/swiss-roll-1000.train"

# Generate testing data
python "Dataset_Generators/generators__manifold-learning/swiss-roll.py" -n 1000 -noise 0.2 -seed 2 -output "Data/swiss-roll-1000.test"

# Train the model
python train.py -input Data/swiss-roll-1000.train -model Models/swiss_roll-1000.model -bits 8 -log Models/swiss_roll-1000-train.log

# Test the model
python test.py -model Models/swiss_roll-1000.model -test_file Data/swiss-roll-1000.test -compressor Compressors.vanilla -log_file_test Results/swiss_roll-1000-vanilla.log -log_file_others Results/swiss_roll-1000-vanilla-others.log

# Create the 2d plot
python plot_hashcodes.py -model ./Models/4clusters_noise20_n1000_bits8.model -input ./Data/4clusters_noise20_n1000.train -compressor Compressors.pc_dominance_by_modes_order


# The MNIST & SIFT Datasets 
The **MNIST** & **SIFT** datasets can be generated by running the *setup.sh* file in their directories. These datasets are included in the *Dataset_Generators* directory. You'll need to have **python3** and **wget** installed on your machine to be able to run these scripts. 

# Test a compressor on a given dataset. The file creates its own model with the given number of bits
python test_exact.py -data ./Data/sift.train -compressor Compressors.vanilla -bits 32 -samples 100

