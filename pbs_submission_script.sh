# declare a name for this job to be sample_job
#PBS -N methyl_cpu
# Specify the gpuq queue
#PBS -q default
# Specify the number of gpus
#PBS -l nodes=1:ppn=10
# Specify the gpu feature
#PBS -l mem=200GB
# request 4 hours and 30 minutes of cpu time
#PBS -l walltime=01:00:00
# mail is sent to you when the job starts and when it terminates or aborts
#PBS -m bea
# specify your email address
#PBS -M joshua.j.levy.gr@dartmouth.edu
# Join error and standard output into one file
#PBS -j oe
# By default, PBS scripts execute in your home directory, not the
# directory from which they were submitted. The following line
# places you in the directory from which the job was submitted.
cd $PBS_O_WORKDIR
# run the program
module load python/3-Anaconda
source activate py36
mkdir backup_final_preprocessed
mv final_preprocessed/methyl_array.pkl backup_final_preprocessed
python preprocess.py imputation_pipeline -i ./combined_outputs/methyl_array.pkl -s fancyimpute -m KNN -k 10
python preprocess.py feature_select -n 27000 -nn 10 -f spectral
#python preprocess.py combine_methylation_arrays -d ./preprocess_outputs/ -e OV
#python visualizations.py transform_plot -o prevae_latest.html -nn 8 -d 0.1
#python embedding.py perform_embedding -n 300 -hlt 1000,500 -kl 15 -b 2. -s warm_restarts -lr 5e-5 -bce -e 200
#python visualizations.py transform_plot -o vae_latest.html -i ./embeddings/vae_methyl_arr.pkl -nn 8 -d 0.1
exit 0