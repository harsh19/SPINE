#!/bin/sh
#SBATCH --mem 8000
#SBATCH --gres=gpu:1
#SBATCH --time=0
source /home/jharsh/anaconda2/bin/activate p2

for epoch in 4000 1000 2000 500; 
	do 
	for sparsity in 0.15 0.015; 
		do 
		for noise in 0.3 0.2 0.1 0.0 
			do
			python main.py --input=data/word2vec/word2vec_15k_300d_leipzig_train.txt --output=spine_logs/word2vec_epochs"$epoch"_noise"$noise"_sparsity"$sparsity".spine --num_epochs=$epoch --denoising --noise=$noise --sparsity=$sparsity | tee ./spine_logs/word2vec_epochs"$epoch"_noise"$noise"_sparsity"$sparsity".log
		done;
	done; 
done


source /home/jharsh/anaconda2/bin/deactivate
