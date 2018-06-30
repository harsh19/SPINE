#mkdir data

# download data and prepare pickle dumps
wget http://angelikilazaridou.github.io/resourses/parsing/NP_dataset.tar.gz
tar -xvzf NP_dataset.tar.gz
python get_data.py

# run classifier
# USAGE: python classify.pickley embedding_src num_classes x_train_pickle y_train_pickle x_val_pickle y_val_pickle x_test_pickle y_test_pickle
task="npbracketing"
for i in {0,1} #,2,3,4,5,6,7,8,9}
do
	echo "$i"
	python classify_bracketing.py ../../../pytorch_harsh/spine_logs/epochs2000_noise0.2_sparsity0.15.spine 2 data/"$task"_train_X"$i".pickle data/"$task"_train_y"$i".pickle data/"$task"_val_X"$i".pickle data/"$task"_val_y"$i".pickle data/"$task"_test_X"$i".pickle data/"$task"_test_y"$i".pickle
done