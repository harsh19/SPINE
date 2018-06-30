#mkdir data

# download data and prepare pickle dumps
cd data
wget http://cogcomp.org/Data/QA/QC/train_5500.label
wget http://cogcomp.org/Data/QA/QC/TREC_10.label
cd ..
python create_data.py

# run classifier
# USAGE: python classify.pickley embedding_src num_classes x_train_pickle y_train_pickle x_val_pickle y_val_pickle x_test_pickle y_test_pickle
task="qa"
python classify.py ../../../pytorch_harsh/spine_logs/epochs2000_noise0.2_sparsity0.15.spine 2 data/"$task"_train_X.pickle data/"$task"_train_y.pickle data/"$task"_val_X.pickle data/"$task"_val_y.pickle data/"$task"_test_X.pickle data/"$task"_test_y.pickle
