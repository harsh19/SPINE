#mkdir data

# download data and prepare pickle dumps
python get_newsgroup_data.py

# run classifier
# USAGE: python classify.py embedding_src num_classes x_train_pickle y_train_pickle x_val_pickle y_val_pickle x_test_pickle y_test_pickle
python classify.py ../../../pytorch_harsh/spine_logs/epochs2000_noise0.2_sparsity0.15.spine 2 data/news_computer_train_X.p data/news_computer_train_y.p data/news_computer_val_X.p data/news_computer_val_y.p data/news_computer_test_X.p data/news_computer_test_y.p