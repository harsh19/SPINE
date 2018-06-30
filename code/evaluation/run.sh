# usage: ./run.sh <embedding_absolute_path>
# e.g. ./run.sh /Users/user/Workspace/projects/SPINE/code/embeddings/spine/SPINE_glove.txt 

cd extrinsic_downstream

echo $1
emb=$1
echo "Evaluating on"
echo "$emb"


echo "---------->>>>>>> Evaluating on newsgroups"
cd newsgroups
# run classifier
# USAGE: python classify.py embedding_src num_classes x_train_pickle y_train_pickle x_val_pickle y_val_pickle x_test_pickle y_test_pickle
python classify.py "$emb" 2 data/news_computer_train_X.p data/news_computer_train_y.p data/news_computer_val_X.p data/news_computer_val_y.p data/news_computer_test_X.p data/news_computer_test_y.p
python classify.py "$emb" 2 data/news_religion_train_X.p data/news_religion_train_y.p data/news_religion_val_X.p data/news_religion_val_y.p data/news_religion_test_X.p data/news_religion_test_y.p


echo "---------->>>>>>> Evaluating on np bracketing"
cd ../np_bracketing
# run classifier
# USAGE: python classify.pickley embedding_src num_classes x_train_pickle y_train_pickle x_val_pickle y_val_pickle x_test_pickle y_test_pickle
task="npbracketing"
for i in {0,1} #,2,3,4,5,6,7,8,9}
do
	echo "$i"
	python classify_bracketing.py "$emb" 2 data/"$task"_train_X"$i".pickle data/"$task"_train_y"$i".pickle data/"$task"_val_X"$i".pickle data/"$task"_val_y"$i".pickle data/"$task"_test_X"$i".pickle data/"$task"_test_y"$i".pickle
	#python classify_bracketing.py "$emb" 2 'data/"$task"_train_X"$i".pickle data/"$task"_train_y"$i".pickle' 'data/"$task"_val_X"$i".pickle data/"$task"_val_y"$i".pickle' 'data/"$task"_test_X"$i".pickle' 'data/"$task"_test_y"$i".pickle'
done


echo "---------->>>>>>> Evaluating on trec"
cd ../TREC/
# run classifier
# USAGE: python classify.pickley embedding_src num_classes x_train_pickle y_train_pickle x_val_pickle y_val_pickle x_test_pickle y_test_pickle
task="qa"
python classify.py  "$emb" 2 data/"$task"_train_X.pickle data/"$task"_train_y.pickle data/"$task"_val_X.pickle data/"$task"_val_y.pickle data/"$task"_test_X.pickle data/"$task"_test_y.pickle




### Intrinsic
cd ../../intrinsic/


python evaluate_wordSim.py "$emb"






