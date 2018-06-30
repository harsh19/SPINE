from nltk import word_tokenize
from sklearn.datasets import fetch_20newsgroups
import pickle

computer_categories = ['comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware']
religion_categories = ['alt.atheism', 'soc.religion.christian']


def get_Xy(data):
    X, y = [], []
    for idx in range(len(data['data'])):
        X.append(word_tokenize(data['data'][idx].lower()))
        y.append(data['target'][idx])
    return X, y


def get_everything(categories, val_fraction, name, data_dir="data/"):
    data_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
    data_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)
    data_test_X, data_test_y = get_Xy(data_test)
    data_X, data_y = get_Xy(data_train)
    
    print "len(data_X):",len(data_X)
    print "len(data_test_X):",len(data_test_X)
    val_threshold = int( (1.0 - val_fraction)* len(data_X) )

    data_train_X, data_train_y = data_X[:val_threshold], data_y[:val_threshold]
    data_val_X, data_val_y = data_X[val_threshold:], data_y[val_threshold:]
    
    pickle.dump(data_train_X, open(data_dir+name+"_train_X.p", 'w'))
    pickle.dump(data_train_y, open(data_dir+name+"_train_y.p", 'w'))
    pickle.dump(data_val_X, open(data_dir+name+"_val_X.p", 'w'))
    pickle.dump(data_val_y, open(data_dir+name+"_val_y.p", 'w'))
    pickle.dump(data_test_X, open(data_dir+name+"_test_X.p", 'w'))
    pickle.dump(data_test_y, open(data_dir+name+"_test_y.p", 'w'))

    
get_everything(computer_categories,0.2,'news_computer')
get_everything(religion_categories,0.2,'news_religion')