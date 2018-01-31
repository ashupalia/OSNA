
from collections import Counter, defaultdict
from itertools import chain, combinations
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from scipy.sparse import csr_matrix
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
import string
import tarfile
import urllib.request
import time

def download_data():
    """ Download and unzip data.
    DONE ALREADY.
    """
    url = 'https://www.dropbox.com/s/xk4glpk61q3qrg2/imdb.tgz?dl=1'
    urllib.request.urlretrieve(url, 'imdb.tgz')
    tar = tarfile.open("imdb.tgz")
    tar.extractall()
    tar.close()


def read_data(path):
    """
    Walks all subdirectories of this path and reads all
    the text files and labels.
    DONE ALREADY.
    Params:
      path....path to files
    Returns:
      docs.....list of strings, one per document
      labels...list of ints, 1=positive, 0=negative label.
               Inferred from file path (i.e., if it contains
               'pos', it is 1, else 0)
    """
    fnames = sorted([f for f in glob.glob(os.path.join(path, 'pos', '*.txt'))])
    data = [(1, open(f).readlines()[0]) for f in sorted(fnames)]
    fnames = sorted([f for f in glob.glob(os.path.join(path, 'neg', '*.txt'))])
    data += [(0, open(f).readlines()[0]) for f in sorted(fnames)]
    data = sorted(data, key=lambda x: x[1])
    return np.array([d[1] for d in data]), np.array([d[0] for d in data])


def tokenize(doc, keep_internal_punct=False):
    """
    Tokenize a string.
    The string should be converted to lowercase.
    If keep_internal_punct is False, then return only the alphanumerics (letters, numbers and underscore).
    If keep_internal_punct is True, then also retain punctuation that
    is inside of a word. E.g., in the example below, the token "isn't"
    is maintained when keep_internal_punct=True; otherwise, it is
    split into "isn" and "t" tokens.
    Params:
      doc....a string.
      keep_internal_punct...see above
    Returns:
      a numpy array containing the resulting tokens.
    >>> tokenize(" Hi there! Isn't this fun?", keep_internal_punct=False)
    array(['hi', 'there', 'isn', 't', 'this', 'fun'],
          dtype='<U5')
    >>> tokenize("Hi there! Isn't this fun? ", keep_internal_punct=True)
    array(['hi', 'there', "isn't", 'this', 'fun'],
          dtype='<U5')
    >>> tokenize("??necronomicon?? geträumte sünden.<br>Hi", True)
    array(['necronomicon', 'geträumte', 'sünden.<br>hi'],
        dtype='<U13')
    >>> tokenize("??necronomicon?? geträumte sünden.<br>Hi", False)
    array(['necronomicon', 'geträumte', 'sünden', 'br', 'hi'],
        dtype='<U12')
    
    if keep_internal_punct == False:
        return np.array(re.sub('\W+', ' ', doc.lower()).split())
    else:
        words = [word.strip(string.punctuation) for word in doc.lower().split(" ")]
        filtered = [word for word in words if word]
        return np.array(filtered)
    """
    res = []
    output = list()
    if keep_internal_punct==False:
        res = re.sub('\W+',' ', doc.lower()).split()
    if keep_internal_punct==True:
        res = re.sub(r'(?<!\S)[^\s\w]+|[^\s\w]+(?!\S)',' ', doc.lower()).split()
    resulting_array = np.array(res)
    return resulting_array

def token_features(tokens, feats):
    """
    Add features for each token. The feature name
    is pre-pended with the string "token=".
    Note that the feats dict is modified in place,
    so there is no return value.
    Params:
      tokens...array of token strings from a document.
      feats....dict from feature name to frequency
    Returns:
      nothing; feats is modified in place.
    >>> feats = defaultdict(lambda: 0)
    >>> token_features(['hi', 'there', 'hi'], feats)
    >>> sorted(feats.items())
    [('token=hi', 2), ('token=there', 1)]
    
    count = Counter(tokens)
    str1="token="
    for token in tokens:
        feats[str1 + token]=count[token]
    """   
    pre = "token="
    pro = "token"
    count=Counter()
    count.update(tokens)
    for i in count:
        feats.update({pre+i:count[i]})

def token_pair_features(tokens, feats, k=3):
    """
    Compute features indicating that two words occur near
    each other within a window of size k.
    For example [a, b, c, d] with k=3 will consider the
    windows: [a,b,c], [b,c,d]. In the first window,
    a_b, a_c, and b_c appear; in the second window,
    b_c, c_d, and b_d appear. This example is in the
    doctest below.
    Note that the order of the tokens in the feature name
    matches the order in which they appear in the document.
    (e.g., a__b, not b__a)
    Params:
      tokens....array of token strings from a document.
      feats.....a dict from feature to value
      k.........the window size (3 by default)
    Returns:
      nothing; feats is modified in place.
    >>> feats = defaultdict(lambda: 0)
    >>> token_pair_features(np.array(['a', 'b', 'c', 'd']), feats)
    >>> sorted(feats.items())
    [('token_pair=a__b', 1), ('token_pair=a__c', 1), ('token_pair=b__c', 2), ('token_pair=b__d', 1), ('token_pair=c__d', 1)]
    """
    count = Counter()
    qr = list()
    raj = {}
    Tokens_list=list(tokens)
    arra = []
    list_new=[]
    a_list = []
    for a in [Tokens_list[i-1:i-1+k] for i in range(1,len(Tokens_list))]:
        if len(a)>=k:
            list_new.append(a)
            a_list.append(a)
    for i in list_new:
        new=list(combinations(i,2))
        count.update(new)

    for keys in sorted(count):
        feats["token_pair=" + keys[0] + "__" + keys[1]] = count[keys]
        
neg_words = set(['bad', 'hate', 'horrible', 'worst', 'boring'])
pos_words = set(['awesome', 'amazing', 'best', 'good', 'great', 'love', 'wonderful'])


def lexicon_features(tokens, feats):
    """
    Add features indicating how many time a token appears that matches either
    the neg_words or pos_words (defined above). The matching should ignore
    case.
    Params:
      tokens...array of token strings from a document.
      feats....dict from feature name to frequency
    Returns:
      nothing; feats is modified in place.
    In this example, 'LOVE' and 'great' match the pos_words,
    and 'boring' matches the neg_words list.
    >>> feats = defaultdict(lambda: 0)
    >>> lexicon_features(np.array(['i', 'LOVE', 'this', 'great', 'boring', 'movie']), feats)
    >>> sorted(feats.items())
    [('neg_words', 1), ('pos_words', 2)]
    """
    positive = 0
    negative = 0
    for token in tokens:
        if token.lower() in pos_words:
            positive += 1
        if token.lower() in neg_words:
            negative += 1
    feats.update({'neg_words':negative})
    feats.update({'pos_words':positive})
    
    
def featurize(tokens, feature_fns):
    """
    Compute all features for a list of tokens from
    a single document.
    Params:
      tokens........array of token strings from a document.
      feature_fns...a list of functions, one per feature
    Returns:
      list of (feature, value) tuples, SORTED alphabetically
      by the feature name.
    >>> feats = featurize(np.array(['i', 'LOVE', 'this', 'great', 'movie']), [token_features, lexicon_features])
    >>> feats
    [('neg_words', 0), ('pos_words', 2), ('token=LOVE', 1), ('token=great', 1), ('token=i', 1), ('token=movie', 1), ('token=this', 1)]
    """
    dic = dict()
    lis = list()
    feats = defaultdict(lambda:0)
    for a in feature_fns:
        a(tokens,feats)
    result = sorted(feats.items())
    return sorted(feats.items())
    pass

def vectorize(tokens_list, feature_fns, min_freq, vocab=None):
    """
    Given the tokens for a set of documents, create a sparse
    feature matrix, where each row represents a document, and
    each column represents a feature.
    Params:
      tokens_list...a list of lists; each sublist is an
                    array of token strings from a document.
      feature_fns...a list of functions, one per feature
      min_freq......Remove features that do not appear in
                    at least min_freq different documents.
    Returns:
      - a csr_matrix: See https://goo.gl/f5TiF1 for documentation.
      This is a sparse matrix (zero values are not stored).
      - vocab: a dict from feature name to column index. NOTE
      that the columns are sorted alphabetically (so, the feature
      "token=great" is column 0 and "token=horrible" is column 1
      because "great" < "horrible" alphabetically),
    >>> docs = ["Isn't this movie great?", "Horrible, horrible movie"]
    >>> tokens_list = [tokenize(d) for d in docs]
    >>> feature_fns = [token_features]
    >>> X, vocab = vectorize(tokens_list, feature_fns, min_freq=1)
    >>> type(X)
    <class 'scipy.sparse.csr.csr_matrix'>
    >>> X.toarray()
    array([[1, 0, 1, 1, 1, 1],
           [0, 2, 0, 1, 0, 0]], dtype=int64)
    >>> sorted(vocab.items(), key=lambda x: x[1])
    [('token=great', 0), ('token=horrible', 1), ('token=isn', 2), ('token=movie', 3), ('token=t', 4), ('token=this', 5)]
    """
    list2 = list()
    VocabUnsort={}
    list1 = [0]
    feats=list()
    NumberOfTerm={}
    list3 = list()
    flag = True
    newdi = defaultdict(lambda:0)
    raj = list()
    for d in tokens_list:
        featureval=featurize(d,feature_fns) 
        feats.append(featureval)
        for s in featureval: 
            
            if s[0] not in NumberOfTerm:
                NumberOfTerm[s[0]]=1
            else:
                NumberOfTerm[s[0]]=NumberOfTerm[s[0]]+1
            
            lenghth = len(VocabUnsort)
            VocabUnsort.setdefault(s[0], lenghth) 
    dirf = list() 
    dirf2 = list()
    #ff
    if vocab==None:        
        vocab=defaultdict(lambda:0)
        sor = sorted(VocabUnsort)
        raja = True
        Vivu = 0
        for i in sor:
            if i in NumberOfTerm and NumberOfTerm[i]>=min_freq:  
                vocab.setdefault(i,len(vocab))
    for d in feats:
        for s in d: 
            if s[0] in vocab:  
                index = vocab[s[0]]
                list2.append(index)
                list3.append(s[1])
        nel = len(list2)
        list1.append(nel)
        
    X=csr_matrix((list3, list2, list1), dtype=np.int64)
    return X,vocab  
    
def accuracy_score(truth, predicted):
    """ Compute accuracy of predictions.
    DONE ALREADY
    Params:
      truth.......array of true labels (0 or 1)
      predicted...array of predicted labels (0 or 1)
    """
    return len(np.where(truth==predicted)[0]) / len(truth)


def cross_validation_accuracy(clf, X, labels, k):
    """
    Compute the average testing accuracy over k folds of cross-validation. You
    can use sklearn's KFold class here (no random seed, and no shuffling
    needed).
    Params:
      clf......A LogisticRegression classifier.
      X........A csr_matrix of features.
      labels...The true labels for each instance in X
      k........The number of cross-validation folds.
    Returns:
      The average testing accuracy of the classifier
      over each fold of cross-validation.
    """
    cv = KFold(len(labels), k)
    accuracy = list()
    acu = list()
    ad = list()
    for training, testing in cv:
        clf.fit(X[training], labels[training])
        predicted = clf.predict(X[testing])
        accuracy.append(accuracy_score(labels[testing], predicted))
    result = np.mean(accuracy)
    return result
    pass

def eval_all_combinations(docs, labels, punct_vals,
                          feature_fns, min_freqs):
    """
    Enumerate all possible classifier settings and compute the
    cross validation accuracy for each setting. We will use this
    to determine which setting has the best accuracy.
    For each setting, construct a LogisticRegression classifier
    and compute its cross-validation accuracy for that setting.
    In addition to looping over possible assignments to
    keep_internal_punct and min_freqs, we will enumerate all
    possible combinations of feature functions. So, if
    feature_fns = [token_features, token_pair_features, lexicon_features],
    then we will consider all 7 combinations of features (see Log.txt
    for more examples).
    Params:
      docs..........The list of original training documents.
      labels........The true labels for each training document (0 or 1)
      punct_vals....List of possible assignments to
                    keep_internal_punct (e.g., [True, False])
      feature_fns...List of possible feature functions to use
      min_freqs.....List of possible min_freq values to use
                    (e.g., [2,5,10])
    Returns:
      A list of dicts, one per combination. Each dict has
      four keys:
      'punct': True or False, the setting of keep_internal_punct
      'features': The list of functions used to compute features.
      'min_freq': The setting of the min_freq parameter.
      'accuracy': The average cross_validation accuracy for this setting, using 5 folds.
      This list should be SORTED in descending order of accuracy.
      This function will take a bit longer to run (~20s for me).
    """
    list_op = list()
    list_ft = list()
    out_lis = list()
    
    lenghth = len(feature_fns)
    for i in range(1, lenghth+1):
        for j in combinations(feature_fns, i):
            list_ft.append(j)
        

    for a in punct_vals:
        document_val = [tokenize(d, a) for d in docs]
        for new_feature in list_ft:
            for b in min_freqs:
                final_dic = {}
                ad = []
                qr = {}
                mod = LogisticRegression()
                matrix, vocab = vectorize(document_val, new_feature, b)
                avg = cross_validation_accuracy(mod, matrix, labels, 5)
                final_dic.update({'punct':a})
                final_dic.update({'features': new_feature})
                final_dic['min_freq'] = b
                final_dic['accuracy'] = avg
                list_op.append(final_dic)
    result = sorted(list_op, key=lambda x: (-x['accuracy'], -x['min_freq']))
    return result

def plot_sorted_accuracies(results):
    """
    Plot all accuracies from the result of eval_all_combinations
    in ascending order of accuracy.
    Save to "accuracies.png".
    """
    
    accuracy = list()
    for a in results:
        accuracy.append(a['accuracy'])

    plt.plot(sorted(accuracy))
    plt.xlabel('setting', size = 14)
    plt.ylabel('accuracy', size = 14)
    plt.savefig("accuracies.png")
    
    
def mean_accuracy_per_setting(results):
    """
    To determine how important each model setting is to overall accuracy,
    we'll compute the mean accuracy of all combinations with a particular
    setting. For example, compute the mean accuracy of all runs with
    min_freq=2.
    Params:
      results...The output of eval_all_combinations
    Returns:
      A list of (accuracy, setting) tuples, SORTED in
      descending order of accuracy.
    """
    outo = list()
    final_res = list()
    for i in results:
        final_res.append(("features", i['features']))
        final_res.append(("min_freq", i['min_freq']))
        final_res.append(("punct", i['punct']))
        
    re = list()
    flag = 0
    reww = True
    ad = set(final_res)
    for result in ad:
        acc_res = list()
        for r in results:
            if r[result[0]] == result[1]:
                acc_res.append(r['accuracy'])
        if 'features' not in result[0]:
            re.append((np.mean(acc_res), (result[0] + "=" + str(result[1]))))
        else:
            f_name = [f.__name__ for f in result[1]]
            re.append((np.mean(acc_res), str(("features=" + ' '.join(f_name)))))
            
    sor = sorted(re, key=lambda x: -x[0])
    return sor



def fit_best_classifier(docs, labels, best_result):
    """
    Using the best setting from eval_all_combinations,
    re-vectorize all the training data and fit a
    LogisticRegression classifier to all training data.
    (i.e., no cross-validation done here)
    Params:
      docs..........List of training document strings.
      labels........The true labels for each training document (0 or 1)
      best_result...Element of eval_all_combinations
                    with highest accuracy
    Returns:
      clf.....A LogisticRegression classifier fit to all
            training data.
      vocab...The dict from feature name to column index.
    """
    token_list = list()
    for ab in docs:
        token_list.append(tokenize(ab,best_result['punct']))

    matrix,vocab=vectorize(token_list,best_result['features'],best_result['min_freq'])
    mod=LogisticRegression()
    mod.fit(matrix,labels)
    return mod,vocab


def top_coefs(clf, label, n, vocab):
    """
    Find the n features with the highest coefficients in
    this classifier for this label.
    See the .coef_ attribute of LogisticRegression.
    Params:
      clf.....LogisticRegression classifier
      label...1 or 0; if 1, return the top coefficients
              for the positive class; else for negative.
      n.......The number of coefficients to return.
      vocab...Dict from feature name to column index.
    Returns:
      List of (feature_name, coefficient) tuples, SORTED
      in descending order of the coefficient for the
      given class label.
    
    """
    
    list11 =list()
    poslist1=list()
    neglist2=list()
    for a,b in zip(sorted(vocab),clf.coef_[0]):
        if b < 0:
            neglist2.append((a,-b))
        else:
            poslist1.append((a,b))
    
    if label != 0:
        poslist1 = sorted(poslist1,key=lambda k:k[1],reverse=True)
        return poslist1[:n]
    else:
        neglist2 = sorted(neglist2,key=lambda k:k[1],reverse=True)
        return neglist2[:n]


def parse_test_data(best_result, vocab):
    """
    Using the vocabulary fit to the training data, read
    and vectorize the testing data. Note that vocab should
    be passed to the vectorize function to ensure the feature
    mapping is consistent from training to testing.
    Note: use read_data function defined above to read the
    test data.
    Params:
      best_result...Element of eval_all_combinations
                    with highest accuracy
      vocab.........dict from feature name to column index,
                    built from the training data.
    Returns:
      test_docs.....List of strings, one per testing document,
                    containing the raw.
      test_labels...List of ints, one per testing document,
                    1 for positive, 0 for negative.
      X_test........A csr_matrix representing the features
                    in the test data. Each row is a document,
                    each column is a feature.
    """
    tokens = list()
    tok = list()
    docs,labels = read_data(os.path.join('data','test'))
    for doc in docs:
        tokens.append(tokenize(doc,best_result['punct']))
    matrix, vocab = vectorize(tokens, best_result['features'], best_result['min_freq'], vocab)
    return docs, labels, matrix


def print_top_misclassified(test_docs, test_labels, X_test, clf, n):
    """
    Print the n testing documents that are misclassified by the
    largest margin. By using the .predict_proba function of
    LogisticRegression <https://goo.gl/4WXbYA>, we can get the
    predicted probabilities of each class for each instance.
    We will first identify all incorrectly classified documents,
    then sort them in descending order of the predicted probability
    for the incorrect class.
    E.g., if document i is misclassified as positive, we will
    consider the probability of the positive class when sorting.
    Params:
      test_docs.....List of strings, one per test document
      test_labels...Array of true testing labels
      X_test........csr_matrix for test data
      clf...........LogisticRegression classifier fit on all training
                    data.
      n.............The number of documents to print.
    Returns:
      Nothing; see Log.txt for example printed output.
    """
    
    val=clf.predict_proba(X_test)
    predict= clf.predict(X_test)
    outpu = list()
    newlist = []
    qr = len(predict)
    for i in range(0,qr):
        d=dict()
        if predict[i] != test_labels[i]:
        
            d.update({'truth':test_labels[i]})
            d.update({'predicted':predict[i]})
            d.update({'proba': val[i]})
            d.update({'doc' : test_docs[i]})
            outpu.append(d)
    value_out = list()
    val1=sorted(outpu,key=lambda x:max(x['proba']),reverse=True)
    for i in val1[:n]:
        print("truth=%d predicted=%d proba=%.5f"%(i['truth'],i['predicted'],max(i['proba'])))
        print(i['doc'])

        
def main():
    """
    Put it all together.
    ALREADY DONE.
    """
    feature_fns = [token_features, token_pair_features, lexicon_features]
    # Download and read data.
    download_data()
    docs, labels = read_data(os.path.join('data', 'train'))
    # Evaluate accuracy of many combinations
    # of tokenization/featurization.
    results = eval_all_combinations(docs, labels,
                                    [True, False],
                                    feature_fns,
                                    [2,5,10])
    #print(results)
    # Print information about these results.
    best_result = results[0]
    worst_result = results[-1]
    print('best cross-validation result:\n%s' % str(best_result))
    print('worst cross-validation result:\n%s' % str(worst_result))
    plot_sorted_accuracies(results)
    print('\nMean Accuracies per Setting:')
    print('\n'.join(['%s: %.5f' % (s,v) for v,s in mean_accuracy_per_setting(results)]))

    # Fit best classifier.
    clf, vocab = fit_best_classifier(docs, labels, results[0])

    # Print top coefficients per class.
    print('\nTOP COEFFICIENTS PER CLASS:')
    print('negative words:')
    print('\n'.join(['%s: %.5f' % (t,v) for t,v in top_coefs(clf, 0, 5, vocab)]))
    print('\npositive words:')
    print('\n'.join(['%s: %.5f' % (t,v) for t,v in top_coefs(clf, 1, 5, vocab)]))

    # Parse test data
    test_docs, test_labels, X_test = parse_test_data(best_result, vocab)

    # Evaluate on test set.
    predictions = clf.predict(X_test)
    print('testing accuracy=%f' %          accuracy_score(test_labels, predictions))


    print('\nTOP MISCLASSIFIED TEST DOCUMENTS:')
    print_top_misclassified(test_docs, test_labels, X_test, clf, 5)


if __name__ == '__main__':
    main()





