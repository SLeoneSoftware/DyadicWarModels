import numpy, os.path
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from datetime import datetime
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import time

svm_best_parameters = dict()
rf_best_parameters = dict()
nn_best_parameters = dict()

svm_best_score = 0
rf_best_score = 0
nn_best_score = 0.7186067827681026

def get_test_train(fname,seed,datatype):
    data = numpy.genfromtxt(fname,delimiter=',',dtype=datatype)
    return data

def load_features(path='data'):
    return get_test_train(os.path.join(path,'war.csv'),seed=1567708903,datatype=float)

def get_data(seed):
    y = load_features()[:,[15]]
    x = load_features()[:,[ 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 ,16, 17, 18, 19, 20]]
    features = numpy.transpose(x)
    labels = numpy.transpose(y)
    new_array= numpy.vstack([features, labels])
    data = numpy.transpose(new_array)
    numpy.random.seed(seed)
    shuffled_idx = numpy.random.permutation(data.shape[0])
    cutoff = int(data.shape[0]*0.8)
    train_data = data[shuffled_idx[:cutoff]]
    test_data = data[shuffled_idx[cutoff:]]
    train_X = train_data[:,:-1].astype(float)
    train_Y = train_data[:,-1].reshape(-1,1)
    test_X = test_data[:,:-1].astype(float)
    test_Y = test_data[:,-1].reshape(-1,1)
    return train_X, train_Y, test_X, test_Y, data

def fold_data_one(train_X, train_Y):
    train_data_x = []
    train_data_y = []
    validate_data_x = []
    validate_data_y = []
    cv=StratifiedKFold(n_splits=3, shuffle=False)
    splits = cv.split(train_X, train_Y)
    vind = []
    j = 0
    for train,test in splits:
        if j == 0:
            vind = test
        j += 1
    for i in range(0, len(train_Y)):
        if i in vind:
            validate_data_y.append(train_Y[i][0])
            validate_data_x.append(train_X[i])
        else:
            train_data_y.append(train_Y[i][0])
            train_data_x.append(train_X[i])
    return train_data_x, train_data_y, validate_data_x, validate_data_y

def fold_data_two(train_X, train_Y):
    train_data_x = []
    train_data_y = []
    validate_data_x = []
    validate_data_y = []
    cv=StratifiedKFold(n_splits=3, shuffle=False)
    splits = cv.split(train_X, train_Y)
    vind = []
    j = 0
    for train,test in splits:
        if j == 1:
            vind = test
        j += 1
    for i in range(0, len(train_Y)):
        if i in vind:
            validate_data_y.append(train_Y[i][0])
            validate_data_x.append(train_X[i])
        else:
            train_data_y.append(train_Y[i][0])
            train_data_x.append(train_X[i])
    return train_data_x, train_data_y, validate_data_x, validate_data_y

def fold_data_three(train_X, train_Y):
    train_data_x = []
    train_data_y = []
    validate_data_x = []
    validate_data_y = []
    cv=StratifiedKFold(n_splits=3, shuffle=False)
    splits = cv.split(train_X, train_Y)
    vind = []
    j = 0
    for train,test in splits:
        if j == 0:
            vind = test
        j += 1
    for i in range(0, len(train_Y)):
        if i in vind:
            validate_data_y.append(train_Y[i][0])
            validate_data_x.append(train_X[i])
        else:
            train_data_y.append(train_Y[i][0])
            train_data_x.append(train_X[i])
    return train_data_x, train_data_y, validate_data_x, validate_data_y

def test_accuracy(predictions, test_Y):
    total = 0
    accurate = 0
    for i in range(0,len(predictions)):
        total += 1
        if predictions[i] == test_Y[i][0]:
            accurate += 1
    print(accurate/total)
    return accurate/total

def run_confusion_matrix(predictions, test_Y):
    cm = confusion_matrix(predictions, test_Y)
    print(cm)

#Grid Search
def run_grid_search(model, train_X, train_Y, param_dict, type):
    global svm_best_parameters
    global rf_best_parameters
    global nn_best_parameters
    global svm_best_score
    global rf_best_score
    global nn_best_score
    clf = GridSearchCV(model, param_dict, cv=StratifiedKFold(n_splits=3, shuffle=False))
    now = datetime. now()
    current_time = now. strftime("%H:%M:%S")
    print("Current Time =", current_time)
    clf.fit(train_X, train_Y)
    now = datetime. now()
    current_time = now. strftime("%H:%M:%S")
    print("Current Time =", current_time)
    print(clf.best_params_)
    if type == "svm":
        svm_best_parameters = clf.best_params_
        svm_best_score = clf.best_score_
    elif type == "rf":
        rf_best_parameters = clf.best_params_
        rf_best_score = clf.best_score_
    elif type == "nn":
        nn_best_parameters = clf.best_params_
        nn_best_score = clf.best_score_
    print(clf.best_score_)
    #print(clf.cv.split(train_X, train_Y))
    #for train, test in clf.cv.split(train_X, train_Y):
    #    print('TRAIN: ', train, ' VALIDATION: ', test)
    #print(clf.cv_results_)
    return clf.best_score_

#Results: {'C': 0.8, 'coef0': 0, 'decision_function_shape': 'ovo', 'gamma': 'auto', 'kernel': 'rbf'}
#0.84967919340055
def grid_search_svm():
    seed = 1567708903
    train_X, train_Y, test_X, test_Y, data = get_data(seed)
    C = [.0001, .001, .01, .1, .2, .4, .6, .8, 1]
    kernel = ['rbf', 'sigmoid']
    gamma = ['auto', 'scale']
    coef0 = [0, .2, .4, .6, .8, 1]
    decision_function_shape = ['ovo', 'ovr']
    param_dict = dict(C = C, kernel = kernel, gamma = gamma, coef0 = coef0, decision_function_shape = decision_function_shape)
    print("SVM GRID SEARCH:")
    run_grid_search(svm.SVC(), train_X, train_Y, param_dict, "svm")

#Results: {'activation': 'tanh', 'alpha': 0.2, 'hidden_layer_sizes': (100, 100, 100, 100, 100, 100), 'learning_rate': 'constant', 'random_state': 1567708903, 'solver': 'adam', 'tol': 1e-05}
#0.7186067827681026
def grid_search_neural_network():
    seed = 1567708903
    train_X, train_Y, test_X, test_Y, data = get_data(seed)
    random_state = [1567708903]
    alpha = [.00001, .0001, .001, .01, .1, .2, .4, .6, .8]
    solver = ['lbfgs', 'sgd', 'adam']
    activation = ['identity', 'logistic', 'tanh', 'relu']
    tol = [.00001, .0001, .001, .01, .1]
    hidden_layer_sizes = [(100,100, 100), (100,100, 100, 100), (100,100, 100, 100, 100),(100,100, 100, 100, 100, 100) ,(100,100, 100, 100, 100, 100, 100), (100,100, 100, 100, 100, 100, 100, 100),(100,100, 100, 100, 100, 100, 100, 100, 100), (100,100, 100, 100, 100, 100, 100, 100, 100, 100), (10,10, 10), (10,10, 10, 10), (10,10, 10, 10, 10),(10,10, 10, 10, 10, 10) ,(10,10, 10, 10, 10, 10, 10), (10,10, 10, 10, 10, 10, 10, 10),(10,10, 10, 10, 10, 10, 10, 10, 10), (10,10, 10, 10, 10, 10, 10, 10, 10, 10)]
    learning_rate = ['constant', 'invscaling', 'adaptive']
    param_dict = dict(hidden_layer_sizes = hidden_layer_sizes,activation=activation,random_state = random_state, alpha = alpha, solver = solver, tol = tol, learning_rate = learning_rate )
    print("Neural Network GRID SEARCH:")
    run_grid_search(MLPClassifier(), train_X, train_Y, param_dict, "nn")

#Results: {'criterion': 'gini', 'max_depth': 10, 'max_features': 'auto', 'min_impurity_decrease': 0, 'n_estimators': 25, 'random_state': 1567708903}
#0.9963369963369964
def grid_search_rf():
    seed = 1567708903
    train_X, train_Y, test_X, test_Y, data = get_data(seed)
    n_estimators = [10, 25, 50, 75, 100]
    criterion = ['gini', 'entropy']
    max_depth = [5, 10, 25, None]
    random_state = [seed]
    min_impurity_decrease = [0, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5 , 1e-4, 1e-3, 1e-2, 1e-1, 1e-0]
    max_features = ['auto', 'sqrt', 'log2']
    param_dict = dict(n_estimators = n_estimators, criterion = criterion, max_depth = max_depth, max_features = max_features, random_state = random_state, min_impurity_decrease = min_impurity_decrease)
    print("RF GRID SEARCH:")
    run_grid_search(RandomForestClassifier(), train_X, train_Y, param_dict, "rf")

def test_linearly_seperable():
    global svm_best_score
    global rf_best_score
    global nn_best_score
    seed = 1567708903
    train_X, train_Y, test_X, test_Y, data = get_data(seed)
    tol = [.00001, .0001, .001, .01, .1]
    C = [.0001, .001, .01, .1, .2, .4, .6, .8, 1]
    param_dict = dict(C = C, tol = tol)
    print("Logistic Regression Parameters and Accuracy:")
    log_score = run_grid_search(LogisticRegression(solver='lbfgs', multi_class='multinomial'), train_X, train_Y, param_dict, "log_reg")
    svm_score = svm_best_score
    nn_score = nn_best_score
    rf_score = rf_best_score
    plt.style.use('ggplot')
    x = ['Log Reg', 'SVM', 'NN', 'RF',]
    scores = [log_score, svm_score, nn_score, rf_score]
    x_pos = [i for i, _ in enumerate(x)]
    plt.bar(x_pos, scores, color='blue')
    plt.xlabel("Algorithm")
    plt.ylabel("Max Accuracy after Grid Search")
    plt.title("Non-linear vs linear models")
    plt.xticks(x_pos, x)
    plt.show()

def graph_three_way_tuning(alpha, tol, accuracy, xtitle, ytitle, title):
    import matplotlib.pyplot
    from mpl_toolkits.mplot3d import Axes3D
    fig = matplotlib.pyplot.figure()
    fig.gca(projection='3d').scatter(alpha,tol,accuracy ,color='r')
    matplotlib.pyplot.xlabel(xtitle)
    matplotlib.pyplot.ylabel(ytitle)
    fig.gca().set_title(title)
    matplotlib.pyplot.show()

def graph_two_way_tuning(alpha, accuracy, xtitle, ytitle, title):
    import matplotlib.pyplot
    from mpl_toolkits.mplot3d import Axes3D
    fig = matplotlib.pyplot.figure()
    fig.gca().scatter(alpha, accuracy ,color='r')
    matplotlib.pyplot.xlabel(xtitle)
    matplotlib.pyplot.ylabel(ytitle)
    fig.gca().set_title(title)
    matplotlib.pyplot.show()

#Unused in final experiment. Further divides train and test data
def split_up(x, y):
    data = []
    for i in range(0,len(x)):
        row = numpy.asarray(x[i])
        numpy.append(row, y[i][0])
        data.append(row)
    data = numpy.array(data)
    shuffled_idx = numpy.random.permutation(data.shape[0])
    cutoff = int(data.shape[0]*0.8)
    train_data = data[shuffled_idx[:cutoff]]
    test_data = data[shuffled_idx[cutoff:]]
    train_X = train_data[:,:-1].astype(float)
    train_Y = train_data[:,-1].reshape(-1,1)
    test_X = test_data[:,:-1].astype(float)
    test_Y = test_data[:,-1].reshape(-1,1)
    return train_X, train_Y, test_X, test_Y

def tune_all():
    tune_rf()
    tune_svm()
    tune_neural_network_at()
    tune_neural_network_ha()

#Original Hyperparameters: {'activation': 'tanh', 'alpha': 0.2, 'hidden_layer_sizes': (100, 100, 100, 100, 100, 100), 'learning_rate': 'constant', 'random_state': 1567708903, 'solver': 'adam', 'tol': 1e-05}
#Final Hyperparameters: {'activation': 'tanh', 'alpha': 0.18, 'hidden_layer_sizes': (100, 100, 100, 100, 100, 100), 'learning_rate': 'constant', 'random_state': 1567708903, 'solver': 'adam', 'tol': 1e-05}
def tune_neural_network_ha():
    tune_neural_network_ha_one()
    tune_neural_network_ha_two()
    tune_neural_network_ha_three()


def tune_neural_network_ha_one():
    global nn_best_parameters
    seed = 1567708903
    numpy.random.seed(seed)
    train_X, train_Y, junkOne, junkTwo, data = get_data(seed)
    train_X, train_Y, test_X, test_Y = fold_data_one(train_X, train_Y)
    hls = nn_best_parameters['hidden_layer_sizes'][0]
    hidden_layer_sizes = [hls - 30, hls - 30, hls - 30, hls - 30, hls - 30, hls - 30, hls - 30, hls - 30,     hls - 20, hls - 20, hls - 20, hls - 20, hls - 20, hls - 20, hls - 20, hls - 20,     hls - 10, hls - 10, hls - 10, hls - 10, hls - 10, hls - 10, hls - 10, hls - 10,       hls, hls, hls, hls, hls, hls, hls, hls,      hls + 10, hls + 10, hls + 10, hls + 10, hls + 10, hls + 10, hls + 10, hls + 10, hls + 20, hls + 20, hls + 20, hls + 20, hls + 20, hls + 20, hls + 20, hls + 20, hls + 10, hls + 10, hls + 10, hls + 10, hls + 10, hls + 10, hls + 10, hls + 10]
    alphas = [nn_best_parameters['alpha'] - 0.05, nn_best_parameters['alpha'] - 0.02, nn_best_parameters['alpha'], nn_best_parameters['alpha'] + 0.03, nn_best_parameters['alpha'] + 0.05, nn_best_parameters['alpha'] + 0.08, nn_best_parameters['alpha'] + 0.1, nn_best_parameters['alpha'] + 0.13,      nn_best_parameters['alpha'] - 0.05, nn_best_parameters['alpha'] - 0.02, nn_best_parameters['alpha'] , nn_best_parameters['alpha'] + 0.03, nn_best_parameters['alpha'] + 0.05, nn_best_parameters['alpha'] + 0.08, nn_best_parameters['alpha'] + 0.1, nn_best_parameters['alpha'] + 0.13,        nn_best_parameters['alpha'] - 0.05, nn_best_parameters['alpha'] - 0.02, nn_best_parameters['alpha'] , nn_best_parameters['alpha'] + 0.03, nn_best_parameters['alpha'] + 0.05, nn_best_parameters['alpha'] + 0.08, nn_best_parameters['alpha'] + 0.1, nn_best_parameters['alpha'] + 0.13, nn_best_parameters['alpha'] - 0.05, nn_best_parameters['alpha'] - 0.02, nn_best_parameters['alpha'] , nn_best_parameters['alpha'] + 0.03, nn_best_parameters['alpha'] + 0.05, nn_best_parameters['alpha'] + 0.08, nn_best_parameters['alpha'] + 0.1, nn_best_parameters['alpha'] + 0.13,            nn_best_parameters['alpha'] - 0.05, nn_best_parameters['alpha'] - 0.02, nn_best_parameters['alpha'] , nn_best_parameters['alpha'] + 0.03, nn_best_parameters['alpha'] + 0.05, nn_best_parameters['alpha'] + 0.08, nn_best_parameters['alpha'] + 0.1, nn_best_parameters['alpha'] + 0.13,      nn_best_parameters['alpha'] - 0.05, nn_best_parameters['alpha'] - 0.02, nn_best_parameters['alpha'] , nn_best_parameters['alpha'] + 0.03, nn_best_parameters['alpha'] + 0.05, nn_best_parameters['alpha'] + 0.08, nn_best_parameters['alpha'] + 0.1, nn_best_parameters['alpha'] + 0.13,        nn_best_parameters['alpha'] - 0.05, nn_best_parameters['alpha'] - 0.02, nn_best_parameters['alpha'] , nn_best_parameters['alpha'] + 0.03, nn_best_parameters['alpha'] + 0.05, nn_best_parameters['alpha'] + 0.08, nn_best_parameters['alpha'] + 0.1, nn_best_parameters['alpha'] + 0.13]
    accuracy = []
    for j in range(0, len(alphas)):
        alpha = alphas[j]
        hls = hidden_layer_sizes[j]
        clf = MLPClassifier(solver=nn_best_parameters['solver'], alpha=alpha, hidden_layer_sizes=(hls, hls, hls, hls, hls, hls), random_state=seed, tol=nn_best_parameters['tol'],activation=nn_best_parameters['activation'],learning_rate=nn_best_parameters['learning_rate']).fit(train_X, train_Y)
        predictions = clf.predict(test_X)
        accurate = 0
        total = 0
        for i in range(0,len(predictions)):
            total += 1
            if predictions[i] == test_Y[i]:
                accurate += 1
        accuracy.append(accurate/total)
    graph_three_way_tuning(alphas, hidden_layer_sizes, accuracy, "Alpha", "Hidden Layer Neurons", "Fold 1: Alpha vs Hidden Layer Neurons vs Accuracy")

def tune_neural_network_ha_two():
    global nn_best_parameters
    seed = 1567708903
    numpy.random.seed(seed)
    train_X, train_Y, junkOne, junkTwo, data = get_data(seed)
    train_X, train_Y, test_X, test_Y = fold_data_two(train_X, train_Y)
    hls = nn_best_parameters['hidden_layer_sizes'][0]
    hidden_layer_sizes = [hls - 30, hls - 30, hls - 30, hls - 30, hls - 30, hls - 30, hls - 30, hls - 30,     hls - 20, hls - 20, hls - 20, hls - 20, hls - 20, hls - 20, hls - 20, hls - 20,     hls - 10, hls - 10, hls - 10, hls - 10, hls - 10, hls - 10, hls - 10, hls - 10,       hls, hls, hls, hls, hls, hls, hls, hls,      hls + 10, hls + 10, hls + 10, hls + 10, hls + 10, hls + 10, hls + 10, hls + 10, hls + 20, hls + 20, hls + 20, hls + 20, hls + 20, hls + 20, hls + 20, hls + 20, hls + 10, hls + 10, hls + 10, hls + 10, hls + 10, hls + 10, hls + 10, hls + 10]
    alphas = [nn_best_parameters['alpha'] - 0.05, nn_best_parameters['alpha'] - 0.02, nn_best_parameters['alpha'], nn_best_parameters['alpha'] + 0.03, nn_best_parameters['alpha'] + 0.05, nn_best_parameters['alpha'] + 0.08, nn_best_parameters['alpha'] + 0.1, nn_best_parameters['alpha'] + 0.13,      nn_best_parameters['alpha'] - 0.05, nn_best_parameters['alpha'] - 0.02, nn_best_parameters['alpha'] , nn_best_parameters['alpha'] + 0.03, nn_best_parameters['alpha'] + 0.05, nn_best_parameters['alpha'] + 0.08, nn_best_parameters['alpha'] + 0.1, nn_best_parameters['alpha'] + 0.13,        nn_best_parameters['alpha'] - 0.05, nn_best_parameters['alpha'] - 0.02, nn_best_parameters['alpha'] , nn_best_parameters['alpha'] + 0.03, nn_best_parameters['alpha'] + 0.05, nn_best_parameters['alpha'] + 0.08, nn_best_parameters['alpha'] + 0.1, nn_best_parameters['alpha'] + 0.13, nn_best_parameters['alpha'] - 0.05, nn_best_parameters['alpha'] - 0.02, nn_best_parameters['alpha'] , nn_best_parameters['alpha'] + 0.03, nn_best_parameters['alpha'] + 0.05, nn_best_parameters['alpha'] + 0.08, nn_best_parameters['alpha'] + 0.1, nn_best_parameters['alpha'] + 0.13,            nn_best_parameters['alpha'] - 0.05, nn_best_parameters['alpha'] - 0.02, nn_best_parameters['alpha'] , nn_best_parameters['alpha'] + 0.03, nn_best_parameters['alpha'] + 0.05, nn_best_parameters['alpha'] + 0.08, nn_best_parameters['alpha'] + 0.1, nn_best_parameters['alpha'] + 0.13,      nn_best_parameters['alpha'] - 0.05, nn_best_parameters['alpha'] - 0.02, nn_best_parameters['alpha'] , nn_best_parameters['alpha'] + 0.03, nn_best_parameters['alpha'] + 0.05, nn_best_parameters['alpha'] + 0.08, nn_best_parameters['alpha'] + 0.1, nn_best_parameters['alpha'] + 0.13,        nn_best_parameters['alpha'] - 0.05, nn_best_parameters['alpha'] - 0.02, nn_best_parameters['alpha'] , nn_best_parameters['alpha'] + 0.03, nn_best_parameters['alpha'] + 0.05, nn_best_parameters['alpha'] + 0.08, nn_best_parameters['alpha'] + 0.1, nn_best_parameters['alpha'] + 0.13]
    accuracy = []
    for j in range(0, len(alphas)):
        alpha = alphas[j]
        hls = hidden_layer_sizes[j]
        clf = MLPClassifier(solver=nn_best_parameters['solver'], alpha=alpha, hidden_layer_sizes=(hls, hls, hls, hls, hls, hls), random_state=seed, tol=nn_best_parameters['tol'],activation=nn_best_parameters['activation'],learning_rate=nn_best_parameters['learning_rate']).fit(train_X, train_Y)
        predictions = clf.predict(test_X)
        accurate = 0
        total = 0
        for i in range(0,len(predictions)):
            total += 1
            if predictions[i] == test_Y[i]:
                accurate += 1
        accuracy.append(accurate/total)
    graph_three_way_tuning(alphas, hidden_layer_sizes, accuracy, "Alpha", "Hidden Layer Neurons", "Fold 2: Alpha vs Hidden Layer Neurons vs Accuracy")

def tune_neural_network_ha_three():
    global nn_best_parameters
    seed = 1567708903
    numpy.random.seed(seed)
    train_X, train_Y, junkOne, junkTwo, data = get_data(seed)
    train_X, train_Y, test_X, test_Y = fold_data_three(train_X, train_Y)
    hls = nn_best_parameters['hidden_layer_sizes'][0]
    hidden_layer_sizes = [hls - 30, hls - 30, hls - 30, hls - 30, hls - 30, hls - 30, hls - 30, hls - 30,     hls - 20, hls - 20, hls - 20, hls - 20, hls - 20, hls - 20, hls - 20, hls - 20,     hls - 10, hls - 10, hls - 10, hls - 10, hls - 10, hls - 10, hls - 10, hls - 10,       hls, hls, hls, hls, hls, hls, hls, hls,      hls + 10, hls + 10, hls + 10, hls + 10, hls + 10, hls + 10, hls + 10, hls + 10, hls + 20, hls + 20, hls + 20, hls + 20, hls + 20, hls + 20, hls + 20, hls + 20, hls + 10, hls + 10, hls + 10, hls + 10, hls + 10, hls + 10, hls + 10, hls + 10]
    alphas = [nn_best_parameters['alpha'] - 0.05, nn_best_parameters['alpha'] - 0.02, nn_best_parameters['alpha'], nn_best_parameters['alpha'] + 0.03, nn_best_parameters['alpha'] + 0.05, nn_best_parameters['alpha'] + 0.08, nn_best_parameters['alpha'] + 0.1, nn_best_parameters['alpha'] + 0.13,      nn_best_parameters['alpha'] - 0.05, nn_best_parameters['alpha'] - 0.02, nn_best_parameters['alpha'] , nn_best_parameters['alpha'] + 0.03, nn_best_parameters['alpha'] + 0.05, nn_best_parameters['alpha'] + 0.08, nn_best_parameters['alpha'] + 0.1, nn_best_parameters['alpha'] + 0.13,        nn_best_parameters['alpha'] - 0.05, nn_best_parameters['alpha'] - 0.02, nn_best_parameters['alpha'] , nn_best_parameters['alpha'] + 0.03, nn_best_parameters['alpha'] + 0.05, nn_best_parameters['alpha'] + 0.08, nn_best_parameters['alpha'] + 0.1, nn_best_parameters['alpha'] + 0.13, nn_best_parameters['alpha'] - 0.05, nn_best_parameters['alpha'] - 0.02, nn_best_parameters['alpha'] , nn_best_parameters['alpha'] + 0.03, nn_best_parameters['alpha'] + 0.05, nn_best_parameters['alpha'] + 0.08, nn_best_parameters['alpha'] + 0.1, nn_best_parameters['alpha'] + 0.13,            nn_best_parameters['alpha'] - 0.05, nn_best_parameters['alpha'] - 0.02, nn_best_parameters['alpha'] , nn_best_parameters['alpha'] + 0.03, nn_best_parameters['alpha'] + 0.05, nn_best_parameters['alpha'] + 0.08, nn_best_parameters['alpha'] + 0.1, nn_best_parameters['alpha'] + 0.13,      nn_best_parameters['alpha'] - 0.05, nn_best_parameters['alpha'] - 0.02, nn_best_parameters['alpha'] , nn_best_parameters['alpha'] + 0.03, nn_best_parameters['alpha'] + 0.05, nn_best_parameters['alpha'] + 0.08, nn_best_parameters['alpha'] + 0.1, nn_best_parameters['alpha'] + 0.13,        nn_best_parameters['alpha'] - 0.05, nn_best_parameters['alpha'] - 0.02, nn_best_parameters['alpha'] , nn_best_parameters['alpha'] + 0.03, nn_best_parameters['alpha'] + 0.05, nn_best_parameters['alpha'] + 0.08, nn_best_parameters['alpha'] + 0.1, nn_best_parameters['alpha'] + 0.13]
    accuracy = []
    for j in range(0, len(alphas)):
        alpha = alphas[j]
        hls = hidden_layer_sizes[j]
        clf = MLPClassifier(solver=nn_best_parameters['solver'], alpha=alpha, hidden_layer_sizes=(hls, hls, hls, hls, hls, hls), random_state=seed, tol=nn_best_parameters['tol'],activation=nn_best_parameters['activation'],learning_rate=nn_best_parameters['learning_rate']).fit(train_X, train_Y)
        predictions = clf.predict(test_X)
        accurate = 0
        total = 0
        for i in range(0,len(predictions)):
            total += 1
            if predictions[i] == test_Y[i]:
                accurate += 1
        accuracy.append(accurate/total)
    graph_three_way_tuning(alphas, hidden_layer_sizes, accuracy, "Alpha", "Hidden Layer Neurons", "Fold 3: Alpha vs Hidden Layer Neurons vs Accuracy")

def tune_neural_network_at():
    global nn_best_parameters
    tune_neural_network_at_one()
    tune_neural_network_at_two()
    tune_neural_network_at_three()
    #The reasons for changing alpha is stated in the report.pdf
    nn_best_parameters['alpha'] = nn_best_parameters['alpha'] - .02


def tune_neural_network_at_one():
    global nn_best_parameters
    seed = 1567708903
    numpy.random.seed(seed)
    train_X, train_Y, junkOne, junkTwo, data = get_data(seed)
    train_X, train_Y, test_X, test_Y = fold_data_one(train_X, train_Y)
    alphas = [nn_best_parameters['alpha'] - 0.05, nn_best_parameters['alpha'] - 0.02, nn_best_parameters['alpha'] , nn_best_parameters['alpha'] + 0.03, nn_best_parameters['alpha'] + 0.05, nn_best_parameters['alpha'] + 0.08, nn_best_parameters['alpha'] + 0.1, nn_best_parameters['alpha'] + 0.13,            nn_best_parameters['alpha'] - 0.05, nn_best_parameters['alpha'] - 0.02, nn_best_parameters['alpha'], nn_best_parameters['alpha'] + 0.03, nn_best_parameters['alpha'] + 0.05, nn_best_parameters['alpha'] + 0.08, nn_best_parameters['alpha'] + 0.1, nn_best_parameters['alpha'] + 0.13,      nn_best_parameters['alpha'] - 0.05, nn_best_parameters['alpha'] - 0.02, nn_best_parameters['alpha'], nn_best_parameters['alpha'] + 0.03, nn_best_parameters['alpha'] + 0.05, nn_best_parameters['alpha'] + 0.08, nn_best_parameters['alpha'] + 0.1, nn_best_parameters['alpha'] + 0.13,        nn_best_parameters['alpha'] - 0.05, nn_best_parameters['alpha'] - 0.02, nn_best_parameters['alpha'] , nn_best_parameters['alpha'] + 0.03, nn_best_parameters['alpha'] + 0.05, nn_best_parameters['alpha'] + 0.08, nn_best_parameters['alpha'] + 0.1, nn_best_parameters['alpha'] + 0.13]
    tols = [nn_best_parameters['tol'] - .000005,nn_best_parameters['tol'] - .000005,nn_best_parameters['tol'] - .000005,nn_best_parameters['tol'] - .000005,nn_best_parameters['tol'] - .000005,nn_best_parameters['tol'] - .000005,nn_best_parameters['tol'] - .000005,nn_best_parameters['tol'] - .000005,                   nn_best_parameters['tol'] ,nn_best_parameters['tol'] ,nn_best_parameters['tol'],nn_best_parameters['tol'],nn_best_parameters['tol'],nn_best_parameters['tol'],nn_best_parameters['tol'],nn_best_parameters['tol'],               nn_best_parameters['tol'] + .000005,nn_best_parameters['tol'] + .000005,nn_best_parameters['tol'] + .000005,nn_best_parameters['tol'] + .000005,nn_best_parameters['tol'] + .000005,nn_best_parameters['tol'] + .000005,nn_best_parameters['tol'] + .000005,nn_best_parameters['tol'] + .000005,                         nn_best_parameters['tol'] + .00001,nn_best_parameters['tol'] + .00001,nn_best_parameters['tol'] + .00001,nn_best_parameters['tol'] + .00001,nn_best_parameters['tol'] + .00001,nn_best_parameters['tol'] + .00001,nn_best_parameters['tol'] + .00001,nn_best_parameters['tol'] + .00001  ]
    accuracy = []
    for j in range(0, len(alphas)):
        alpha = alphas[j]
        tol = tols[j]
        clf = MLPClassifier(solver=nn_best_parameters['solver'], alpha=alpha, hidden_layer_sizes=nn_best_parameters['hidden_layer_sizes'], random_state=seed, tol=tol,activation=nn_best_parameters['activation'],learning_rate=nn_best_parameters['learning_rate']).fit(train_X, train_Y)
        predictions = clf.predict(test_X)
        accurate = 0
        total = 0
        for i in range(0,len(predictions)):
            total += 1
            if predictions[i] == test_Y[i]:
                accurate += 1
        accuracy.append(accurate/total)
    graph_three_way_tuning(alphas, tols, accuracy, "Alpha", "Tol", "Fold 1: Alpha vs Tol vs Validation Accuracy")


def tune_neural_network_at_two():
    global nn_best_parameters
    seed = 1567708903
    numpy.random.seed(seed)
    train_X, train_Y, junkOne, junkTwo, data = get_data(seed)
    train_X, train_Y, test_X, test_Y = fold_data_two(train_X, train_Y)
    alphas = [nn_best_parameters['alpha'] - 0.05, nn_best_parameters['alpha'] - 0.02, nn_best_parameters['alpha'] , nn_best_parameters['alpha'] + 0.03, nn_best_parameters['alpha'] + 0.05, nn_best_parameters['alpha'] + 0.08, nn_best_parameters['alpha'] + 0.1, nn_best_parameters['alpha'] + 0.13,            nn_best_parameters['alpha'] - 0.05, nn_best_parameters['alpha'] - 0.02, nn_best_parameters['alpha'], nn_best_parameters['alpha'] + 0.03, nn_best_parameters['alpha'] + 0.05, nn_best_parameters['alpha'] + 0.08, nn_best_parameters['alpha'] + 0.1, nn_best_parameters['alpha'] + 0.13,      nn_best_parameters['alpha'] - 0.05, nn_best_parameters['alpha'] - 0.02, nn_best_parameters['alpha'], nn_best_parameters['alpha'] + 0.03, nn_best_parameters['alpha'] + 0.05, nn_best_parameters['alpha'] + 0.08, nn_best_parameters['alpha'] + 0.1, nn_best_parameters['alpha'] + 0.13,        nn_best_parameters['alpha'] - 0.05, nn_best_parameters['alpha'] - 0.02, nn_best_parameters['alpha'] , nn_best_parameters['alpha'] + 0.03, nn_best_parameters['alpha'] + 0.05, nn_best_parameters['alpha'] + 0.08, nn_best_parameters['alpha'] + 0.1, nn_best_parameters['alpha'] + 0.13]
    tols = [nn_best_parameters['tol'] - .000005,nn_best_parameters['tol'] - .000005,nn_best_parameters['tol'] - .000005,nn_best_parameters['tol'] - .000005,nn_best_parameters['tol'] - .000005,nn_best_parameters['tol'] - .000005,nn_best_parameters['tol'] - .000005,nn_best_parameters['tol'] - .000005,                   nn_best_parameters['tol'] ,nn_best_parameters['tol'] ,nn_best_parameters['tol'],nn_best_parameters['tol'],nn_best_parameters['tol'],nn_best_parameters['tol'],nn_best_parameters['tol'],nn_best_parameters['tol'],               nn_best_parameters['tol'] + .000005,nn_best_parameters['tol'] + .000005,nn_best_parameters['tol'] + .000005,nn_best_parameters['tol'] + .000005,nn_best_parameters['tol'] + .000005,nn_best_parameters['tol'] + .000005,nn_best_parameters['tol'] + .000005,nn_best_parameters['tol'] + .000005,                         nn_best_parameters['tol'] + .00001,nn_best_parameters['tol'] + .00001,nn_best_parameters['tol'] + .00001,nn_best_parameters['tol'] + .00001,nn_best_parameters['tol'] + .00001,nn_best_parameters['tol'] + .00001,nn_best_parameters['tol'] + .00001,nn_best_parameters['tol'] + .00001  ]
    accuracy = []
    for j in range(0, len(alphas)):
        alpha = alphas[j]
        tol = tols[j]
        clf = MLPClassifier(solver=nn_best_parameters['solver'], alpha=alpha, hidden_layer_sizes=nn_best_parameters['hidden_layer_sizes'], random_state=seed, tol=tol,activation=nn_best_parameters['activation'],learning_rate=nn_best_parameters['learning_rate']).fit(train_X, train_Y)
        predictions = clf.predict(test_X)
        accurate = 0
        total = 0
        for i in range(0,len(predictions)):
            total += 1
            if predictions[i] == test_Y[i]:
                accurate += 1
        accuracy.append(accurate/total)
    graph_three_way_tuning(alphas, tols, accuracy, "Alpha", "Tol", "Fold 2: Alpha vs Tol vs Validation Accuracy")

def tune_neural_network_at_three():
    global nn_best_parameters
    seed = 1567708903
    numpy.random.seed(seed)
    train_X, train_Y, junkOne, junkTwo, data = get_data(seed)
    train_X, train_Y, test_X, test_Y = fold_data_three(train_X, train_Y)
    alphas = [nn_best_parameters['alpha'] - 0.05, nn_best_parameters['alpha'] - 0.02, nn_best_parameters['alpha'] , nn_best_parameters['alpha'] + 0.03, nn_best_parameters['alpha'] + 0.05, nn_best_parameters['alpha'] + 0.08, nn_best_parameters['alpha'] + 0.1, nn_best_parameters['alpha'] + 0.13,            nn_best_parameters['alpha'] - 0.05, nn_best_parameters['alpha'] - 0.02, nn_best_parameters['alpha'], nn_best_parameters['alpha'] + 0.03, nn_best_parameters['alpha'] + 0.05, nn_best_parameters['alpha'] + 0.08, nn_best_parameters['alpha'] + 0.1, nn_best_parameters['alpha'] + 0.13,      nn_best_parameters['alpha'] - 0.05, nn_best_parameters['alpha'] - 0.02, nn_best_parameters['alpha'], nn_best_parameters['alpha'] + 0.03, nn_best_parameters['alpha'] + 0.05, nn_best_parameters['alpha'] + 0.08, nn_best_parameters['alpha'] + 0.1, nn_best_parameters['alpha'] + 0.13,        nn_best_parameters['alpha'] - 0.05, nn_best_parameters['alpha'] - 0.02, nn_best_parameters['alpha'] , nn_best_parameters['alpha'] + 0.03, nn_best_parameters['alpha'] + 0.05, nn_best_parameters['alpha'] + 0.08, nn_best_parameters['alpha'] + 0.1, nn_best_parameters['alpha'] + 0.13]
    tols = [nn_best_parameters['tol'] - .000005,nn_best_parameters['tol'] - .000005,nn_best_parameters['tol'] - .000005,nn_best_parameters['tol'] - .000005,nn_best_parameters['tol'] - .000005,nn_best_parameters['tol'] - .000005,nn_best_parameters['tol'] - .000005,nn_best_parameters['tol'] - .000005,                   nn_best_parameters['tol'] ,nn_best_parameters['tol'] ,nn_best_parameters['tol'],nn_best_parameters['tol'],nn_best_parameters['tol'],nn_best_parameters['tol'],nn_best_parameters['tol'],nn_best_parameters['tol'],               nn_best_parameters['tol'] + .000005,nn_best_parameters['tol'] + .000005,nn_best_parameters['tol'] + .000005,nn_best_parameters['tol'] + .000005,nn_best_parameters['tol'] + .000005,nn_best_parameters['tol'] + .000005,nn_best_parameters['tol'] + .000005,nn_best_parameters['tol'] + .000005,                         nn_best_parameters['tol'] + .00001,nn_best_parameters['tol'] + .00001,nn_best_parameters['tol'] + .00001,nn_best_parameters['tol'] + .00001,nn_best_parameters['tol'] + .00001,nn_best_parameters['tol'] + .00001,nn_best_parameters['tol'] + .00001,nn_best_parameters['tol'] + .00001  ]
    accuracy = []
    for j in range(0, len(alphas)):
        alpha = alphas[j]
        tol = tols[j]
        clf = MLPClassifier(solver=nn_best_parameters['solver'], alpha=alpha, hidden_layer_sizes=nn_best_parameters['hidden_layer_sizes'], random_state=seed, tol=tol,activation=nn_best_parameters['activation'],learning_rate=nn_best_parameters['learning_rate']).fit(train_X, train_Y)
        predictions = clf.predict(test_X)
        accurate = 0
        total = 0
        for i in range(0,len(predictions)):
            total += 1
            if predictions[i] == test_Y[i]:
                accurate += 1
        accuracy.append(accurate/total)
    graph_three_way_tuning(alphas, tols, accuracy, "Alpha", "Tol", "Fold 3: Alpha vs Tol vs Validation Accuracy")

#Original Hyperparameters: {'C': 0.8, 'decision_function_shape': 'ovo', 'gamma': 'auto', 'kernel': 'rbf'}
#Final Hyperparameters: {'C': 0.75, 'decision_function_shape': 'ovo', 'gamma': 'auto', 'kernel': 'rbf'}
def tune_svm():
    global svm_best_parameters
    tune_svm_one()
    tune_svm_two()
    tune_svm_three()
    #The change for this parameter is explained/reasoned in the report pdf
    svm_best_parameters['C'] = svm_best_parameters['C'] - 0.05

def tune_svm_one():
    global svm_best_parameters
    seed = 1567708903
    numpy.random.seed(seed)
    train_X, train_Y, test_X, test_Y, data = get_data(seed)
    train_X, train_Y, test_X, test_Y = fold_data_one(train_X, train_Y)
    bestc = svm_best_parameters['C']
    cs = [bestc - .15, bestc - .1, bestc - .05, bestc, bestc + .05, bestc + .1, bestc + .15]
    accuracy = []
    tol = .00001
    for j in range(0, len(cs)):
        c = cs[j]
        clf = svm.SVC(kernel = 'rbf', gamma = 'auto', decision_function_shape = 'ovo', C = c).fit(train_X, train_Y)
        predictions = clf.predict(test_X)
        accurate = 0
        total = 0
        for i in range(0,len(predictions)):
            total += 1
            if predictions[i] == test_Y[i]:
                accurate += 1
        accuracy.append(accurate/total)
    graph_two_way_tuning(cs, accuracy, "C", "Accuracy", "Fold 1: C vs Accuracy")

def tune_svm_two():
    global svm_best_parameters
    seed = 1567708903
    numpy.random.seed(seed)
    train_X, train_Y, test_X, test_Y, data = get_data(seed)
    train_X, train_Y, test_X, test_Y = fold_data_two(train_X, train_Y)
    bestc = svm_best_parameters['C']
    cs = [bestc - .15, bestc - .1, bestc - .05, bestc, bestc + .05, bestc + .1, bestc + .15]
    accuracy = []
    tol = .00001
    for j in range(0, len(cs)):
        c = cs[j]
        clf = svm.SVC(kernel = 'rbf', gamma = 'auto', decision_function_shape = 'ovo', C = c).fit(train_X, train_Y)
        predictions = clf.predict(test_X)
        accurate = 0
        total = 0
        for i in range(0,len(predictions)):
            total += 1
            if predictions[i] == test_Y[i]:
                accurate += 1
        accuracy.append(accurate/total)
    graph_two_way_tuning(cs, accuracy, "C", "Accuracy", "Fold 2: C vs Accuracy")

def tune_svm_three():
    global svm_best_parameters
    seed = 1567708903
    numpy.random.seed(seed)
    train_X, train_Y, test_X, test_Y, data = get_data(seed)
    train_X, train_Y, test_X, test_Y = fold_data_three(train_X, train_Y)
    bestc = svm_best_parameters['C']
    cs = [bestc - .15, bestc - .1, bestc - .05, bestc, bestc + .05, bestc + .1, bestc + .15]
    accuracy = []
    tol = .00001
    for j in range(0, len(cs)):
        c = cs[j]
        clf = svm.SVC(kernel = 'rbf', gamma = 'auto', decision_function_shape = 'ovo', C = c).fit(train_X, train_Y)
        predictions = clf.predict(test_X)
        accurate = 0
        total = 0
        for i in range(0,len(predictions)):
            total += 1
            if predictions[i] == test_Y[i]:
                accurate += 1
        accuracy.append(accurate/total)
    graph_two_way_tuning(cs, accuracy, "C", "Accuracy", "Fold 3: C vs Accuracy")

#Original Hyperparameters: {'criterion': 'gini', 'max_depth': 10, 'max_features': 'auto', 'min_impurity_decrease': 0, 'n_estimators': 25}
#Final Hyperparameters: {'criterion': 'gini', 'max_depth': 10, 'max_features': 'auto', 'min_impurity_decrease': 0, 'n_estimators': 70 }
def tune_rf():
    global rf_best_parameters
    tune_rf_one()
    tune_rf_two()
    tune_rf_three()
    rf_best_parameters['n_estimators'] = rf_best_parameters['n_estimators'] + 45

def tune_rf_one():
    global rf_best_parameters
    seed = 1567708903
    numpy.random.seed(seed)
    train_X, train_Y, test_X, test_Y, data = get_data(seed)
    train_X, train_Y, test_X, test_Y = fold_data_one(train_X, train_Y)
    bestnt = rf_best_parameters['n_estimators']
    bestmd = rf_best_parameters['max_depth']
    mds = [bestmd - 5, bestmd, bestmd + 5, bestmd + 10, bestmd + 15, bestmd + 20, bestmd + 25, bestmd + 30,          bestmd - 5, bestmd, bestmd + 5, bestmd + 10, bestmd + 15, bestmd + 20, bestmd + 25, bestmd + 30,            bestmd - 5, bestmd, bestmd + 5, bestmd + 10, bestmd + 15, bestmd + 20, bestmd + 25, bestmd + 30,          bestmd - 5, bestmd, bestmd + 5, bestmd + 10, bestmd + 15, bestmd + 20, bestmd + 25, bestmd + 30,            bestmd - 5, bestmd, bestmd + 5, bestmd + 10, bestmd + 15, bestmd + 20, bestmd + 25, bestmd + 30  ]
    ntrees = [bestnt - 15, bestnt -15, bestnt - 15, bestnt - 15, bestnt - 15, bestnt - 15, bestnt - 15, bestnt - 15,          bestnt , bestnt , bestnt , bestnt , bestnt , bestnt , bestnt , bestnt ,        bestnt + 15, bestnt + 15, bestnt + 15, bestnt + 15, bestnt + 15, bestnt + 15, bestnt + 15, bestnt + 15,          bestnt + 30, bestnt +30, bestnt +30, bestnt +30, bestnt +30, bestnt +30, bestnt +30, bestnt +30,            bestnt + 45, bestnt +45, bestnt +45, bestnt +45, bestnt +45, bestnt +45, bestnt +45, bestnt +45  ]
    accuracy = []
    for j in range(0, len(mds)):
        md = mds[j]
        nt = ntrees[j]
        clf = RandomForestClassifier(criterion= rf_best_parameters['criterion'], max_depth= md, max_features= rf_best_parameters['max_features'], min_impurity_decrease= rf_best_parameters['min_impurity_decrease'], n_estimators= nt, random_state= 1567708903).fit(train_X, train_Y)
        predictions = clf.predict(test_X)
        accurate = 0
        total = 0
        for i in range(0,len(predictions)):
            total += 1
            if predictions[i] == test_Y[i]:
                accurate += 1
        accuracy.append(accurate/total)
    graph_three_way_tuning(mds, ntrees, accuracy, "Max Depth", "Number of Estimators", "Fold 1: Max Depth vs Number of Estimators vs Accuracy")

def tune_rf_two():
    global rf_best_parameters
    seed = 1567708903
    numpy.random.seed(seed)
    train_X, train_Y, test_X, test_Y, data = get_data(seed)
    train_X, train_Y, test_X, test_Y = fold_data_two(train_X, train_Y)
    bestnt = rf_best_parameters['n_estimators']
    bestmd = rf_best_parameters['max_depth']
    mds = [bestmd - 5, bestmd, bestmd + 5, bestmd + 10, bestmd + 15, bestmd + 20, bestmd + 25, bestmd + 30,          bestmd - 5, bestmd, bestmd + 5, bestmd + 10, bestmd + 15, bestmd + 20, bestmd + 25, bestmd + 30,            bestmd - 5, bestmd, bestmd + 5, bestmd + 10, bestmd + 15, bestmd + 20, bestmd + 25, bestmd + 30,          bestmd - 5, bestmd, bestmd + 5, bestmd + 10, bestmd + 15, bestmd + 20, bestmd + 25, bestmd + 30,            bestmd - 5, bestmd, bestmd + 5, bestmd + 10, bestmd + 15, bestmd + 20, bestmd + 25, bestmd + 30  ]
    ntrees = [bestnt - 15, bestnt -15, bestnt - 15, bestnt - 15, bestnt - 15, bestnt - 15, bestnt - 15, bestnt - 15,          bestnt , bestnt , bestnt , bestnt , bestnt , bestnt , bestnt , bestnt ,        bestnt + 15, bestnt + 15, bestnt + 15, bestnt + 15, bestnt + 15, bestnt + 15, bestnt + 15, bestnt + 15,          bestnt + 30, bestnt +30, bestnt +30, bestnt +30, bestnt +30, bestnt +30, bestnt +30, bestnt +30,            bestnt + 45, bestnt +45, bestnt +45, bestnt +45, bestnt +45, bestnt +45, bestnt +45, bestnt +45  ]
    accuracy = []
    for j in range(0, len(mds)):
        md = mds[j]
        nt = ntrees[j]
        clf = RandomForestClassifier(criterion= rf_best_parameters['criterion'], max_depth= md, max_features= rf_best_parameters['max_features'], min_impurity_decrease= rf_best_parameters['min_impurity_decrease'], n_estimators= nt, random_state= 1567708903).fit(train_X, train_Y)
        predictions = clf.predict(test_X)
        accurate = 0
        total = 0
        for i in range(0,len(predictions)):
            total += 1
            if predictions[i] == test_Y[i]:
                accurate += 1
        accuracy.append(accurate/total)
    graph_three_way_tuning(mds, ntrees, accuracy, "Max Depth", "Number of Estimators", "Fold 2: Max Depth vs Number of Estimators vs Accuracy")

def tune_rf_three():
    global rf_best_parameters
    seed = 1567708903
    numpy.random.seed(seed)
    train_X, train_Y, test_X, test_Y, data = get_data(seed)
    train_X, train_Y, test_X, test_Y = fold_data_three(train_X, train_Y)
    bestnt = rf_best_parameters['n_estimators']
    bestmd = rf_best_parameters['max_depth']
    mds = [bestmd - 5, bestmd, bestmd + 5, bestmd + 10, bestmd + 15, bestmd + 20, bestmd + 25, bestmd + 30,          bestmd - 5, bestmd, bestmd + 5, bestmd + 10, bestmd + 15, bestmd + 20, bestmd + 25, bestmd + 30,            bestmd - 5, bestmd, bestmd + 5, bestmd + 10, bestmd + 15, bestmd + 20, bestmd + 25, bestmd + 30,          bestmd - 5, bestmd, bestmd + 5, bestmd + 10, bestmd + 15, bestmd + 20, bestmd + 25, bestmd + 30,            bestmd - 5, bestmd, bestmd + 5, bestmd + 10, bestmd + 15, bestmd + 20, bestmd + 25, bestmd + 30  ]
    ntrees = [bestnt - 15, bestnt -15, bestnt - 15, bestnt - 15, bestnt - 15, bestnt - 15, bestnt - 15, bestnt - 15,          bestnt , bestnt , bestnt , bestnt , bestnt , bestnt , bestnt , bestnt ,        bestnt + 15, bestnt + 15, bestnt + 15, bestnt + 15, bestnt + 15, bestnt + 15, bestnt + 15, bestnt + 15,          bestnt + 30, bestnt +30, bestnt +30, bestnt +30, bestnt +30, bestnt +30, bestnt +30, bestnt +30,            bestnt + 45, bestnt +45, bestnt +45, bestnt +45, bestnt +45, bestnt +45, bestnt +45, bestnt +45  ]
    accuracy = []
    for j in range(0, len(mds)):
        md = mds[j]
        nt = ntrees[j]
        clf = RandomForestClassifier(criterion= rf_best_parameters['criterion'], max_depth= md, max_features= rf_best_parameters['max_features'], min_impurity_decrease= rf_best_parameters['min_impurity_decrease'], n_estimators= nt, random_state= 1567708903).fit(train_X, train_Y)
        predictions = clf.predict(test_X)
        accurate = 0
        total = 0
        for i in range(0,len(predictions)):
            total += 1
            if predictions[i] == test_Y[i]:
                accurate += 1
        accuracy.append(accurate/total)
    graph_three_way_tuning(mds, ntrees, accuracy, "Max Depth", "Number of Estimators", "Fold 3: Max Depth vs Number of Estimators vs Accuracy")

#Divide Test Data into 50 folds, finds errors for each fold to calculate confidence intervals
def test_test_data(train_X, train_Y, test_X, test_Y, model):
    predictions = model.predict(test_X)
    cv = StratifiedKFold(n_splits=50, shuffle=False)
    splits = cv.split(test_X, test_Y)
    errors = []
    for train, test in splits:
        total = 0
        accurate = 0
        for i in range(0,len(predictions)):
            if i in test:
                total += 1
                if predictions[i] == test_Y[i][0]:
                    accurate += 1
        if not (total == 0):
            error = 1 - (accurate/total)
            errors.append(error)
    x_bar = numpy.mean(errors)
    sigma = 0
    for x in errors:
        sigma += (x - x_bar) * (x - x_bar)
    sigma = numpy.sqrt(sigma/49)
    margin_error = sigma / numpy.sqrt(50)
    #For alpha = .05, z = 1.9600. 
    margin_error = margin_error * 1.96
    print("Margin of Error:")
    print(margin_error)
    print("Mean Accuracy:")
    print(1 - x_bar)

def test_fold_mean_validation_accuracy():
    global svm_best_parameters
    global rf_best_parameters
    global nn_best_parameters
    seed = 1567708903
    svm_accuracies = []
    nn_accuracies = []
    rf_accuracies = []
    total_svm_time = 0
    total_nn_time = 0
    total_rf_time = 0
    #Fold 1
    train_X, train_Y, test_X, test_Y, data = get_data(seed)
    train_X, train_Y, test_X, test_Y = fold_data_one(train_X, train_Y)
    millis = int(round(time.time() * 1000))
    clf_svm = svm.SVC(kernel = svm_best_parameters['kernel'], gamma = svm_best_parameters['gamma'], decision_function_shape = svm_best_parameters['decision_function_shape'], C = svm_best_parameters['C']).fit(train_X, train_Y)
    total_svm_time += int(round(time.time() * 1000)) - millis
    millis = int(round(time.time() * 1000))
    clf_rf = RandomForestClassifier(criterion= rf_best_parameters['criterion'], max_depth= rf_best_parameters['max_depth'], max_features= rf_best_parameters['max_features'], min_impurity_decrease= rf_best_parameters['min_impurity_decrease'], n_estimators= rf_best_parameters['n_estimators'], random_state= 1567708903).fit(train_X, train_Y)
    total_rf_time += int(round(time.time() * 1000)) - millis
    millis = int(round(time.time() * 1000))
    clf_nn = MLPClassifier(solver=nn_best_parameters['solver'], alpha=nn_best_parameters['alpha'], hidden_layer_sizes=nn_best_parameters['hidden_layer_sizes'], random_state=seed, tol=nn_best_parameters['tol'],activation='tanh',learning_rate='constant').fit(train_X, train_Y)
    total_nn_time += int(round(time.time() * 1000)) - millis
    predictions_svm = clf_svm.predict(test_X)
    predictions_rf = clf_rf.predict(test_X)
    predictions_nn = clf_nn.predict(test_X)
    svm_accuracies.append(test_fold_accuracy(predictions_svm, test_Y))
    rf_accuracies.append(test_fold_accuracy(predictions_rf, test_Y))
    nn_accuracies.append(test_fold_accuracy(predictions_nn, test_Y))
    #Fold 2
    train_X, train_Y, test_X, test_Y, data = get_data(seed)
    train_X, train_Y, test_X, test_Y = fold_data_two(train_X, train_Y)
    millis = int(round(time.time() * 1000))
    clf_svm = svm.SVC(kernel = svm_best_parameters['kernel'], gamma = svm_best_parameters['gamma'], decision_function_shape = svm_best_parameters['decision_function_shape'], C = svm_best_parameters['C']).fit(train_X, train_Y)
    total_svm_time += int(round(time.time() * 1000)) - millis
    millis = int(round(time.time() * 1000))
    clf_rf = RandomForestClassifier(criterion= rf_best_parameters['criterion'], max_depth= rf_best_parameters['max_depth'], max_features= rf_best_parameters['max_features'], min_impurity_decrease= rf_best_parameters['min_impurity_decrease'], n_estimators= rf_best_parameters['n_estimators'], random_state= 1567708903).fit(train_X, train_Y)
    total_rf_time += int(round(time.time() * 1000)) - millis
    millis = int(round(time.time() * 1000))
    clf_nn = MLPClassifier(solver=nn_best_parameters['solver'], alpha=nn_best_parameters['alpha'], hidden_layer_sizes=nn_best_parameters['hidden_layer_sizes'], random_state=seed, tol=nn_best_parameters['tol'],activation='tanh',learning_rate='constant').fit(train_X, train_Y)
    total_nn_time += int(round(time.time() * 1000)) - millis
    predictions_svm = clf_svm.predict(test_X)
    predictions_rf = clf_rf.predict(test_X)
    predictions_nn = clf_nn.predict(test_X)
    svm_accuracies.append(test_fold_accuracy(predictions_svm, test_Y))
    rf_accuracies.append(test_fold_accuracy(predictions_rf, test_Y))
    nn_accuracies.append(test_fold_accuracy(predictions_nn, test_Y))
    #Fold 3
    train_X, train_Y, test_X, test_Y, data = get_data(seed)
    train_X, train_Y, test_X, test_Y = fold_data_three(train_X, train_Y)
    millis = int(round(time.time() * 1000))
    clf_svm = svm.SVC(kernel = svm_best_parameters['kernel'], gamma = svm_best_parameters['gamma'], decision_function_shape = svm_best_parameters['decision_function_shape'], C = svm_best_parameters['C']).fit(train_X, train_Y)
    total_svm_time += int(round(time.time() * 1000)) - millis
    millis = int(round(time.time() * 1000))
    clf_rf = RandomForestClassifier(criterion= rf_best_parameters['criterion'], max_depth= rf_best_parameters['max_depth'], max_features= rf_best_parameters['max_features'], min_impurity_decrease= rf_best_parameters['min_impurity_decrease'], n_estimators= rf_best_parameters['n_estimators'], random_state= 1567708903).fit(train_X, train_Y)
    total_rf_time += int(round(time.time() * 1000)) - millis
    millis = int(round(time.time() * 1000))
    clf_nn = MLPClassifier(solver=nn_best_parameters['solver'], alpha=nn_best_parameters['alpha'], hidden_layer_sizes=nn_best_parameters['hidden_layer_sizes'], random_state=seed, tol=nn_best_parameters['tol'],activation='tanh',learning_rate='constant').fit(train_X, train_Y)
    total_nn_time += int(round(time.time() * 1000)) - millis
    predictions_svm = clf_svm.predict(test_X)
    predictions_rf = clf_rf.predict(test_X)
    predictions_nn = clf_nn.predict(test_X)
    svm_accuracies.append(test_fold_accuracy(predictions_svm, test_Y))
    rf_accuracies.append(test_fold_accuracy(predictions_rf, test_Y))
    nn_accuracies.append(test_fold_accuracy(predictions_nn, test_Y))
    print("Single mean accuracy statistics: ")
    print("Random Forest:")
    print(numpy.mean(rf_accuracies))
    print("SVM:")
    print(numpy.mean(svm_accuracies))
    print("Neural Network:")
    print(numpy.mean(nn_accuracies))
    #Generate Confidence Intervals for each model
    print ("Confidence Intervals (margin of error, mean accuracy): ")
    #RF
    x_bar = numpy.mean([(1 - rf_accuracies[0]), (1 - rf_accuracies[1]), (1 - rf_accuracies[2])])
    sigma = 0
    for error in rf_accuracies:
        error = 1 - error
        sigma += (error - x_bar) * (error - x_bar)
    sigma = numpy.sqrt(sigma/2)
    margin_error = sigma / numpy.sqrt(3)
    #For alpha = .05, z = 1.9600. 
    margin_error = margin_error * 1.96
    print("Random Forest:")
    print(margin_error)
    print(1 - x_bar)
    #SVM
    x_bar = numpy.mean([(1 - svm_accuracies[0]), (1 - svm_accuracies[1]), (1 - svm_accuracies[2])])
    sigma = 0
    for error in svm_accuracies:
        error = 1 - error
        sigma += (error - x_bar) * (error - x_bar)
    sigma = numpy.sqrt(sigma/2)
    margin_error = sigma / numpy.sqrt(3)
    #For alpha = .05, z = 1.9600.
    margin_error = margin_error * 1.96
    print("SVM")
    print(margin_error)
    print(1 - x_bar)
    #NN
    x_bar = numpy.mean([(1 - nn_accuracies[0]), (1 - nn_accuracies[1]), (1 - nn_accuracies[2])])
    sigma = 0
    for error in nn_accuracies:
        error = 1 - error
        sigma += (error - x_bar) * (error - x_bar)
    sigma = numpy.sqrt(sigma/2)
    margin_error = sigma / numpy.sqrt(3)
    #For alpha = .05, z = 1.9600. This was calculated using a calculator that provides access to norminv. If this is necessary to be recalculated, it can be proven by following the computation at https://www.mathworks.com/help/stats/norminv.html.
    margin_error = margin_error * 1.96
    print("Neural Network")
    print(margin_error)
    print(1 - x_bar)
    print("Run Time statistics (in milliseconds): ")
    print("Random Forest time per model: ")
    print(total_rf_time / 3)
    print("SVM time per model: ")
    print(total_svm_time / 3)
    print("Neural Network time per model: ")
    print(total_nn_time / 3)

def test_fold_accuracy(predictions, test_Y):
    total = 0
    accurate = 0
    for i in range(0,len(predictions)):
        total += 1
        if predictions[i] == test_Y[i]:
            accurate += 1
    return accurate/total

def grid_search_all():
    global nn_best_parameters
    grid_search_rf()
    grid_search_svm()
    grid_search_neural_network()
    #IMPORTANT Note: if time is of essence, grid search neural network can be skipped by commenting the above line out, and uncommenting out the below, line 763
    #Doing this automatically sets the parameters to the parameters returned by grid search.
    #nn_best_parameters = {'activation': 'tanh', 'alpha': 0.2, 'hidden_layer_sizes': (100, 100, 100, 100, 100, 100), 'learning_rate': 'constant', 'random_state': 1567708903, 'solver': 'adam', 'tol': 1e-05}

def run_finalized_accuracies():
    global rf_best_parameters
    global nn_best_parameters
    global svm_best_parameters
    seed = 1567708903
    train_X, train_Y, test_X, test_Y, data = get_data(seed)
    millis = int(round(time.time() * 1000))
    clf = RandomForestClassifier(criterion= rf_best_parameters['criterion'], max_depth= rf_best_parameters['max_depth'], max_features= rf_best_parameters['max_features'], min_impurity_decrease= rf_best_parameters['min_impurity_decrease'], n_estimators= rf_best_parameters['n_estimators'], random_state= 1567708903).fit(train_X, train_Y)
    print("Final Time to Train Neural Network, in milliseconds: ")
    print(int(round(time.time() * 1000)) - millis)
    print("Random Forest Train Accuracy: ")
    predictions = clf.predict(train_X)
    test_accuracy(predictions, train_Y)
    print("Random Forest Test Accuracy: ")
    predictions = clf.predict(test_X)
    test_accuracy(predictions, test_Y)
    print("Random Forest Matrix: ")
    run_confusion_matrix(test_Y, predictions)
    print("Random Forest Confidence Interval")
    test_test_data(train_X, train_Y, test_X, test_Y, clf)
    millis = int(round(time.time() * 1000))
    clf = svm.SVC(kernel = svm_best_parameters['kernel'], gamma = svm_best_parameters['gamma'], decision_function_shape = svm_best_parameters['decision_function_shape'], C = svm_best_parameters['C']).fit(train_X, train_Y)
    print("Final Time to Train Neural Network, in milliseconds: ")
    print(int(round(time.time() * 1000)) - millis)
    print("SVM Train Accuracy: ")
    predictions = clf.predict(train_X)
    test_accuracy(predictions, train_Y)
    print("SVM Test Accuracy: ")
    predictions = clf.predict(test_X)
    test_accuracy(predictions, test_Y)
    print("SVM Matrix: ")
    run_confusion_matrix(test_Y, predictions)
    print("SVM Confidence Interval: ")
    test_test_data(train_X, train_Y, test_X, test_Y, clf)
    millis = int(round(time.time() * 1000))
    clf = MLPClassifier(solver=nn_best_parameters['solver'], alpha=nn_best_parameters['alpha'], hidden_layer_sizes=nn_best_parameters['hidden_layer_sizes'], random_state=seed, tol=nn_best_parameters['tol'],activation=nn_best_parameters['activation'],learning_rate=nn_best_parameters['learning_rate']).fit(train_X, train_Y)
    print("Final Time to Train Neural Network, in milliseconds: ")
    print(int(round(time.time() * 1000)) - millis)
    print("Neural Network Train Accuracy: ")
    predictions = clf.predict(train_X)
    test_accuracy(predictions, train_Y)
    print("Neural Network Test Accuracy: ")
    predictions = clf.predict(test_X)
    test_accuracy(predictions, test_Y)
    print("Neural Network Matrix: ")
    run_confusion_matrix(test_Y, predictions)
    print("Neural Network Confidence Interval: ")
    test_test_data(train_X, train_Y, test_X, test_Y, clf)


def run_full_experiment():
    seed = 1567708903
    train_X, train_Y, test_X, test_Y, data = get_data(seed)
    grid_search_all()
    test_linearly_seperable()
    tune_all()
    test_fold_mean_validation_accuracy()
    run_finalized_accuracies()


run_full_experiment()

