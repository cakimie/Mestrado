# python -m classifiers.rSTSF
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from rSTSF_functions import *

def rSTSF (X_train, y_train, X_test, y_test):
    agg_fns = [np.mean, np.std, np.polyfit, np.median, np.min, np.max, iqr, np.percentile, np.quantile]
    repr_types = [1,2,3,4] # 1: Raw series, 2: Periodogram, 3: First-order Difference, 4: Autoregressive
    d = 50 # Number of sets of candidate discriminatory interval features to compute
    r = 500 # Number of trees

    clf_rSTSF = rstsf(agg_fns=agg_fns, repr_types=repr_types[:1], d=1, r=2)
    clf_rSTSF.fit(X_train,y_train)

    dset_names = ["PopularTimesGoogle"]

    nruns = 10

    accuracies = np.zeros((len(dset_names),nruns))
    training_times = []
    testing_times = []

    cont_dsets = 0
    for dset_name in dset_names:
        print("Dataset: ", dset_name)

        inner_training_time = []
        inner_testing_time = []
        
        for nrun in range(nruns):
            print('run ',str(nrun+1))
            timeA = time.perf_counter()
            
            clf_rstsf = rstsf()
            clf_rstsf.fit(X_train, y_train)

            current_training_time = time.perf_counter()-timeA
            inner_training_time.append(current_training_time)
            print(f"training time: {current_training_time}")

            timeA = time.perf_counter()
            
            rSTSF_pred = clf_rstsf.predict(X_test)

            current_testing_time = time.perf_counter()-timeA
            inner_testing_time.append(current_testing_time)
            print(f"testing time: {current_testing_time}")

            accu = np.sum(rSTSF_pred == y_test)/len(y_test)
            print('accuracy: ', accu)
            accuracies[cont_dsets,nrun] = accu

        avg_accuracy_this_dataset = np.mean(accuracies[cont_dsets,:])
        print('avg accuracy for ' + str(nruns) + ' runs: ' , avg_accuracy_this_dataset)
        
        training_times.append(np.mean(inner_training_time))
        testing_times.append(np.mean(inner_testing_time))

        cont_dsets+=1
        print("\n")

    ## comment/uncomment the lines below according to the number of runs
    columns = {'Dataset':dset_names,
            'run1':accuracies[:,0],
                'run2':accuracies[:,1],
                'run3':accuracies[:,2],
                'run4':accuracies[:,3],
                'run5':accuracies[:,4],
                'run6':accuracies[:,5],
                'run7':accuracies[:,6],
                'run8':accuracies[:,7],
                'run9':accuracies[:,8],
                'run10':accuracies[:,9],
            'avgAccu':np.mean(accuracies,axis=1),
            'avgTrainTime':np.array(training_times),'avgTestTime':np.array(testing_times)}
    dfResults = pd.DataFrame(columns)
    dfResults = dfResults[['Dataset',
                        'run1',
                            'run2',
                            'run3',
                            'run4',
                            'run5',
                            'run6',
                            'run7',
                            'run8',
                            'run9',
                            'run10',
                        'avgAccu','avgTrainTime','avgTestTime'
                        ]]

    return {
        'accuracy_score': accuracy_score(y_test, rSTSF_pred), 
        'f1_score': f1_score(y_test, rSTSF_pred, average='weighted'), 
        'precision_score': precision_score(y_test, rSTSF_pred, average='weighted'), 
        'recall_score': recall_score(y_test, rSTSF_pred, average='weighted'),
    }

if __name__ == '__main__':
    import time
    start_time = time.time()

    import numpy as np
    import pandas as pd
    from classifiers.load_fold import load_fold

    from clearml import Task
    params = {
        'k': 1,
        'K': 10,
        'country': 0,
        'city': 0,
        'category': None,
    }
    task = Task.init(project_name='PopularTimesFold/Classifier', task_name="rstsf")
    task.connect(params)

    df = pd.read_csv('weekdays_datasets/df_timeseries.csv')
    name, X_train, y_train, X_test, y_test = load_fold(
        df,
        params['k'],
        params['K'],
        params['country'],
        params['city'],
        params['category'],
    )
    print(f'Loaded: {name}')
    
    # Executes main function:
    main_time = time.time()
    results = rstsf(X_train, y_train, X_test, y_test)
    task.get_logger().report_scalar('execution_time', 'main', iteration=0, value=time.time() - main_time)

    # Reports results:
    for key, value in results.items():
        task.get_logger().report_scalar('metrics', key, iteration=0, value=value)
    task.get_logger().report_scalar('execution_time', 'total', iteration=0, value=time.time() - start_time)