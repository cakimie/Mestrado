# Filters data:
def extract_x_y(df, country=None, city=None, category=None):
    import numpy as np
    filtered_df = df if country is None and city is None and category is None else df[
        (country is None  or df['country'] == country) &
        (city is None     or df['city'] == city) &
        (category is None or df['category'] == category)
    ]
    df_x = filtered_df.drop(columns=['id', 'category'])
    try:
        df_x = df_x.drop(columns=['country', 'city'])
    except:
        pass
    return np.array(df_x), np.array(filtered_df['category'])

# Extracts fold number 'k' out of 'K' folds for the specified filter:
def load_fold(df, k, K, country=None, city=None, category=None, random_state=1):
    from sklearn.model_selection import KFold
    name = 'timeseries'
    if country is not None:
        name = f'{name}_{country}'
    if city is not None:
        name = f'{name}_{city}'
    if category is not None:
        name = f'{name}_{category}'
    X, y = extract_x_y(df, country, city, category)

    if k < 1 or k > K or K <= 1:
        print(name, X.shape, y.shape, f'ERROR: k={k}, K={K}')
        return name, X, y, None, None

    # Returns the fold:
    kfold = KFold(K, shuffle=True, random_state=random_state)
    train, test = list(kfold.split(X))[k - 1]
    X_train = X[train]
    y_train = y[train]
    X_test = X[test]
    y_test = y[test]
    s = len(f'{K}')
    name = f'{name}_f{f"{{:0{s}d}}".format(k)}-{K}'
    print(name, X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    print(f'Train indices: {train}')
    print(f'Test indices: {test}')
    
    # Verificação para garantir que os índices de treino e teste não se sobrepõem
    intersection = set(train).intersection(set(test))
    if intersection:
        print(f'ERROR: Overlapping indices found in fold {k} of {K}: {intersection}')
    else:
        print(f'Fold {k} of {K} is correctly split with no overlapping indices.')

    return name, X_train, y_train, X_test, y_test

if __name__ == '__main__':
    import pandas as pd
    # Loads full dataset:
    df = pd.read_csv('weekdays_datasets/df_timeseries.csv')
    for k in range(1, 6):
        print(load_fold(df, k=k, K=5, country=0, city=0))