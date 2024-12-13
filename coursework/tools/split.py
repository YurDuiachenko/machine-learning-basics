def split(X, y, test_size):
    test_size = int(len(X) * test_size)
    X_train = X[:-test_size]
    X_test = X[-test_size:]
    y_train = y[:-test_size]
    y_test = y[-test_size:]

    return X_train, X_test, y_train, y_test