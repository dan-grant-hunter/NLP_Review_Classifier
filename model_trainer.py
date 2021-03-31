from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, precision_score, f1_score
from sklearn.metrics import confusion_matrix


# ==================================================================================================


class TrainModel:


    '''
    The TrainModel object finds the best hyperparameters for a given model using
    StratifiedKFold and then fits the best estimator to the training data.

    Parameters
    ----------
    model : obj (estimator)
        The model to be used for grid search and training.
    params : dict
        The hyperparameters to search over
    X_train : matrix
        Training data
    X_valid : matrix
        Validation data
    y_train : column vector
        Training labels
    y_valid : column vector
        Validation labels

    Attributes
    ----------
    best_params : dict
        The best hyperparameters found for the model
    best_estimator : obj (estimator)
        The best estimator
    pred : array
        Predicitions for the validation data
    model_name : str
        Model name
    classfication_report : DataFrame
        Classification report including precision, recall and f1-score for all classes
    accuracy : float
        Accuracy score
    precision : float
        Precision score (macro average)
    f1 : float
        f1-score (macro average)
    confusion_matrix : array
        Confusion matrix 
    '''


    def __init__(self, model, params, X_train, X_valid, y_train, y_valid):
        self.model = model
        self.X_train = X_train
        self.X_valid = X_valid
        self.y_train = y_train
        self.y_valid = y_valid
        self.params = params 
    

    def train(self):
        cv = StratifiedKFold(n_splits=3, random_state=42, shuffle=True)
        gs = GridSearchCV(estimator=self.model, param_grid=self.params, cv=cv, 
                          scoring='accuracy', verbose=2)
        gs.fit(self.X_train, self.y_train)
        self.best_params = gs.best_params_
        self.best_estimator = gs.best_estimator_
        self.best_estimator.fit(self.X_train, self.y_train)
        self.pred = self.best_estimator.predict(self.X_valid)
        self.model_name = type(self.best_estimator).__name__
        self.classification_report = round(pd.DataFrame(classification_report(
            self.y_valid, self.pred, output_dict=True)).T.iloc[0:5, 0:3],3)
        self.accuracy = round(accuracy_score(self.y_valid, self.pred),3)
        self.precision = round(precision_score(self.y_valid, self.pred, average='macro'),3)
        self.f1 = round(f1_score(self.y_valid, self.pred, average='macro'),3)
        self.confusion_matrix = confusion_matrix(self.y_valid, self.pred)
