from metaflow import FlowSpec, current, step, argo_base, argo


@argo_base(image='mlf.docker.repositories.sapcdn.io/aif/metaflow-sklearn:0.0.1',
#            env=[{'name': 'AWS_ACCESS_KEY_ID', 'value': 'AKIAT4WR4G4TEMEBCPU6'},
#                 {'name': 'AWS_SECRET_ACCESS_KEY', 'value': '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'}],
           envFrom=[{'secretRef': {'name': 'default-object-store-secret'}}],
           imagePullSecrets=[{'name': 'docker-registry-secret'}],
           annotations={'scenarios.ai.sap.com/name': 'metaflow-demo', 'executables.ai.sap.com/name': 'hyperparamtuning'},
           labels={'scenarios.ai.sap.com/id': 'metaflow-demo', 'ai.sap.com/version': '0.0.1', 'ai.sap.com/resourcePlan': 'starter'}           
           )
class HyperParamTuning(FlowSpec):
  """
  Parallel training of MultiLayerPerceptron and RandomForest using hyperparameter search.
  Select model with best performance.
  """

  NUM_FOLDS = 4  # k-fold cross validation

  @step
  def start(self):
    """
    download iris data
    split into train test
    """
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    iris_df = datasets.load_iris(as_frame=True)['frame']
    train, test = train_test_split(iris_df, test_size=0.4, stratify=iris_df['target'], random_state=42)
    self.X_train = train[train.columns[0:4]].to_numpy()
    self.y_train = train.target.to_numpy()
    self.X_test = test[train.columns[0:4]].to_numpy()
    self.y_test = test.target.to_numpy()

    self.next(self.multilayer_perceptron, self.random_forest)

  @step
  def multilayer_perceptron(self):
    from sklearn.neural_network import MLPClassifier

    self.mlp = MLPClassifier(alpha=1, max_iter=3000, random_state=42)

    self.next(self.gridsearch_cv)

  @step
  def gridsearch_cv(self):
    """
    full hyperparameter search
    """
    from sklearn.model_selection import GridSearchCV

    parameter_space = {
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'learning_rate': ['constant', 'adaptive'],
    }

    self.clf = GridSearchCV(self.mlp, parameter_space, cv=HyperParamTuning.NUM_FOLDS, n_jobs=-1)
    self.clf.fit(self.X_train, self.y_train) 

    self.next(self.compare_models)

  @step
  def random_forest(self):
    """
    Random Forest classifier
    """
    from sklearn.ensemble import RandomForestClassifier

    self.rf = RandomForestClassifier(random_state=42)

    self.next(self.randomizedsearch_cv)

  @step
  def randomizedsearch_cv(self):
    """
    randomized search due to big parameter space for random forest
    """
    from sklearn.model_selection import RandomizedSearchCV

    random_grid = {
      'n_estimators': [20, 30, 50, 70, 100],
      'max_features': ['auto', 0.7],
      'max_depth': [5, 10, 30, 50, 80]
    }
    
    self.clf = RandomizedSearchCV(self.rf, random_grid, n_iter=8, cv=HyperParamTuning.NUM_FOLDS, n_jobs=-1, random_state=42)
    self.clf.fit(self.X_train, self.y_train)

    self.next(self.compare_models)

  @step
  def compare_models(self, inputs):
    """
    join parallel training results
    """
    self.clfs = [inp.clf for inp in inputs]
    self.merge_artifacts(inputs, exclude=['clf'])

    for clf in self.clfs:
      print('Best parameters found:\n', clf.best_params_)
      means = clf.cv_results_['mean_test_score']
      stds = clf.cv_results_['std_test_score']
      for mean, std, params in zip(means, stds, clf.cv_results_['params']):
          print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
          
    self.next(self.end)

  @step
  def end(self):
    """
    evaluate best models on test set
    """
    from sklearn.metrics import classification_report
    from sklearn.metrics import accuracy_score

    best_score = 0
    best_clf = ''
    for clf in self.clfs:
      # Evaluate on test set
      y_true, y_pred = self.y_test , clf.predict(self.X_test)
      print('Results on the test set for ', clf.estimator.__class__.__name__)
      print(classification_report(y_true, y_pred, target_names=['Setosa     ', 'Versicolour', 'Virginica  ']))    
      if accuracy_score(y_true, y_pred) > best_score:
        best_clf = clf.estimator.__class__.__name__
        best_score = accuracy_score(y_true, y_pred)
        
    # save best model
    print('\nBest model: {} with test accuracy: {}'.format(best_clf, best_score))


if __name__ == '__main__':
    HyperParamTuning()
