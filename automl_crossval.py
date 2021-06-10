from metaflow import FlowSpec, current, step, argo_base


@argo_base(image='mlf.docker.repositories.sapcdn.io/aif/metaflow-sklearn:0.0.1',
           envFrom=[{'secretRef': {'name': 'default-object-store-secret'}}],
           imagePullSecrets=[{'name': 'docker-registry-secret'}],
           annotations={'scenarios.ai.sap.com/name': 'metaflow-demo', 'executables.ai.sap.com/name': 'automl-crossval'},
           labels={'scenarios.ai.sap.com/id': 'metaflow-demo', 'ai.sap.com/version': '0.0.1', 'ai.sap.com/resourcePlan': 'starter'}           
           )
class AutoMLCrossVal(FlowSpec):
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

        self.next(self.classifiers)

    @step
    def classifiers(self):
        """
        set up classifiers and 
        train in parallel
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.neural_network import MLPClassifier
        from sklearn.ensemble import RandomForestClassifier

        names = ["LogisticRegression", "Random Forest", "Neural Net"]
        classifiers = [
            LogisticRegression(solver='liblinear'),
            RandomForestClassifier(max_depth=2, n_estimators=3, max_features=1, random_state=42),
            MLPClassifier(alpha=1, max_iter=600, random_state=42)
        ]

        self.classifier = list(zip(names, classifiers))

        self.next(self.cross_validation, foreach='classifier')

    @step
    def cross_validation(self):
        """
        split into folds for cross-validation
        which are trained in parallel 
        """
        from sklearn.model_selection import StratifiedKFold

        self.nm_clf = self.input  # one classifier
        print(self.nm_clf[0])

        skf = StratifiedKFold(n_splits=AutoMLCrossVal.NUM_FOLDS)
        self.k_folds = list(skf.split(self.X_train, self.y_train))

        self.next(self.fold_score, foreach='k_folds')

    @step
    def fold_score(self):
        """
        calculate and persist score per fold
        """
        train, test = self.input
        name, clf = self.nm_clf
        clf.fit(self.X_train[train], self.y_train[train])
        accuracy = clf.score(self.X_train[test], self.y_train[test])

        self.kfold_score = [current.task_id, name, clf, accuracy]
        print(self.kfold_score)

        self.next(self.join_folds)

    @step
    def join_folds(self, inputs):
        """
        collect scores per fold
        """
        self.cv_score = [inp.kfold_score for inp in inputs]
        self.merge_artifacts(inputs, exclude=['kfold_score', 'nm_clf', 'k_folds'])

        self.next(self.join_classifiers)

    @step
    def join_classifiers(self, inputs):
        """
        collect CV scores per classifier
        """
        self.clf_scores = [inp.cv_score for inp in inputs]
        self.merge_artifacts(inputs, exclude=['cv_score', 'nm_clf'])

        self.next(self.end)

    @step
    def end(self):
        """
        compare scores from single step to parallel calculation
        """
        from sklearn import metrics
        import statistics

        best_clf = ''
        best_score = 0
        i = 0
        for clf in self.clf_scores:
            clf_name = clf[i][1]
            print(clf_name)
            fold_scores = [s[3] for s in clf] 
            print('  score: ', fold_scores)
            print('mean accuracy: {}  with standard deviation {}'.format(statistics.mean(fold_scores),
                                                                         statistics.pstdev(fold_scores)))
            trained_classifier = clf[i][2]
            test_score = metrics.accuracy_score(trained_classifier.predict(self.X_test), self.y_test)
            print('accuracy on test set: ', test_score)
            if test_score > best_score:
                best_score = test_score
                best_clf = clf_name
            print()
            i += 1

        # save best model
        print('model with best performance: {} with test accuracy: {}'.format(best_clf, best_score))


if __name__ == '__main__':
    AutoMLCrossVal()
