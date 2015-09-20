import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import RandomizedSearchCV
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import json
import os

class IrisModel:
    
    def datapalio_interface(self, **kwargs):
        
        """
        This is the method used by DataPal.io to interact with the model.
        
        Inputs:
                
        - pipe_id (integer): id of the pipe that has to be used.
        
        - input_data (dictionary): dictionary that contains the input data. The keys of the dictionary 
        correspond to the names of the inputs specified in models_definition.json for the selected pipe.
        Each key has an associated value. For the input variables the associated value is the value
        of the variable, whereas for the input files the associated value is its filename. 
        
        - input_files_dir (string): Relative path of the directory where the input files are stored
        (the algorithm has to read the input files from there).

        - output_files_dir (string): Relative path of the directory where the output files must be stored
        (the algorithm must store the output files in there).
        
        Outputs:
        
        - output_data (dictionary): dictionary that contains the output data. The keys of the dictionary 
        correspond to the names of the outputs specified in models_definition.json for the selected pipe. 
        Each key has an associated value. For the output variables the associated value is the value
        of the variable, whereas for the output files the associated value is its filename.  

        """
        
        pipe_id = kwargs['pipe_id']
        input_data = kwargs['input_data']
        input_files_dir = kwargs['input_files_dir']
        output_files_dir = kwargs['output_files_dir']
        
        output_data = self.train_or_predict(pipe_id, input_data, input_files_dir, output_files_dir)
        
        return output_data
        
    
    def train_or_predict(self, pipe_id, input_data, input_files_dir, output_files_dir):
        
        """
        Handles user requests.
        
        """
        
        # pipes for prediction
        if pipe_id in [0,1]:

            # load data
            if pipe_id == 0:
                data = []
                for feature in self.feature_names:
                    data.append(input_data[feature])

            elif pipe_id == 1:
                data, data_args = self.load_input_files(input_data=input_data, input_files_dir=input_files_dir, training_data=False)
                data = data["features"]

            # make prediction
            prediction, out_args = self.predict(features=data)

            # decode prediction
            prediction = self.LabelEncoder.inverse_transform(prediction)

            # return answer
            if pipe_id == 0:
                output_data = {}
                output_data[self.target_name] = prediction[0]
                return output_data

            elif pipe_id == 1:
                # save output in csv file and return it
                filename = 'predictions.csv'
                filepath = os.path.join(output_files_dir, filename)
                df = pd.DataFrame(data=data, columns=self.feature_names)
                df[self.target_name] = prediction
                df.to_csv(filepath, index=False)
                output_data = {}
                output_data["file with predictions"] = filename
                return output_data
            else:
                return
        
        # training pipe
        elif pipe_id in [2]:

            # load data
            data, data_args = self.load_input_files(input_data=input_data, input_files_dir=input_files_dir, training_data=True)

            # find best hyperparameters and fit model
            predictor, out_args = self.find_best_parameters_and_get_fitted_model(data=data, set_predictor_after_training=True)
            # set fitted predictor as the default for the DPModel
            self.predictor = predictor

            # save the label encoder
            self.LabelEncoder = data_args['LabelEncoder']

            # get unbiased predictions on training data
            y_pred, unbiased_prediction_args = self.get_unbiased_predictions_on_training_data(data=data)

            output_data = {}

            # save model_configuration.json
            model_definition, out_args = self.get_model_definition(data_args=data_args)
            path = os.path.join(output_files_dir, "model_definition.json")
            self.save_json_file(dict_to_save=model_definition, path=path)
            output_data["model_definition"] = "model_definition.json"

            # save scores.json
            scores, out_args = self.get_scores(data=data, predicted_values=y_pred, data_args=data_args)
            path = os.path.join(output_files_dir, "scores.json")
            self.save_json_file(dict_to_save=scores, path=path)
            output_data["scores"] = "scores.json"

            return output_data
            
        else:
            return


    def load_input_files(self, **kwargs):
        
        """
        Loads both files containing training data and data for prediction. 
        
        Encodes the target labels to integers. 
        
        In case the it is training data, it will return in the output args the 
        LabelEncoder used to encode the target labels to integers. We return it 
        instead of directly storing it, because it will be saved in case the training
        ends without errors.
        
        Inputs:
        - files_paths (string): path the input files.
        - training_data (bool): specifies whether the files containing training
        data or data for making predictions.
        
        Outputs:
        - LabelEncoder (LabelEncoder) (optional): Encodes the labels of the target
        variables to integers.
        
        """
        
        input_data = kwargs['input_data']
        input_files_dir = kwargs['input_files_dir']

        input_file_path = input_files_dir + input_data['database']
        df = pd.read_csv(input_file_path)
        
        training_data = kwargs.pop('training_data', False)
        
        # if we are loading training data, we have to assign an integer to each possible
        # target label in the dataset. We do it by fitting a LabelEncoder,
        if training_data:

            le = LabelEncoder()
            col_name = df.columns[4]
            df[col_name] = le.fit_transform(df[col_name])

            data = {}
            data['features'] = df[df.columns[0:4]].values
            data['targets'] = df[df.columns[4]].values

            self.feature_names = list(df.columns[0:4])
            self.target_name = df.columns[4]    

            out_args = {}
            out_args['LabelEncoder'] = le
            
            return data, out_args

        # if the data is for making predictions
        else:
            data = {}
            # ensure that the columns are in the correct order
            data['features'] = df[self.feature_names].values
            
            out_args = {}
        
            return data, out_args
    
    
    def find_best_parameters_and_get_fitted_model(self, **kwargs):
        
        """
        Finds the best set of hyperparameters for a Random Forest for the provided data. 
        The best hyperparameters are found by repeatedly drawing random samples from a distribution 
        of parameters and evaluating them by using cross validation.
        
        Inputs:
        - data (dict): Dictionary that composed by two keys:
            - features (array or list): Two dimensional numpy array or list that contains the training
            features.
            - targets (array or list): Numpy array or list that contains the target labels.
        
        - set_predictor_after_training (boolean): If True, the best predictor is stored as
        predictor for the DPModel. If False the best predictor is returned.
        
        Outputs:
        - predictor (optional): Best predictor found. Only returned if set_predictor_after_training
        is set to False.
        
        
        """
        
        # load data
        data = kwargs['data']
        X = data['features']
        y = data['targets']
        out_args = {}
        
        # we choose Random Fores Classifier as the Machine Learning algorithm for
        # this DPModel.
        rc = RandomForestClassifier()
        
        # here we define the space of parameters over which we want to perform the random search
        param_distributions = {}
        param_distributions["n_estimators"] = [50, 100, 150]

        # do random search
        random_search_outer = RandomizedSearchCV(rc, param_distributions=param_distributions,
            cv=5, n_iter=3)
        random_search_outer.fit(X, y)
            
        predictor = random_search_outer.best_estimator_

        return predictor, out_args
        

    def predict(self, **kwargs):
        
        """
        Makes predictions using the stored predictor of the DPModel.
        
        Inputs:
        - features (list or numpy array): Two dimensional numpy array or list that the 
        input samples as the come out the load_input_files method.
        
        Outputs:
        - prediction (numpy array): Array containing the output predictions.
        
        """
    
        features = kwargs['features']
        predictor = kwargs.pop('predictor', self.predictor)
        
        X = features
        prediction = predictor.predict(X)
        
        out_args = {}
        
        return prediction, out_args


    def get_unbiased_predictions_on_training_data(self, **kwargs):
        
        """
        This method provides unbiased predictions for all our training samples.
        We accomplish that by performing a nested cross validation:
        We leave a hold out set out, and we past the rest of the data to the 
        find_best_parameters_and_get_fitted_model method, which contains a cross validation itself. 
        Then we make predictions on the hold out set with the resulted predictor. This way, we found
        the best hyperparameters without using the hold out data. We repeat this process leaving out 
        different training samples each time by performing a cross validation.
        
        Inputs:
        - data (dict): Dictionary that composed by two keys:
            - features (numpy array or list): Two dimensional array or list that contains the training
            features.
            - targets (numpy array or list): Array or list that contains the target labels.
            
        Outputs:
        - y_pred (numpy array): Array that contains the unbiased predictions.
        
        """
        
        data = kwargs['data']
        
        y_true = None
        y_pred = None
        out_args = {}
        
        X = np.array(data['features'])
        y = np.array(data['targets'])
        out_args = {}
        
        # make unbiased predictions using nested CV
        # We will use this unbiased predictions in order to calculate the performance of the
        # algorithm using multiple scores.
        cv = StratifiedKFold(y, n_folds=5)
        for i, (train, test) in enumerate(cv):
            
            data_fold = {}
            data_fold['features'] = X[train]
            data_fold['targets'] = y[train]
                        
            predictor, out_args = self.find_best_parameters_and_get_fitted_model(data=data_fold, set_predictor_after_training=False)
            y_test_pred, out_args = self.predict(predictor=predictor, features=X[test])
            
            if y_true == None:
                y_true = y[test]
                y_pred = y_test_pred
            else:
                y_true = np.hstack((y_true, y[test]))
                y_pred = np.hstack((y_pred, y_test_pred))

        return y_pred, out_args
    
    
    def get_model_definition(self, **kwargs):
        
        """
        Returns model_definition.json dictionary.
        
        """

        model_definition = {}
        model_definition["name"] = "Iris flower classifier"
        model_definition["schema_version"] = "0.02"
        model_definition["environment_name"] = "python2.7.9_June14th2015"
        model_definition["description"] = "Based on the knowledge stored in a database, this predictor has learned " \
                                     "to predict what type of Iris flower is the one you have " \
                                     "found.<br /> Sir Ronald Fisher created a dataset that consists of samples " \
                                     "from each of three species of a flower named Iris (Iris setosa, " \
                                     "Iris virginica and Iris versicolor). Four features were measured " \
                                     "from each sample: the length and the width of the sepals and petals, " \
                                     "in centimetres. You have now found an Iris flower, and you want to " \
                                     "find out what type of Iris it is."
        model_definition["retraining_allowed"] = True
        model_definition["base_algorithm"] = "Random Forest Classifier"     
        model_definition["score_minimized"] = "gini"        

        pipes, out_args = self.get_pipes(**kwargs)
        model_definition["pipes"] = pipes
        
        out_args = {}
        
        return model_definition, out_args
    
    
    def get_pipes(self, **kwargs):
        
        """
        Returns pipes.json dictionary.
        
        """
        
        # get labels for targets
        data_args = kwargs["data_args"]
        le = data_args["LabelEncoder"]
        
        pipes = [ 
                    {
                        "id": 0,
                        "action": "predict",
                        "name":"One by one prediction",
                        "description": "Predict Iris type one by one.",
                        "inputs": [
                            {
                                "name": "sepal length",
                                "type": "variable",
                                "variable_type": "float", 
                                "required": True
                            }, 
                            {
                                "name": "sepal width",
                                "type": "variable",
                                "variable_type": "float", 
                                "required": True
                            }, 
                            {
                                "name": "petal length",
                                "type": "variable",
                                "variable_type": "float", 
                                "required": True
                            }, 
                            {
                                "name": "petal width",
                                "type": "variable",
                                "variable_type": "float", 
                                "required": True
                            }
                        ],
                        "outputs": [
                            {
                                "name": "iris type",
                                "type": "variable",
                                "variable_type": "string", 
                                "values": list(le.classes_)
                            }
                        ]
                    },
                    {
                        "id": 1,
                        "action": "predict",
                        "name":"Bulk prediction",
                        "description": "Upload csv file.",
                        "inputs": [
                            {
                                "name": "database",
                                "type": "file",
                                "extensions": ["csv"],
                                "required": True
                            }
                        ],
                        "outputs": [
                            {
                                "name": "file with predictions",
                                "type": "file",
                                "extensions": ["csv"]
                            }
                        ]
                    },
                    {
                        "id": 2,
                        "action": "train",
                        "name":"Training pipe",
                        "description": "Upload database with target labels.",
                        "inputs": [
                            {
                                "name": "database",
                                "type": "file",
                                "extensions": ["csv"],
                                "required": True
                            }
                        ],
                        "outputs": [
                            {
                                "name": "model_definition",
                                "type": "file",
                                "filenames": ["model_definition.json"]
                            },
                            {
                                "name": "scores",
                                "type": "file",
                                "filenames": ["scores.json"]
                            }
                        ]
                    }

                ]
            
        out_args = {}
        
        return pipes, out_args

    
    def get_scores(self, **kwargs):
        
        """
        Calculate scores.
        
        """
        
        data = kwargs['data']
        true_values = np.array(data['targets'])
        predicted_values = kwargs['predicted_values']
        le = kwargs["data_args"]["LabelEncoder"]

        out_args = {}
        scores = []

        sc = accuracy_score (true_values, predicted_values)
        score = {}
        score['name'] = 'Accuracy'
        score['value'] = sc
        scores.append(score)        
        
        sc = f1_score(true_values, predicted_values, average='weighted')
        score_by_class = f1_score(true_values, predicted_values, average=None)        
        score = {}
        score['name'] = 'F1 score'
        score['summary_name'] = 'Weighted average F1 score'
        score['summary_value'] = sc
        score['class_wise'] = {}
        score['class_wise']['names'] = list(le.classes_)
        score['class_wise']['values'] = list(score_by_class)
        scores.append(score)
        
        sc = precision_score(true_values, predicted_values, average='weighted')
        score_by_class = precision_score (true_values, predicted_values, average=None)        
        score = {}
        score['name'] = 'Precision'
        score['summary_name'] = 'Weighted average precision score'
        score['summary_value'] = sc
        score['class_wise'] = {}
        score['class_wise']['names'] = list(le.classes_)
        score['class_wise']['values'] = list(score_by_class)
        scores.append(score)
        
        sc = recall_score(true_values, predicted_values, average='weighted')
        score_by_class = precision_score (true_values, predicted_values, average=None)
        score = {}
        score['name'] = 'Recall'
        score['summary_name'] = 'Weighted average recall score'
        score['summary_value'] = sc
        score['class_wise'] = {}
        score['class_wise']['names'] = list(le.classes_)
        score['class_wise']['values'] = list(score_by_class)
        scores.append(score)
        
        scores_out = {}
        scores_out["scores"] = scores
        scores_out["schema_version"] = "0.02"

        return scores_out, out_args
    
    
    def save_json_file(self, **kwargs):
        

        """
        Saves dictionary in path.
        
        """
        
        dict_to_save = kwargs["dict_to_save"]
        path = kwargs["path"]
        with open(path,'wb') as fp:
            json.dump(dict_to_save, fp)
            
        return