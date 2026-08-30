import sys , os
from dataclasses import dataclass
import numpy as np 
import pandas as pd 
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

'''
------------------------------------------------------------------------------------
Data Transformation :

1. Read the training and testing datasets from their respective CSV files.

2. Create the preprocessing object containing the required transformations
   for the input features (e.g., numerical scaling and categorical encoding).

3. Separate the target column (math_score) from the input features.
   - Input features (X): all columns except math_score.
  - Target feature (y): math_score, which the model needs to predict.

4. Fit the preprocessing object ONLY on the training input features and
   transform them. The transformer learns parameters such as mean, standard
   deviation, encoding mappings, etc. from the training data.

5. Transform the testing input features using the SAME fitted preprocessing
   object. We use transform() instead of fit_transform() so that the test
   data does not influence the preprocessing parameters, thereby avoiding
   data leakage.

6. The target feature (math_score) is kept in its original form because it
   is the prediction target and is not part of the input-feature
   preprocessing. A separate target transformation can be applied if
   required by the ML algorithm/problem.

7. Combine the transformed input features and the original target column
   using np.c_ (column-wise concatenation) to create the final train_arr
   and test_arr.

Final structure:

   train_arr = [ transformed X_train | y_train ]
   test_arr  = [ transformed X_test  | y_test  ]

This produces the final NumPy arrays that can be saved and later used
for model training and evaluation.
------------------------------------------------------------------------------------
'''

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts',"preprocessor.pkl")

class DataTranformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns = [
                'reading_score',
                'writing_score'
            ]
            categorical_columns=[
                'gender', 
                'race_ethnicity', 
                'parental_level_of_education', 
                'lunch', 
                'test_preparation_course'
            ]
            num_pipeline = Pipeline(                          #numerical pipeline
                steps=[
                    ("imputer",SimpleImputer(strategy="median")), #replace empty values with medain 
                    ("scaler",StandardScaler()),                  #standard scaling
                ]
            )
            logging.info("Numerical columns standard scaling completed")

            cat_pipeline = Pipeline(                          #categorical pipeline
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")), #replace empty values with most frequent value 
                    ("one_hot_encoder",OneHotEncoder()),                 #perform one hot encoding for categorical values
                    ("scaler",StandardScaler(with_mean=False)) 
                ]
            )
            logging.info("Categorical columns Encoding completed")

            preprocessor = ColumnTransformer( #Applies transformers to columns of an array or pandas DataFrame
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipeline",cat_pipeline,categorical_columns)
                ]
            )
            return preprocessor
        
        except:
            raise CustomException
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "math_score"
            numerical_columns = ['reading_score','writing_score']

            #train df 
            input_feature_train_df = train_df.drop(columns=[target_column_name])
            target_feature_train_df = train_df[target_column_name]

            #test df 
            input_feature_test_df = test_df.drop(columns=[target_column_name])
            target_feature_test_df = test_df[target_column_name]

            logging.info(f"Applying preprocessing object on training df and testing df")

            #preprocessing transformer on train df and test df
            # Fit ONLY on training features
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            # Transform test features using the SAME fitted transformer
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            #combine transformed X + original y
            train_arr = np.c_[                     # np.c_ concatenates arrays column-wise
                input_feature_train_arr,
                np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr,
                np.array(target_feature_test_df)
            ]

            logging.info("Saved preprocessing object")

            save_object(   #save obj as a pkl file - function present in util.py
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )


        except Exception as e:
            raise CustomException(e,sys)
            