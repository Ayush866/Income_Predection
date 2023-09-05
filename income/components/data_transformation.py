from income.entity import artifact_entity, config_entity
from income.exception import CustomException
from income.logger import logging
from typing import Optional
import os, sys
from sklearn.pipeline import Pipeline
import pandas as pd
from income import utils
import numpy as np
from sklearn.preprocessing import LabelEncoder
from imblearn.combine import SMOTETomek
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from income.config import TARGET_COLUMN
from sklearn.preprocessing import LabelEncoder


class DataTransformation:

    def __init__(self, data_transformation_config: config_entity.DataTransformationConfig,
                 data_ingestion_artifact: artifact_entity.DataIngestionArtifact):
        try:
            logging.info(f"{'>>' * 20} Data Transformation {'<<' * 20}")
            self.data_transformation_config = data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact
        except Exception as e:
            raise CustomException(e, sys)

    @classmethod
    def get_data_transformer_object(cls) -> Pipeline:
        try:
            numerical_columns = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]

            categorical_columns = ["workclass", "marital-status", "occupation", "relationship", "race", "sex",
                                   "country"]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median"))

                ]
            )
            cat_pipeline = Pipeline(

                steps=[
                    ("imputer", SimpleImputer(missing_values=np.NAN, strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),

                ]

            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipelines", cat_pipeline, categorical_columns),

                ]

            )
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor)
            ])
            return pipeline

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, ) -> artifact_entity.DataTransformationArtifact:
        try:
            # reading training and testing file
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            # selecting input feature for train and test dataframe
            input_feature_train_df = train_df.drop(TARGET_COLUMN, axis=1)
            input_feature_test_df = test_df.drop(TARGET_COLUMN, axis=1)

            # selecting target feature for train and test dataframe
            target_feature_train_df = train_df[TARGET_COLUMN]
            target_feature_test_df = test_df[TARGET_COLUMN]

            label_encoder = LabelEncoder()
            label_encoder.fit(target_feature_train_df)
            label_encoder.fit(target_feature_test_df)

            # transformation on target columns
            target_feature_train_arr = label_encoder.transform(target_feature_train_df)
            target_feature_test_arr = label_encoder.transform(target_feature_test_df)

            transformation_pipleine = DataTransformation.get_data_transformer_object()
            transformation_pipleine.fit(input_feature_train_df)

            # transforming input features
            input_feature_train_arr = transformation_pipleine.transform(input_feature_train_df)
            input_feature_test_arr = transformation_pipleine.transform(input_feature_test_df)
            logging.info(f"{target_feature_train_arr}")

            smt = SMOTETomek(random_state=42)
            # logging.info(f"Before resampling in training set Input: {input_feature_train_arr.shape} Target:{target_feature_train_arr.shape}")
            input_feature_train_arr, target_feature_train_arr = smt.fit_resample(input_feature_train_arr,
                                                                                 target_feature_train_arr)
            # logging.info(f"After resampling in training set Input: {input_feature_train_arr.shape} Target:{target_feature_train_arr.shape}")

            # logging.info(f"Before resampling in testing set Input: {input_feature_test_arr.shape} Target:{target_feature_test_arr.shape}")
            input_feature_test_arr, target_feature_test_arr = smt.fit_resample(input_feature_test_arr,
                                                                               target_feature_test_arr)
            # logging.info(f"After resampling in testing set Input: {input_feature_test_arr.shape} Target:{target_feature_test_arr.shape}")
            # logging.info(f"{input_feature_train_arr.shape}")
            # logging.info(f"{input_feature_train_arr}")
            logging.info(f"{target_feature_train_arr.shape}")

            input_feature_train_arr1 = input_feature_train_arr.toarray()
            input_feature_test_arr1 = input_feature_test_arr.toarray()

            target_feature_train_arr1 = target_feature_train_arr.reshape(-1, 1)
            # logging.info(f"{target_feature_train_arr1.shape}")
            target_feature_test_arr1 = target_feature_test_arr.reshape(-1, 1)
            # logging.info(f"{target_feature_test_arr1.shape}")

            # logging.info(f"{input_feature_test_arr}")
            # logging.info(f"{target_feature_test_arr1}")
            # print(type(input_feature_test_arr1))
            # print(type(target_feature_test_arr1))

            # target encoder
            test_arr = np.concatenate((input_feature_test_arr1, target_feature_test_arr1), axis=1)
            train_arr = np.concatenate((input_feature_train_arr1, target_feature_train_arr1), axis=1)

            # save numpy array
            utils.save_numpy_array_data(file_path=self.data_transformation_config.transformed_train_path,
                                        array=train_arr)

            utils.save_numpy_array_data(file_path=self.data_transformation_config.transformed_test_path,
                                        array=test_arr)

            utils.save_object(file_path=self.data_transformation_config.transform_object_path,
                              obj=transformation_pipleine)

            utils.save_object(file_path=self.data_transformation_config.target_encoder_path,
                              obj=label_encoder)

            data_transformation_artifact = artifact_entity.DataTransformationArtifact(
                transform_object_path=self.data_transformation_config.transform_object_path,
                transformed_train_path=self.data_transformation_config.transformed_train_path,
                transformed_test_path=self.data_transformation_config.transformed_test_path,
                target_encoder_path=self.data_transformation_config.target_encoder_path

            )

            logging.info(f"Data transformation object {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise CustomException(e, sys)