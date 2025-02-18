import os
import sys
from dataclasses import dataclass
import numpy as np
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from mlxtend.regressor import StackingCVRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class CarModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'car', 'model.pkl')

@dataclass
class HouseModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'house', 'model.pkl')

class CarModelTrainer:
    def __init__(self):
        self.model_trainer_config = CarModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            logging.info(f"Train array shape: {train_array.shape}")
            logging.info(f"Test array shape: {test_array.shape}")
            
            # For training data, separate features and target
            X_train = train_array[:,:-1]
            y_train = train_array[:,-1]
            
            # For test data, we already have only features
            X_test = test_array
            
            logging.info(f"X_train shape: {X_train.shape}")
            logging.info(f"y_train shape: {y_train.shape}")
            logging.info(f"X_test shape: {X_test.shape}")

            xgb_params = {
                'learning_rate': 0.01,
                'n_estimators': 3460,
                'max_depth': 3,
                'min_child_weight': 0,
                'gamma': 0,
                'subsample': 0.7,
                'colsample_bytree': 0.7,
                'objective': 'reg:squarederror',
                'random_state': 27
            }

            model = XGBRegressor(**xgb_params)
            
            logging.info("Started training XGBoost model")
            model.fit(X_train, y_train)
            logging.info("Finished training XGBoost model")

            # Make predictions on training data to check performance
            train_predictions = model.predict(X_train)
            train_r2 = r2_score(y_train, train_predictions)
            train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
            
            logging.info(f"Training R2 Score: {train_r2}")
            logging.info(f"Training RMSE: {train_rmse}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model
            )
            
            logging.info("Model saved successfully")
            return train_r2

        except Exception as e:
            logging.error("Error in model training")
            raise CustomException(e, sys)

class HouseModelTrainer:
    def __init__(self):
        self.model_trainer_config = HouseModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            logging.info(f"Train array shape: {train_array.shape}")
            logging.info(f"Test array shape: {test_array.shape}")

            X_train = train_array[:,:-1]
            y_train = train_array[:,-1]
            X_test = test_array
            
            logging.info(f"X_train shape: {X_train.shape}")
            logging.info(f"y_train shape: {y_train.shape}")
            logging.info(f"X_test shape: {X_test.shape}")

            # Initialize base models
            ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=[0.1, 1.0, 10.0]))
            lasso = make_pipeline(RobustScaler(), LassoCV(alphas=[0.0001, 0.0003, 0.0006]))
            elasticnet = make_pipeline(RobustScaler(), ElasticNetCV(alphas=[0.0001, 0.0003, 0.0006]))
            svr = make_pipeline(RobustScaler(), SVR(C=20, epsilon=0.008, gamma=0.0003))
            gbr = GradientBoostingRegressor(
                n_estimators=3000,
                learning_rate=0.05,
                max_depth=4,
                max_features='sqrt',
                min_samples_leaf=15,
                min_samples_split=10,
                random_state=42
            )
            lightgbm = LGBMRegressor(
                objective='regression',
                num_leaves=4,
                learning_rate=0.01,
                n_estimators=5000,
                max_bin=200,
                bagging_fraction=0.75,
                bagging_freq=5,
                bagging_seed=7,
                feature_fraction=0.2,
                feature_fraction_seed=7,
                verbose=-1
            )
            xgboost = XGBRegressor(
                learning_rate=0.01,
                n_estimators=3460,
                max_depth=3,
                min_child_weight=0,
                gamma=0,
                subsample=0.7,
                colsample_bytree=0.7,
                objective='reg:squarederror',
                random_state=27
            )

            # Initialize stacking model
            stacked_model = StackingCVRegressor(
                regressors=[ridge, lasso, elasticnet, gbr, xgboost, lightgbm],
                meta_regressor=xgboost,
                use_features_in_secondary=True
            )

            logging.info("Started training stacked model")
            stacked_model.fit(X_train, y_train)
            logging.info("Finished training stacked model")

            # Make predictions on training data
            train_predictions = stacked_model.predict(X_train)
            train_r2 = r2_score(y_train, train_predictions)
            train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
            
            logging.info(f"Training R2 Score: {train_r2}")
            logging.info(f"Training RMSE: {train_rmse}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=stacked_model
            )

            logging.info("Model saved successfully")
            return train_r2

        except Exception as e:
            logging.error("Error in model training")
            raise CustomException(e, sys)

if __name__=="__main__":
    try:
        # Test Car Model Training
        # logging.info("\n>>>>>>> Starting Car Model Training >>>>>")
        # car_trainer = CarModelTrainer()
        
        # from src.components.data_transformation import CarDataTransformation
        # car_transform = CarDataTransformation()
        # car_train_path = os.path.join('artifacts', 'car', 'train.csv')
        # car_test_path = os.path.join('artifacts', 'car', 'test.csv')
        # car_train_arr, car_test_arr, _ = car_transform.initiate_data_transformation(car_train_path, car_test_path)
        
        # r2_car = car_trainer.initiate_model_trainer(car_train_arr, car_test_arr)
        # logging.info(f"Car Model R2 Score: {r2_car}")

        # Test House Model Training
        logging.info("\n>>>>>>> Starting House Model Training >>>>>")
        house_trainer = HouseModelTrainer()
        
        from src.components.data_transformation import HouseDataTransformation
        house_transform = HouseDataTransformation()
        house_train_path = os.path.join('artifacts', 'house', 'train.csv')
        house_test_path = os.path.join('artifacts', 'house', 'test.csv')
        house_train_arr, house_test_arr, _ = house_transform.initiate_data_transformation(house_train_path, house_test_path)
        
        r2_house = house_trainer.initiate_model_trainer(house_train_arr, house_test_arr)
        logging.info(f"House Model R2 Score: {r2_house}")

    except Exception as e:
        logging.error("Error occurred in model training")
        raise CustomException(e, sys)