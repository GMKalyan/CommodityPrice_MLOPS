import sys
from dataclasses import dataclass
import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer  # Added KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler, OrdinalEncoder  # Added StandardScaler and OrdinalEncoder
from scipy.stats import skew
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

@dataclass
class HouseDataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'house', 'preprocessor.pkl')

class HouseDataTransformation:
    def __init__(self):
        self.data_transformation_config = HouseDataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            # Define numeric and categorical columns
            numeric_features = [
                'LotFrontage', 'LotArea', 'YearBuilt', 'YearRemodAdd', 
                'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
                'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
                'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
                'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
                'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea',
                'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
                'ScreenPorch', 'PoolArea', 'MiscVal'
            ]

            categorical_features = [
                'MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape',
                'LandContour', 'Utilities', 'LotConfig', 'LandSlope',
                'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
                'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
                'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond',
                'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
                'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC',
                'CentralAir', 'Electrical', 'KitchenQual', 'Functional',
                'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
                'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature',
                'SaleType', 'SaleCondition'
            ]

            # Numeric Pipeline
            numeric_transformer = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', RobustScaler())
                ]
            )

            # Categorical Pipeline
            categorical_transformer = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='constant', fill_value='None')),
                    ('onehot', OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore')),
                ]
            )

            logging.info(f"Categorical columns: {categorical_features}")
            logging.info(f"Numerical columns: {numeric_features}")

            # Combine both pipelines
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features),
                    ('cat', categorical_transformer, categorical_features)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def create_feature_engineering(self, df):
        try:
            logging.info("Started feature engineering")
            
            # Create new features
            df['YrBltAndRemod'] = df['YearBuilt'] + df['YearRemodAdd']
            df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
            df['Total_sqr_footage'] = (df['BsmtFinSF1'] + df['BsmtFinSF2'] +
                                     df['1stFlrSF'] + df['2ndFlrSF'])
            df['Total_Bathrooms'] = (df['FullBath'] + (0.5 * df['HalfBath']) +
                                   df['BsmtFullBath'] + (0.5 * df['BsmtHalfBath']))
            df['Total_porch_sf'] = (df['OpenPorchSF'] + df['3SsnPorch'] +
                                  df['EnclosedPorch'] + df['ScreenPorch'] +
                                  df['WoodDeckSF'])
            
            # Create binary features
            df['haspool'] = df['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
            df['has2ndfloor'] = df['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
            df['hasgarage'] = df['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
            df['hasbsmt'] = df['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
            df['hasfireplace'] = df['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
            
            logging.info("Completed feature engineering")
            return df

        except Exception as e:
            raise CustomException(e, sys)

    def handle_skewed_features(self, df):
        try:
            logging.info("Started handling skewed features")
            
            # Find numeric columns
            numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
            numeric_features = [col for col in df.columns if df[col].dtype in numeric_dtypes]
            
            # Calculate skewness
            skewed_features = df[numeric_features].apply(lambda x: skew(x)).sort_values(ascending=False)
            high_skew = skewed_features[skewed_features > 0.5]
            skew_index = high_skew.index

            # Apply Box-Cox transformation
            for feature in skew_index:
                df[feature] = boxcox1p(df[feature], boxcox_normmax(df[feature] + 1))
                
            logging.info("Completed handling skewed features")
            return df

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data completed")

            # Drop Id column
            if 'Id' in train_df.columns:
                train_df = train_df.drop(['Id'], axis=1)
            if 'Id' in test_df.columns:
                test_df = test_df.drop(['Id'], axis=1)

            # Remove outliers from training data
            train_df = train_df[train_df.GrLivArea < 4500].reset_index(drop=True)

            # Feature Engineering
            train_df = self.create_feature_engineering(train_df)
            test_df = self.create_feature_engineering(test_df)
            
            # Handle Skewed Features
            train_df = self.handle_skewed_features(train_df)
            test_df = self.handle_skewed_features(test_df)
            
            logging.info("Feature engineering and skewness handling completed")

            # Handle target variable
            target_column_name = "SalePrice"
            target_feature_train_df = np.log1p(train_df[target_column_name])
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            input_feature_test_df = test_df

            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()

            logging.info("Applying preprocessing object on training and testing datasets.")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = input_feature_test_arr

            logging.info("Saved preprocessing object.")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
        
@dataclass
class CarDataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'car', 'preprocessor.pkl')

class CarDataTransformation:
    def __init__(self):
        self.data_transformation_config = CarDataTransformationConfig()
        self.luxury_brands = ['Mercedes-Benz', 'BMW', 'Audi', 'Lexus', 'Tesla']

    def get_data_transformer_object(self):
        try:
            # Update columns to match actual dataset
            numeric_features = [
                'model_year', 
                'milage', 
                'engine_hp'  # Removed 'price' as it's our target
            ]

            categorical_features = [
                'brand',
                'model', 
                'fuel_type',
                'transmission',
                'ext_col',
                'int_col',
                'accident',
                'clean_title'
            ]

            # KNN Imputation for missing values
            numeric_transformer = Pipeline(
                steps=[
                    ('imputer', KNNImputer(n_neighbors=5)),
                    ('scaler', StandardScaler())
                ]
            )

            # Categorical Pipeline
            categorical_transformer = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                    ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
                ]
            )

            logging.info(f"Categorical columns: {categorical_features}")
            logging.info(f"Numerical columns: {numeric_features}")

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features),
                    ('cat', categorical_transformer, categorical_features)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def create_features(self, df):
        try:
            logging.info("Started feature engineering for car data")
            # Clean engine data - extract numeric values
            def extract_engine_hp(x):
                try:
                    if pd.isna(x):
                        return None
                    if 'HP' in str(x):
                        return float(str(x).split('HP')[0].strip())
                    return None
                except:
                    return None

        # Add engine processing
            df['engine_hp'] = df['engine'].apply(extract_engine_hp)
            df = df.drop('engine', axis=1)  # Drop original engine column
            
            df['rare_fuel_type'] = df['fuel_type'].apply(lambda x: 0 if x in ['Petrol', 'Diesel'] else 1)
            df['is_automatic'] = df['transmission'].apply(lambda x: 1 if x == 'Automatic' else 0)
            df['has_accident_history'] = df['accident'].apply(lambda x: 1 if x != 'Unknown' and x != 'None' else 0)
            df['color_match'] = df.apply(lambda row: 1 if row['ext_col'] == row['int_col'] else 0, axis=1)
            df['is_luxury_brand'] = df['brand'].apply(lambda x: 1 if x in self.luxury_brands else 0)
            
            logging.info("Completed feature engineering")
            return df

        except Exception as e:
            raise CustomException(e, sys)

    def remove_outliers_iqr(self, df, column):
        try:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1   
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR    
            df_out = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
            return df_out
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read car train and test data completed")
            
            # Debug information
            logging.info(f"Train columns: {train_df.columns.tolist()}")
            logging.info(f"Test columns: {test_df.columns.tolist()}")

            # Drop ID column
            if 'id' in train_df.columns:
                train_df = train_df.drop(['id'], axis=1)
            if 'id' in test_df.columns:
                test_df = test_df.drop(['id'], axis=1)

            # Remove outliers
            train_df = self.remove_outliers_iqr(train_df, 'milage')
            train_df = self.remove_outliers_iqr(train_df, 'price')
            train_df.reset_index(drop=True, inplace=True)

            # Create features
            train_df = self.create_features(train_df)
            test_df = self.create_features(test_df)

            # Split features and target
            target_column_name = "price"
            target_feature_train_df = train_df[target_column_name]
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)

            # Get preprocessing object
            preprocessing_obj = self.get_data_transformer_object()

            # Transform data
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = input_feature_test_arr

            # Save preprocessor
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)        

# if __name__=="__main__":
#     try:
#         # Test Car Price Transformation only
#         logging.info("\n>>>>>>> Starting Car Price Data Transformation >>>>>")
#         car_obj = CarDataTransformation()
#         car_train_path = os.path.join('artifacts', 'car', 'train.csv')
#         car_test_path = os.path.join('artifacts', 'car', 'test.csv')
        
#         # Debug: Print file paths
#         logging.info(f"Car train path: {car_train_path}")
#         logging.info(f"Car test path: {car_test_path}")
        
#         # Load and check data
#         train_df = pd.read_csv(car_train_path)
#         test_df = pd.read_csv(car_test_path)
#         logging.info(f"Train DataFrame Shape: {train_df.shape}")
#         logging.info(f"Test DataFrame Shape: {test_df.shape}")
#         logging.info(f"Train columns: {train_df.columns.tolist()}")
#         logging.info(f"Test columns: {test_df.columns.tolist()}")
        
#         car_train_arr, car_test_arr, _ = car_obj.initiate_data_transformation(car_train_path, car_test_path)
#         logging.info("Car data transformation completed successfully")
#         logging.info(f"Car Transformed Train array shape: {car_train_arr.shape}")
#         logging.info(f"Car Transformed Test array shape: {car_test_arr.shape}")

#     except Exception as e:
#         logging.error("Error occurred in data transformation")
#         raise CustomException(e, sys)

if __name__=="__main__":
    try:
        # Test Car Price Transformation only
        logging.info("\n>>>>>>> Starting Car Price Data Transformation >>>>>")
        car_obj = CarDataTransformation()
        car_train_path = os.path.join('artifacts', 'car', 'train.csv')
        car_test_path = os.path.join('artifacts', 'car', 'test.csv')
        
        # Verify directories
        logging.info(f"Checking if car artifacts directory exists: {os.path.exists(os.path.join('artifacts', 'car'))}")
        
        # Load and check data
        train_df = pd.read_csv(car_train_path)
        test_df = pd.read_csv(car_test_path)
        logging.info("Step 1: Data Loading - Complete")
        logging.info(f"Train DataFrame Shape: {train_df.shape}")
        logging.info(f"Test DataFrame Shape: {test_df.shape}")
        
        # Attempt transformation
        logging.info("Step 2: Starting Data Transformation")
        car_train_arr, car_test_arr, preprocessor_path = car_obj.initiate_data_transformation(car_train_path, car_test_path)
        
        # Verify preprocessor file
        logging.info(f"Step 3: Checking preprocessor file")
        logging.info(f"Preprocessor path: {preprocessor_path}")
        logging.info(f"Preprocessor file exists: {os.path.exists(preprocessor_path)}")
        
        logging.info("Car data transformation completed successfully")
        logging.info(f"Car Transformed Train array shape: {car_train_arr.shape}")
        logging.info(f"Car Transformed Test array shape: {car_test_arr.shape}")

    except Exception as e:
        logging.error(f"Error occurred in data transformation: {str(e)}")
        raise CustomException(e, sys)