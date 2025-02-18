import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict_car_price(self, features):
        try:
            model_path = os.path.join('artifacts', 'car', 'model.pkl')
            preprocessor_path = os.path.join('artifacts', 'car', 'preprocessor.pkl')
            
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds

        except Exception as e:
            raise CustomException(e, sys)

    def predict_house_price(self, features):
        try:
            model_path = os.path.join('artifacts', 'house', 'model.pkl')
            preprocessor_path = os.path.join('artifacts', 'house', 'preprocessor.pkl')
            
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return np.expm1(preds)  # Reverse log transformation for house prices

        except Exception as e:
            raise CustomException(e, sys)

class CarData:
    def __init__(
        self,
        brand: str,
        model: str,
        model_year: int,
        milage: float,
        fuel_type: str,
        engine: str,
        transmission: str,
        ext_col: str,
        int_col: str,
        accident: str,
        clean_title: str
    ):
        self.brand = brand
        self.model = model
        self.model_year = model_year
        self.milage = milage
        self.fuel_type = fuel_type
        self.engine = engine
        self.transmission = transmission
        self.ext_col = ext_col
        self.int_col = int_col
        self.accident = accident
        self.clean_title = clean_title

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "brand": [self.brand],
                "model": [self.model],
                "model_year": [self.model_year],
                "milage": [self.milage],
                "fuel_type": [self.fuel_type],
                "engine": [self.engine],
                "transmission": [self.transmission],
                "ext_col": [self.ext_col],
                "int_col": [self.int_col],
                "accident": [self.accident],
                "clean_title": [self.clean_title]
            }
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)

class HouseData:
    def __init__(
        self,
        MSSubClass: int,
        MSZoning: str,
        LotArea: int,
        LotShape: str,
        BldgType: str,
        HouseStyle: str,
        OverallQual: int,
        OverallCond: int,
        YearBuilt: int,
        RoofStyle: str,
        Exterior1st: str,
        Foundation: str,
        BsmtQual: str,
        BsmtExposure: str,
        BsmtFinType1: str,
        HeatingQC: str,
        CentralAir: str,
        GrLivArea: int,
        BedroomAbvGr: int,
        KitchenQual: str,
        TotRmsAbvGrd: int,
        Functional: str,
        Fireplaces: int,
        GarageType: str,
        GarageFinish: str,
        GarageCars: int,
        GarageArea: int,
        PavedDrive: str,
        WoodDeckSF: int,
        OpenPorchSF: int,
        Fence: str,
        MoSold: int,
        YrSold: int
    ):
        # Store all the house features
        self.MSSubClass = MSSubClass
        self.MSZoning = MSZoning
        self.LotArea = LotArea
        # ... store all other features ...

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "MSSubClass": [self.MSSubClass],
                "MSZoning": [self.MSZoning],
                "LotArea": [self.LotArea],
                # ... add all other features ...
            }
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)

if __name__=="__main__":
    # Test car prediction
    test_car = CarData(
        brand="Toyota",
        model="Camry",
        model_year=2020,
        milage=30000,
        fuel_type="Gasoline",
        engine="203.0HP 2.5L 4 Cylinder Engine Gasoline Fuel",
        transmission="Automatic",
        ext_col="White",
        int_col="Black",
        accident="None",
        clean_title="Yes"
    )
    car_df = test_car.get_data_as_data_frame()
    predict_pipeline = PredictPipeline()
    car_price = predict_pipeline.predict_car_price(car_df)
    print(f"Predicted car price: ${car_price[0]:,.2f}")