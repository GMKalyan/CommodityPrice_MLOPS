from fastapi import FastAPI, HTTPException
from .models import CarPredictionInput, HousePredictionInput, PredictionResponse
from src.pipeline.predict_pipeline import PredictPipeline, CarData, HouseData

app = FastAPI(
    title="Price Prediction API",
    description="API for predicting car and house prices",
    version="1.0.0"
)

predict_pipeline = PredictPipeline()

@app.get("/")
def read_root():
    return {"message": "Welcome to Price Prediction API"}

@app.post("/predict/car", response_model=PredictionResponse)
async def predict_car_price(input_data: CarPredictionInput):
    try:
        # Convert input to CarData format
        car_data = CarData(
            brand=input_data.brand,
            model=input_data.model,
            model_year=input_data.model_year,
            milage=input_data.milage,
            fuel_type=input_data.fuel_type,
            engine=input_data.engine,
            transmission=input_data.transmission,
            ext_col=input_data.ext_col,
            int_col=input_data.int_col,
            accident=input_data.accident,
            clean_title=input_data.clean_title
        )
        
        # Get prediction
        features = car_data.get_data_as_data_frame()
        prediction = predict_pipeline.predict_car_price(features)
        
        return PredictionResponse(predicted_price=float(prediction[0]))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/house", response_model=PredictionResponse)
async def predict_house_price(input_data: HousePredictionInput):
   try:
       # Convert input to HouseData format
       house_data = HouseData(
           MSSubClass=input_data.MSSubClass,
           MSZoning=input_data.MSZoning,
           LotArea=input_data.LotArea,
           LotShape=input_data.LotShape,
           BldgType=input_data.BldgType,
           HouseStyle=input_data.HouseStyle,
           OverallQual=input_data.OverallQual,
           OverallCond=input_data.OverallCond,
           YearBuilt=input_data.YearBuilt,
           RoofStyle=input_data.RoofStyle,
           Exterior1st=input_data.Exterior1st,
           Foundation=input_data.Foundation,
           BsmtQual=input_data.BsmtQual,
           BsmtExposure=input_data.BsmtExposure,
           BsmtFinType1=input_data.BsmtFinType1,
           HeatingQC=input_data.HeatingQC,
           CentralAir=input_data.CentralAir,
           GrLivArea=input_data.GrLivArea,
           BedroomAbvGr=input_data.BedroomAbvGr,
           KitchenQual=input_data.KitchenQual,
           TotRmsAbvGrd=input_data.TotRmsAbvGrd,
           Functional=input_data.Functional,
           Fireplaces=input_data.Fireplaces,
           GarageType=input_data.GarageType,
           GarageFinish=input_data.GarageFinish,
           GarageCars=input_data.GarageCars,
           GarageArea=input_data.GarageArea,
           PavedDrive=input_data.PavedDrive,
           WoodDeckSF=input_data.WoodDeckSF,
           OpenPorchSF=input_data.OpenPorchSF,
           Fence=input_data.Fence,
           MoSold=input_data.MoSold,
           YrSold=input_data.YrSold
       )
       
       # Get prediction
       features = house_data.get_data_as_data_frame()
       prediction = predict_pipeline.predict_house_price(features)
       
       return PredictionResponse(predicted_price=float(prediction[0]))
   except Exception as e:
       raise HTTPException(status_code=500, detail=str(e))