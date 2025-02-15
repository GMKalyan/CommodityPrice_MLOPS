import os
import sys
import logging
from src.exception import CustomException
from src.logger import logging  # Make sure this import is correct
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from pymongo import MongoClient

@dataclass
class DataIngestionConfig:
    train_data_path: str
    test_data_path: str
    raw_data_path: str

class CarDataIngestionConfig:
    def __init__(self):
        self.train_data_path = os.path.join('artifacts', 'car', 'train.csv')
        self.test_data_path = os.path.join('artifacts', 'car', 'test.csv')
        self.raw_data_path = os.path.join('artifacts', 'car', 'raw.csv')

class HouseDataIngestionConfig:
    def __init__(self):
        self.train_data_path = os.path.join('artifacts', 'house', 'train.csv')
        self.test_data_path = os.path.join('artifacts', 'house', 'test.csv')
        self.raw_data_path = os.path.join('artifacts', 'house', 'raw.csv')

class DataIngestion:
    def __init__(self, database_name="price_prediction", collection_name=None):
        self.database_name = database_name
        self.collection_name = collection_name
        if collection_name == "car_data":
            self.ingestion_config = CarDataIngestionConfig()
        else:
            self.ingestion_config = HouseDataIngestionConfig()

    def get_data_from_mongodb(self):
        """Read data from MongoDB"""
        try:
            logging.info("Reading data from MongoDB started")
            client = MongoClient('mongodb://localhost:27017/')
            db = client[self.database_name]
            collection = db[self.collection_name]
            
            # Convert MongoDB cursor to DataFrame
            df = pd.DataFrame(list(collection.find({}, {'_id': 0})))
            logging.info(f"Read {len(df)} records from MongoDB")
            
            client.close()
            return df
            
        except Exception as e:
            logging.error("Error in reading data from MongoDB")
            raise CustomException(e, sys)

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # Read data from MongoDB
            df = self.get_data_from_mongodb()
            logging.info("Read the dataset as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Train test split initiated")
            
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)

if __name__=="__main__":
    # Test car data ingestion
    car_ingestion = DataIngestion(collection_name="car_data")
    car_train_path, car_test_path = car_ingestion.initiate_data_ingestion()
    logging.info(f"Car data ingestion completed. Train path: {car_train_path}, Test path: {car_test_path}")

    # Test house data ingestion
    house_ingestion = DataIngestion(collection_name="house_data")
    house_train_path, house_test_path = house_ingestion.initiate_data_ingestion()
    logging.info(f"House data ingestion completed. Train path: {house_train_path}, Test path: {house_test_path}")

if __name__=="__main__":
    try:
        print("Starting data ingestion process...")
        logging.info("Starting data ingestion process...")

        # Test car data ingestion
        print("Initiating car data ingestion...")
        car_ingestion = DataIngestion(collection_name="car_data")
        car_train_path, car_test_path = car_ingestion.initiate_data_ingestion()
        print(f"Car data ingestion completed. Train path: {car_train_path}, Test path: {car_test_path}")
        logging.info(f"Car data ingestion completed. Train path: {car_train_path}, Test path: {car_test_path}")

        # Test house data ingestion
        print("Initiating house data ingestion...")
        house_ingestion = DataIngestion(collection_name="house_data")
        house_train_path, house_test_path = house_ingestion.initiate_data_ingestion()
        print(f"House data ingestion completed. Train path: {house_train_path}, Test path: {house_test_path}")
        logging.info(f"House data ingestion completed. Train path: {house_train_path}, Test path: {house_test_path}")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        logging.error(f"Error occurred: {str(e)}")