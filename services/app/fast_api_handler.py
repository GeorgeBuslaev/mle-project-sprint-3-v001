# coding: utf-8
"""Класс FastApiHandler, который обрабатывает запросы API."""

from catboost import CatBoostRegressor
import joblib
import pandas as pd
import csv
import geopy.distance

class FastApiHandler:
    """Класс FastApiHandler, который обрабатывает запрос и возвращает предсказание."""

    def __init__(self):
        """Инициализация переменных класса."""

        # Типы параметров запроса для проверки
        self.param_types = {
            "user_id": str,
            "model_params": dict
            }
        self.dataset_path = "./data/initial_data.csv"
        self.dataset_prep_path = "./data/initial_data_prep.csv"
        self.features_name_path = "fin_assets/union_features.csv"
        self.model_path = "models/fitted_model.pkl"
        self.tranformer_path = "models/fitted_transformer.pkl"
        self.load_assets(assets_path=self.features_name_path)
        self.load_model()
        self.load_transformer(tranformer_path=self.tranformer_path)
        self.MOSCOW_0KM = (55.755806972288845, 37.61769525232508)

        # Необходимые параметры для предсказаний стоимости объектов недвижимости
        self.required_model_params = [
            'floor', 'kitchen_area', 'living_area', 'rooms', 'is_apartment',
            'studio', 'total_area', 'build_year', 'building_type_int',
            'flats_count', 'ceiling_height', 'longitude', 'latitude',
            'floors_total', 'has_elevator'
        ]

    def load_dataset(self, dataset_path: str):
        """Загружаем перечень отобранных признаков:
            dataset_path (str): Путь до файла с обучающим датасетом.
        """
        try:
            self.dataset = pd.read_csv(dataset_path, index_col = 0)
            self.dataset = self.dataset.astype({'floor' : 'int',
                                                'kitchen_area' : 'float',
                                                'living_area' : 'float',
                                                'rooms' : 'float',
                                                'is_apartment' : 'object',
                                                'studio' : 'float',
                                                'total_area' : 'float',
                                                'build_year' : 'float',
                                                'building_type_int' : 'object',
                                                'flats_count' : 'int',
                                                'ceiling_height' : 'float',
                                                'longitude' : 'float',
                                                'latitude' : 'float',
                                                'floors_total' : 'int',
                                                'has_elevator' : 'object',
                                                'price' : 'float'
                                               })
            
        except Exception as e:
            print(f"Failed to load dataset: {e}")
        
        try:
            dataset_prep = pd.read_csv(self.dataset_prep_path)
            self.dataset_prep = dataset_prep.astype({'floor' : 'int',                                                                                   'kitchen_area' : 'float',
                            'is_apartment' : 'object',
                            'total_area' : 'float',
                            'building_type_int' : 'object',
                            'flats_count' : 'int',
                            'ceiling_height' : 'float',
                            'longitude' : 'float',
                            'latitude' : 'float',
                            'floors_total' : 'int',
                            'has_elevator' : 'object',
                            'centr_dist_km' : 'float',
                            'building_age' : 'int',
                            'first_floor' : 'object',
                            'last_floor' : 'object',
                            'kitchen_area_ratio' : 'float',
                            'living_area_ratio' : 'float',
                            'avg_room_area' : 'float'
                           })
            
        except Exception as e:
            print(f"Failed to load dataset_prep: {e}")
    
    def dataset_processing(self):
        def duplicated_features():
            data = self.dataset
            feature_cols = data.columns.drop(['price']).tolist()
            # Выведем объекты дублирующие друг друга
            is_duplicated_features = data.duplicated(subset=feature_cols, keep='first')
            data = data[~is_duplicated_features]
            return data
        
        def features_maker(row): # Убран лишний код
            '''Функция создает признаки возраста здания
            и расстояния объекта от центра Москвы'''
        
            # Тут задал константу MOSCOW_0KM
            coords = (row['latitude'], row['longitude'])
            row['centr_dist_km'] = (geopy.distance.geodesic(self.MOSCOW_0KM, coords).km)
            row['building_age'] = 2024-row['build_year']
        
            row['first_floor'] = row['floor'] == 1
            row['last_floor'] = row['floor'] == row['floors_total']
            return row
        
        def outliers_filter(data, threshold):
            '''Функция выделяет индексы объектов с выбросами
            по заданным признакам и threshold'''
            num_cols = data.drop('price', axis=1).select_dtypes(['float', 'int']).columns
            potential_outliers = pd.DataFrame()
            for col in num_cols:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3-Q1
                margin = threshold*IQR
                lower = Q1 - margin
                upper = Q3 + margin 
                potential_outliers[col] = ~data[col].between(lower, upper)
                outliers = potential_outliers.any(axis=1)
            return outliers
            
        data = duplicated_features()
        data = data[~outliers_filter(data=data, threshold=1.5)]  
        data = data.apply(features_maker, axis = 1)
        data['kitchen_area_ratio'] = data['kitchen_area'] / data['total_area']
        data['living_area_ratio'] = data['living_area'] / data['total_area']
        data['avg_room_area'] = data['living_area']/data['rooms']
        data = data.astype({'floor' : 'int',
                            'kitchen_area' : 'float',
                            'is_apartment' : 'object',
                            'total_area' : 'float',
                            'building_type_int' : 'object',
                            'flats_count' : 'int',
                            'ceiling_height' : 'float',
                            'longitude' : 'float',
                            'latitude' : 'float',
                            'floors_total' : 'int',
                            'has_elevator' : 'object',
                            'centr_dist_km' : 'float',
                            'building_age' : 'int',
                            'first_floor' : 'object',
                            'last_floor' : 'object',
                            'kitchen_area_ratio' : 'float',
                            'living_area_ratio' : 'float',
                            'avg_room_area' : 'float'
                           })
        data.to_csv('data/initial_data_prep.csv', index=None)
        self.dataset = data
        return data

    def data_processing_params(self, model_params): #: dict):
        """Предсказываем вероятность прогнозирования стоимости.

        Args:
            model_params (dict): Параметры для модели.

        Returns:
            list - Предобработанные параметры.
        """
        row = pd.DataFrame(model_params, index=[0])
        coords = (row['latitude'].values, row['longitude'].values)
        row['centr_dist_km'] = (geopy.distance.geodesic(self.MOSCOW_0KM, coords).km)
        row['building_age'] = 2024-row['build_year']
        row['first_floor'] = row['floor'] == 1
        row['last_floor'] = row['floor'] == row['floors_total']
        row['kitchen_area_ratio'] = row['kitchen_area'] / row['total_area']
        row['living_area_ratio'] = row['living_area'] / row['total_area']
        row['avg_room_area'] = row['living_area']/row['rooms']
        row = row.astype({'floor' : 'int',
                            'kitchen_area' : 'float',
                            'is_apartment' : 'object',
                            'total_area' : 'float',
                            'building_type_int' : 'object',
                            'flats_count' : 'int',
                            'ceiling_height' : 'float',
                            'longitude' : 'float',
                            'latitude' : 'float',
                            'floors_total' : 'int',
                            'has_elevator' : 'object',
                            'centr_dist_km' : 'float',
                            'building_age' : 'int',
                            'first_floor' : 'object',
                            'last_floor' : 'object',
                            'kitchen_area_ratio' : 'float',
                            'living_area_ratio' : 'float',
                            'avg_room_area' : 'float'
                            })
        self.row = row
        return row

    
    def fit_model(self):
    	# загрузите результат предыдущего шага: inital_data.csv
        data = self.dataset_prep
        test_size = 0.1
        X_train_val, X_test, y_train_val, y_test = train_test_split(
        data.drop('price', axis=1),
        data['price'],
        test_size=test_size,
        random_state=23
    )
    	# реализуем основную логику шага с использованием гиперпараметров
        cat_features = data.select_dtypes(include='object')
        potential_binary_features = cat_features.nunique() == 2
        binary_cat_features = cat_features[potential_binary_features[potential_binary_features].index]
        other_cat_features = cat_features[potential_binary_features[~potential_binary_features].index]
        num_features = data.drop('price', axis=1).select_dtypes(['float', 'int'])
    
        preprocessor = ColumnTransformer(
            [
            ('binary', OneHotEncoder(drop='if_binary', sparse_output=False), binary_cat_features.columns.tolist()),
            ('cat', CatBoostEncoder(return_df=False), other_cat_features.columns.tolist()),
            ('num', StandardScaler(), num_features.columns.tolist())
            ],
            remainder='drop',
            verbose_feature_names_out=False
        )
    
        model = CatBoostRegressor(verbose=500, random_state=23)   
    
        pipeline = Pipeline(
            [
                ('preprocessor', preprocessor),
                ('model', model)
            ]
        )
        pipeline.fit(X_train_val, y_train_val) 
    	# сохраните обученную модель в models/fitted_model.pkl
        self.model = model
        os.makedirs('models', exist_ok=True) # создание директории, если её ещё нет
        with open('models/fitted_model.pkl', 'wb') as fd:
            joblib.dump(pipeline, fd)
        return pipeline
    
    def load_model(self):
        """Загружаем обученную модель прогнозирования стоимости.
        Args:
            model_path (str): Путь до модели.
        """
        try:
            with open(self.model_path, 'rb') as fd:
                self.model = joblib.load(fd)
        except Exception as e:
            print(f"Failed to load model: {e}")

    def predict_params(self):
        try:
            self.load_model()
            prediction = self.model.predict(self.row)
        except Exception as e:
            print(f"Error with model loading: {e}")
        return prediction


    def load_assets(self, assets_path: str):
        """Загружаем перечень отобранных признаков:
            assets_path (str): Путь до файла с названиями признаков.
        """
        try:
            self.features_name = pd.read_csv(assets_path, index_col = 0).values
        except Exception as e:
            print(f"Failed to load assets: {e}")

    def load_transformer(self, tranformer_path: str):
        """Загружаем обученный трансформер для генерации признаков:
            model_path (str): Путь до модели.
        """
        try:
            self.processor = joblib.load(tranformer_path)
        except Exception as e:
            print(f"Failed to load transformer: {e}")

    def real_estate_transform_predict(self, model_params: dict) -> float:
        """Подготовка данных и прогнозирование стоимости объекта.

        Args:
            model_params (dict): Параметры для модели.

        Returns:
            float - Прогнозная стоимость.
        """
        data = self.data_processing(model_params)
        self.data = data

        try:
            transformed_data = self.processor.transform(self.data)
        except Exception as e:
            print(f"Failed to transform data: {e}")
        
        try:
            price = self.model.predict(transformed_data)
            return price
        except Exception as e:
            print(f"Failed to predict price: {e}")
      
    def check_required_query_params(self, query_params: dict) -> bool:
        """Проверяем параметры запроса на наличие обязательного набора параметров.
        
        Args:
            query_params (dict): Параметры запроса.
        
        Returns:
                bool: True - если есть нужные параметры, False - иначе
        """
        if "user_id" not in query_params or "model_params" not in query_params:
            return False
        
        if not isinstance(query_params["user_id"], self.param_types["user_id"]):
            return False
                
        if not isinstance(query_params["model_params"], self.param_types["model_params"]):
            return False
        return True
    
    def check_required_model_params(self, model_params: dict) -> bool:
        """Проверяем параметры пользователя на наличие обязательного набора.
    
        Args:
            model_params (dict): Параметры объекта для предсказания.
    
        Returns:
            bool: True - если есть нужные параметры, False - иначе
        """
        if set(model_params.keys()) == set(self.required_model_params):
            return True
        return False
    
    def validate_params(self, params: dict) -> bool:
        """Разбираем запрос и проверяем его корректность.
    
        Args:
            params (dict): Словарь параметров запроса.
    
        Returns:
            - **dict**: Cловарь со всеми параметрами запроса.
        """
        if self.check_required_query_params(params):
            print("All query params exist")
        else:
            print("Not all query params exist")
            return False
        
        if self.check_required_model_params(params["model_params"]):
            print("All model params exist")
        else:
            print("Not all model params exist")
            return False
        return True
		
    def handle(self, params):
        """Функция для обработки запросов API параметров входящего запроса.
    
        Args:
            params (dict): Словарь параметров запроса.
    
        Returns:
            - **dict**: Словарь, содержащий результат выполнения запроса.
        """
        try:
            # Валидируем запрос к API
            if not self.validate_params(params):
                print("Error while handling request")
                response = {"Error": "Problem with parameters"}
            else:
                model_params = params["model_params"]
                user_id = params["user_id"]
                print(f"Predicting for user_id: {user_id} and model_params:\n{model_params}")
                # Получаем предсказания модели
                self.data_processing_params(model_params)
                price = self.predict_params().item()
                response = {
                    "user_id": user_id, 
                    "price": price
                    }
        except Exception as e:
            print(f"Error while handling request: {e}")
            return {"Error": "Problem with request"}
        else:
            return response
        
if __name__ == "__main__":

    # Создаем тестовый запрос
    test_params = {
	    "user_id": "123",
        "model_params": {
            "floor": 6,
            "kitchen_area": 8.7,
            "living_area": 33.2,
            "rooms": 2,
            "is_apartment": false,
            "studio": false,
            "total_area": 56.1,
            "build_year": 2006,
            "building_type_int": 1,
            "flats_count": 158,
            "ceiling_height": 3.0,
            "longitude": 37.66888427734375,
            "latitude": 55.78324508666992,
            "floors_total": 7,
            "has_elevator": true
                }
                }

    # Создаем обработчик запросов для API
    handler = FastApiHandler()

    # Делаем тестовый запрос
    response = handler.handle(test_params)
    print(f"Response: {response}")