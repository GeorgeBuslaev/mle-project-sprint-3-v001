from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from fast_api_handler import FastApiHandler
import logging

# Создаем приложение Fast API
app = FastAPI()

# Создаем обработчик запросов для API
app.handler = FastApiHandler()

# Пример модели для параметров
class ModelParams(BaseModel):
    floor: int
    kitchen_area: float
    living_area: float
    rooms: int
    is_apartment: bool
    studio: bool
    total_area: float
    build_year: int
    building_type_int: int
    flats_count: int
    ceiling_height: float
    longitude: float
    latitude: float
    floors_total: int
    has_elevator: bool

'''class ResponseModel(BaseModel):
    user_id: str
    price: float'''

@app.get("/")
def read_root():
    return {"Hello": "World"}

# Открытый маршрут для проверки статуса
@app.get("/service-status")
def health_check():
    return {"status": "ok"}

# Предсказание цены объекта недвижимости
@app.post("/api/real_estate/{user_id}") #, response_model=ResponseModel) 
def get_prediction_for_item(
    user_id: str,
    model_params: ModelParams = Body(default = {
        "floor" : 6,
        "kitchen_area": 8.7,
        "living_area": 33.2,
        "rooms": 2,
        "is_apartment": False,
        "studio": False,
        "total_area": 56.1,
        "build_year": 2006,
        "building_type_int": 1,
        "flats_count": 158,
        "ceiling_height": 3.0,
        "longitude": 37.66888427734375,
        "latitude": 55.78324508666992,
        "floors_total": 7,
        "has_elevator": True
    }
          )
          ):
    """Функция для получения прогнозной стоимости объекта недвижимости.

    Args:
        user_id (str): Идентификатор пользователя.
        model_params (ModelParams): Параметры объекта недвижимости, которые мы должны подать в модель.

    Returns:
        ResponseModel: Прогнозная стоимость объекта недвижимости.
    """
    try:
        # Декодирование параметров и логика обработки
        all_params = {
            "user_id": user_id,
            "model_params": model_params.dict()
        }

        # Здесь должна происходить логика обработки и предсказания модели
        predicted_value = app.handler.handle(all_params)

        return {"predicted_value": predicted_value}
    
    except Exception as e:
        logging.error(f"Error while processing the prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
