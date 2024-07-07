import requests
import pandas as pd
import time
import json

# Функция для отправки post-запроса.
# Выбирает n случайных строк из файла weather.csv.gz, упаковывает в json и отправляет
# На указанный адрес
def main(ip = "127.0.0.1", port = 4601, i = 10, t = 5):
    dataset = pd.read_csv("services/app/data/initial_data.csv", index_col = 0).dropna().sample(n=i, random_state=23)
    dataset = dataset.astype({'floor' : 'str',
                                        'kitchen_area' : 'str',
                                        'living_area' : 'str',
                                        'rooms' : 'str',
                                        'is_apartment' : 'str',
                                        'studio' : 'str',
                                        'total_area' : 'str',
                                        'build_year' : 'str',
                                        'building_type_int' : 'str',
                                        'flats_count' : 'str',
                                        'ceiling_height' : 'str',
                                        'longitude' : 'str',
                                        'latitude' : 'str',
                                        'floors_total' : 'str',
                                        'has_elevator' : 'str',
                                        'price' : 'str'
                                        })
    dataset.drop(["price"], axis=1, inplace=True)

    for id in dataset.index:
        data = dict(dataset.loc[id, :])
        data = json.dumps(data)
        response = requests.post(
            f"http://{ip}:{port}/api/real_estate/{id}",
            data=data,
        )
        print(f"Status code:\n{response.status_code}")
        print(f"Result:\n{response.json()}")

        time.sleep(t)

if __name__ == "__main__":
    main()