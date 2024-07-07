
#! /usr/bin/bash
curl -X 'POST' \
  'http://localhost:4601/api/real_estate/123' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "floor": 6,
  "kitchen_area": 8.7,
  "living_area": 33.2,
  "rooms": 2,
  "is_apartment": False,
  "studio": False,
  "total_area": 56.1,
  "build_year": 2006,
  "building_type_int": 1,
  "flats_count": 158,
  "ceiling_height": 3,
  "longitude": 37.66888427734375,
  "latitude": 55.78324508666992,
  "floors_total": 7,
  "has_elevator": True
}'