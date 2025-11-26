curl -X POST 'https://corbenol-wine-price-predictor.hf.space/predict' \
-H 'Content-Type: application/json' \
-d '{
    "country": "france",
    "description": "Tart and snappy, the flavors of lime flesh and rind dominate",
    "province": "Alsace",
    "millesime": "2022" 
}'
