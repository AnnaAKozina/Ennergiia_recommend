# Recommender model based on sales history and product images
INPUT: test/customers.json - Users carts with city/shop mark (Format: [{"user_id": 0, "erp_shop_id":"9701", "cart_items": [...]}])<br /> 
       test/items_{erp_shop_id}.json - List of product_item_id that are available in the city/shop (Format: [...])<br /> 
OUTPUT: test/recommendations.json ([{"user_id": 0, "item": "316942", "recommendation": [...]},
                                    {"user_id": 0, "item": "9033925", "recommendation": [...]},
                                    ...}])

EXAMPLES:<br />
input - 
![](images/im1.png)

output - 
![](images/rec1.png)

input - 
![](images/im2.png)

output - 
![](images/rec2.png)

input - 
![](images/im3.png)

output - 
![](images/rec3.png)
