import json
import numpy as np
import pandas as pd
import pickle
import psycopg2
import requests
import torch
from torch.autograd import Variable
import torch.nn as nn

from gensim.models import Word2Vec
from io import BytesIO
from PIL import Image
from torchvision.transforms import transforms


class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input


class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))


def get_w2v_dict():
    with open('./models/w2v.pkl', 'rb') as f:
        d = pickle.load(f)
    return d


def get_image_vectors():
    with open('./models/image_vectors.pkl', 'rb') as f:
        d = pickle.load(f)
    return d


def save_image_vectors(d):
    with open('./models/image_vectors.pkl', 'wb') as f:
        pickle.dump(d, f)


def items_on_stock(erp_shop_id):
    conn = psycopg2.connect(dbname='e2_availability',
                            user='readonly',
                            password='DB4Cg3Tb2XXDJwQc',
                            host='e2-pg-ro.ennergiia.com')
    with conn.cursor() as cursor:
        query = """select product_item_id
                    from stocks
                    where erp_shop_id = %s
                      and count != 0"""
        cursor.execute(query, (erp_shop_id,))
        items = cursor.fetchall()
        return [str(i[0]) for i in items]


def items_on_stock_from_json(erp_shop_id):
    with open(f'./test/items_{erp_shop_id}.json', 'r') as f:
        items = json.load(f)
    return items


def product_items_attributes(items):
    conn = psycopg2.connect(dbname='e2_catalog',
                            user='readonly',
                            password='DB4Cg3Tb2XXDJwQc',
                            host='e2-pg-ro.ennergiia.com')

    placeholders = ', '.join(['%s'] * len(items))
    with conn.cursor() as cursor:
        query = """select pi.product_item_id,  generic_product_id,
                (select jj -> 'value' ->> 'ru' sex
                from (select jsonb_array_elements(doc -> 'product_options' -> 'facets_options') jj
                      from product_items
                      where product_item_id = pi.product_item_id) tt
                where jj -> 'property_name' ->> 'ru' = 'Пол'
                limit 1) word, 
                (pi.doc -> 'brand' -> 'name' ->> 'ru') brand,
               (select jj -> 'value' ->> 'ru' size
                from (select jsonb_array_elements(doc -> 'product_options' -> 'facets_options') jj
                      from product_items
                      where product_item_id = pi.product_item_id) tt
                where jj -> 'property_name' ->> 'ru' = 'Размер'
                limit 1) size
        from product_items pi
        where pi.business_id = 'ENRG'
        and pi.product_item_id IN ({})""".format(placeholders)

        cursor.execute(query, tuple(items))
        rows = cursor.fetchall()

    with conn.cursor() as cursor:
        query = """select id, doc -> 'name' ->> 'ru'
                    from generic_products
                    where business='ENRG'"""

        cursor.execute(query)
        gp_rows = cursor.fetchall()

    with conn.cursor() as cursor:
        query = """select product_item_id,
                           image_group_id
                    from product_items_images
                    where business_id='ENRG'"""

        cursor.execute(query)
        image_rows = cursor.fetchall()

    df = pd.DataFrame(rows, columns=['product_item_id', 'generic_product_id',
                                     'word', 'brand', 'size']).merge(pd.DataFrame(gp_rows,
                                                                columns=['generic_product_id', 'generic_product_name']),
                                                    on='generic_product_id').merge(pd.DataFrame(image_rows,
                                                                                                columns=[
                                                                                                    'product_item_id',
                                                                                                    'image_group']),
                                                                                   on='product_item_id')
    df['word'] = df['generic_product_name'] + '_' + df['word'] + '_' + df['brand'] + '_' + df['size']
    df = df[~df['word'].isna()][['product_item_id', 'word', 'image_group', 'brand', 'size']]
    return df


def get_w2v_model():
    return Word2Vec.load('./models/w2v_nocolor.model')


def compute_distances(y, x):
    dists = np.sum(np.abs(y - x), axis=1)
    return dists


def load_stylenet():
    stylenet = nn.Sequential(  # Sequential,
        nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1)),
        nn.ReLU(),
        nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
        nn.ReLU(),
        nn.Dropout(0.25),
        nn.MaxPool2d((4, 4), (4, 4)),
        nn.BatchNorm2d(64, 0.001, 0.9, True),
        nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1)),
        nn.ReLU(),
        nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1)),
        nn.ReLU(),
        nn.Dropout(0.25),
        nn.MaxPool2d((4, 4), (4, 4)),
        nn.BatchNorm2d(128, 0.001, 0.9, True),
        nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1)),
        nn.ReLU(),
        nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1)),
        nn.ReLU(),
        nn.Dropout(0.25),
        nn.MaxPool2d((4, 4), (4, 4)),
        nn.BatchNorm2d(256, 0.001, 0.9, True),
        nn.Conv2d(256, 128, (1, 1)),
        nn.ReLU(),
        Lambda(lambda x: x.view(x.size(0), -1)),  # Reshape,
        nn.Sequential(Lambda(lambda x: x.view(1, -1) if 1 == len(x.size()) else x), nn.Linear(3072, 128))
    )

    stylenet.load_state_dict(torch.load('./models/stylenet.model.pth'))
    return stylenet.eval()


def get_descriptor(stylenet, image_group):
    url = f'https://cdn.ennergiia.com/new-images/ennergiia-catalog/ig{image_group}/0/360x540.webp'
    response = requests.get(url)
    pic = Image.open(BytesIO(response.content))

    normalize = transforms.Normalize(
       mean=[0.5657177752729754, 0.5381838567195789, 0.4972228365504561],
       std=[0.29023818639817184, 0.2874722565279285, 0.2933830104791508]
    )
    preprocess = transforms.Compose([
       transforms.Resize((256, 384)),
       transforms.ToTensor(),
       normalize
    ])

    img_tensor = preprocess(pic)
    img_tensor.unsqueeze_(0)

    img_variable = Variable(img_tensor)
    return stylenet.forward(img_variable)[0].clone().detach().numpy()