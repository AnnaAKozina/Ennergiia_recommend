from utils import *
import warnings
from time import time
import sys

warnings.filterwarnings("ignore")

WORD_SIMILARITY = 0.9
RECOMMEND_NUM = 8
ERP_SHOP_ID = '9701'

stylenet = load_stylenet()
model_w2v = get_w2v_model()
image_vectors = get_image_vectors()


class ERP_Stock:
    def __init__(self, erp):
        self.erp = erp

    def get_erp_items_attributes(self):
        stock = items_on_stock(self.erp)
        df = product_items_attributes(stock)
        image_vectors = get_image_vectors()
        df['image_vector'] = df['image_group'].map(image_vectors)
        if df[df['image_vector'].isna()].shape:
            try:
                for im in df[df['image_vector'].isna()]['image_group'].unique():
                    image_vectors[im] = get_descriptor(stylenet, im)
                save_image_vectors(image_vectors)
                df['image_vector'] = df['image_group'].map(image_vectors)
            except:
                df = df[~df['image_vector'].isna()]
        return df


class Recommendation:
    def __init__(self, item, erp_items_attributes):
        self.product_item_id = item
        self.erp_items_attributes = erp_items_attributes

    def get_recommendations(self):
        df = self.erp_items_attributes
        try:
            self.word, self.image_group, self.image_vector, \
            self.brand, self.size = df[df['product_item_id'] ==
                                                                self.product_item_id][['word',
                                                                                       'image_group',
                                                                                       'image_vector',
                                                                                       'brand',
                                                                                       'size']].values[0]
            self.image_group = f'https://cdn.ennergiia.com/new-images/ennergiia-catalog/ig{self.image_group}/0/360x540.webp'
        except:
            return {'item': [], 'recommendation': []}
        try:
            y_df = df[df['word'].isin([w for w, s in model_w2v.wv.most_similar(self.word) if s > WORD_SIMILARITY]
                                      + [self.word]) &
                      (df['product_item_id'] != self.product_item_id) &
                      (df['image_group'] != self.image_group)]
            start = time()
            y_df['dist'] = compute_distances(list(y_df['image_vector']), self.image_vector)
            print('get distances time ', time() - start, file=sys.stderr)
            y_df['image_group'] = df['image_group'].apply(lambda x:
                                         f'https://cdn.ennergiia.com/new-images/ennergiia-catalog/ig{x}/0/360x540.webp')
            n = min(y_df.shape[0], RECOMMEND_NUM)
            jsn = {'item': [{'product_item_id': self.product_item_id,
                             'image_group': self.image_group, 'word': self.word,
                             'brand': self.brand, 'size': self.size}],
                   'recommendation': y_df.sort_values('dist')[['product_item_id',
                                                               'image_group',
                                                               'word',
                                                               'brand',
                                                               'size'
                                                               ]].iloc[:n].to_dict(orient='records')}
        except:
            jsn = {'item': [{'product_item_id': self.product_item_id,
                             'image_group': self.image_group, 'word': self.word,
                             'brand': self.brand, 'size': self.size}],
                   'recommendation': []}
        return jsn
