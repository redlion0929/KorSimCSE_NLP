import torch
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    EvalPrediction,
    BertModel,
    AutoModel
)
import numpy as np
import pandas as pd
import faiss
import pickle
from tqdm import tqdm
import argparse
from torch import nn

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * (np.linalg.norm(b)))

class simcse_review:
    def __init__(self, mode):
        super().__init__()
        self.check_point = 'sup_cse_klue_bert'
        self.device = torch.device("cuda:" + str(0) if torch.cuda.is_available() else "cpu")
        #모델 선언
        self.config = AutoConfig.from_pretrained(self.check_point)
        self.tokenizer = AutoTokenizer.from_pretrained(self.check_point)
        self.model = AutoModel.from_pretrained(self.check_point, config=self.config)

        self.model.resize_token_embeddings(len(self.tokenizer))

        #multi gpu
        self.model = nn.DataParallel(self.model)
        self.model = self.model.to(self.device)

        #데이터
        self.product_name = []
        self.review = []
        self.mode = mode
        if self.mode == 'tablet':
            self.df = pd.read_csv('inference_data/tablet.csv')
        elif self.mode == 'notebook':
            self.df = pd.read_csv('inference_data/notebook.csv')

    def make_index(self):
        self.product_index = faiss.IndexFlatL2(768)
        self.product_index = faiss.IndexIDMap2(self.product_index)

        self.review_index = faiss.IndexFlatL2(768)

        key = self.df['제품 이름'].unique()
        self.product_info = {}

        for idx, k in tqdm(enumerate(key)):
            new_df = self.df[self.df['제품 이름'] == k]['내용']
            review_num = min(40, len(new_df))
            review_ = new_df.tolist()[:review_num]
            review_list = [k + '[SEP]' + content for content in review_ ]
            input = self.tokenizer(review_list, padding = True , truncation = True, return_tensors='pt')
            input = input.to(self.device)
            embeddings = self.model(**input, output_hidden_states = True, return_dict = True).pooler_output
            product_embeddings = torch.mean(embeddings, 0)

            self.review_index.add(embeddings.cpu().detach().numpy())
            self.product_index.add_with_ids(product_embeddings.unsqueeze(0).cpu().detach().numpy(), np.array([idx]))

            self.product_info[idx] = k


        with open('inference_data/'+self.mode +'.pickle', 'wb') as fw:
            pickle.dump(self.product_info, fw)

        faiss.write_index(self.review_index, 'inference_data/review_index'+self.mode)
        faiss.write_index(self.product_index, 'inference_data/product_index'+self.mode)

    def load_index(self, review_dir, product_dir, product_info_dir):
        self.review_index = faiss.read_index(review_dir)
        self.product_index = faiss.read_index(product_dir)
        with open(product_info_dir, 'rb') as fr:
            self.product_info = pickle.load(fr)

    def view_product(self):
        for k,v in self.product_info.items():
            print("ID : ", k, "제품 이름 :", v)

    def keyword_search(self, keyword, top_k):
        input = self.tokenizer(keyword, padding=True, truncation=True, return_tensors='pt')
        input = input.to(self.device)
        embeddings = self.model(**input, output_hidden_states=True, return_dict=True).pooler_output

        distances, indices = self.product_index.search(embeddings.cpu().detach().numpy().astype('float32'), top_k)

        print('keyword : ', keyword)
        print(top_k, '개의 검색 결과 출력')
        for idx in indices[0]:
            print('제품 이름 : ', self.product_info[idx])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--category', required=False, default ='tablet', help='tablet or notebook')
    parser.add_argument('--keyword', required=True, help='keyword')

    args = parser.parse_args()

    search_class = simcse_review(args.category)
    search_class.make_index()
    #search_class.load_index('inference_data/review_index'+args.category, 'inference_data/product_index'+args.category, 'inference_data/'+args.category + '.pickle')
    #search_class.view_product()
    search_class.keyword_search(args.keyword, 5)
