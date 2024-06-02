from transformers import BertModel, BertTokenizer


def get_model_for_embedding(model_name):
    return BertModel.from_pretrained(model_name)


def get_tokenizer_for_embedding(model_name):
    return BertTokenizer.from_pretrained(model_name)