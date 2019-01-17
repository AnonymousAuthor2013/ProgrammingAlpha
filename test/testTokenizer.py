from programmingalpha.tokenizers.bert_tokenizer import BertTokenizer
from programmingalpha.tokenizers.corenlp_tokenizer import CoreNLPTokenizer
from programmingalpha.tokenizers.spacy_tokenizer import SpacyTokenizer
import programmingalpha


s="What were the first areas of research into Artificial Intelligence and what were some pupeer?"

print('\n test bert tokenizer')
tokenizer=BertTokenizer.from_pretrained(programmingalpha.BertBasePath)
print(tokenizer.tokenize(s).words(True))
print(tokenizer.tokenize(s).untokenize())

print('\n test spacy tokenizer')
tokenizer=SpacyTokenizer()
print(tokenizer.tokenize(s).words(True))
print(tokenizer.tokenize(s).untokenize())

#tokenizer=CoreNLPTokenizer()

#print(tokenizer.tokenize(s))
