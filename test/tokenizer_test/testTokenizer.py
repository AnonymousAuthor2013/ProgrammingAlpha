from programmingalpha.tokenizers import (
    BertTokenizer,SpacyTokenizer,CoreNLPTokenizer, TransfoXLTokenizer, OpenAIGPTTokenizer, GPT2Tokenizer)

import programmingalpha


s="What were the first areas of research into Artificial Intelligence and what were some pupeer? spacy.tokeniz()? or jet.neety()"

print('\n test bert tokenizer')
tokenizer=BertTokenizer.from_pretrained(programmingalpha.BertBasePath)
print(tokenizer.tokenize(s))

print('\n test spacy tokenizer')
tokenizer=SpacyTokenizer()
print(tokenizer.tokenize(s))

print('\n test corenlp tokenizer')
tokenizer=CoreNLPTokenizer()
print(tokenizer.tokenize(s))

print('\n test transformerXL tokenizer')
tokenizer=TransfoXLTokenizer()
print(tokenizer.tokenize(s))

print('\n test openai tokenizer')
tokenizer=OpenAIGPTTokenizer(programmingalpha.openAIGPTPath+"/openai-gpt-vocab.json",programmingalpha.openAIGPTPath+"/openai-gpt-merges.txt")
print(tokenizer.tokenize(s))

print('\n test gpt2 tokenizer')
tokenizer=GPT2Tokenizer(programmingalpha.GPT2Path+"/gpt2-vocab.json",programmingalpha.GPT2Path+"/gpt2-merges.txt")
print(tokenizer.encode(s))


