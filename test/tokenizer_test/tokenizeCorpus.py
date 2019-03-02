from programmingalpha.tokenizers import (CoreNLPTokenizer,SpacyTokenizer,BertTokenizer,GPT2Tokenizer,OpenAIGPTTokenizer,TransfoXLTokenizer)
import logging,argparse
import programmingalpha
from multiprocessing import Pool


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)



def init():
    global tokenizer, subTokenizer
    global blankFilter
    if args.tokenizer =='m1':
        tokenizer=CoreNLPTokenizer()
        subTokenizer=SpacyTokenizer()
    elif args.tokenizer=='m2':
        tokenizer=BertTokenizer.from_pretrained(programmingalpha.BertBasePath,do_lower_case=True)
    elif args.tokenizer=='m3':
        tokenizer=GPT2Tokenizer.from_pretrained(programmingalpha.GPT2Path)
    elif args.tokenizer=='m4':
        tokenizer=OpenAIGPTTokenizer.from_pretrained(programmingalpha.openAIGPTPath)
    elif args.tokenizer=='m5':
        tokenizer=TransfoXLTokenizer.from_pretrained(programmingalpha.transformerXL)

    blankFilter=lambda s:s and s.strip()

def tokenize(text):
    if args.tokenizer=='m1':
        try:
            words=tokenizer.tokenize(text)
        except:
            words=subTokenizer.tokenize(text)
    else:
        words=tokenizer.tokenize(text)
    return words

def tokenizeCore(texts):
    tokens=[]
    #print(len(texts),texts[0])
    for text in texts:
        #print("text is",text)
        words=tokenize(text)

        words=list(filter(blankFilter,words))
        tokens.append(' '.join(words)+'\n')
        #print("tokens are",tokens[0])
        #exit(10)
    return tokens

def readlines(filename,n_line):
    lines=[]
    index=0
    with open(filename,"r") as f:
        for line in f:
            if index<args.skip_lines:
                index+=1
                continue
            lines.append(line)
            if len(lines)>=n_line:
                yield lines
                lines.clear()

    if len(lines)>0:
        yield lines

def tokenizeCorpus(filename,output,batch_size):
    f=open(output,"a")
    cache=[]

    workers=Pool(args.processes,initializer=init)

    count=args.skip_lines
    lines_batch_reader=readlines(filename,batch_size)

    for lines in lines_batch_reader:
        process_num=len(lines)//args.processes
        line_batch=[lines[i:i+process_num] for i in range(0,len(lines),process_num)]
        for batch in workers.map(tokenizeCore,line_batch):
            cache.extend(batch)

        f.writelines(cache)
        count+=len(cache)
        logger.info("processed {} lines".format(count))
        cache.clear()


    workers.close()
    workers.join()

    if len(cache)>0:
        f.writelines(cache)
        logger.info("processed {} lines".format(count+len(cache)))
        cache.clear()

    f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=10000)
    parser.add_argument('--processes', type=int, default=20)
    parser.add_argument('--skip_lines', type=int, default=0)
    parser.add_argument('--input', type=str, default=programmingalpha.DataPath+"Corpus/stackexchange.txt")
    parser.add_argument('--output', type=str, default=programmingalpha.DataPath+"Corpus/stackexchange-tokenized-BPE.txt")
    parser.add_argument('--tokenizer', type=str, default="m1",help="m1:spacy/corenlp,m2:bert,m3:gpt2,m4:openai,m5:transformerxl")

    args = parser.parse_args()

    tokenizeCorpus(args.input,args.output,args.batch_size)
