import regex as re
from pyentrp import entropy as ent
import programmingalpha
class PreprocessPostContent(object):
    code_snippet=re.compile(r"<pre><code>.*?</code></pre>")
    code_insider=re.compile(r"<code>.*?</code>")
    plain_text=re.compile(r"<.*?>")
    remove_code_quote=re.compile(r"\(.*?\)")
    paragraph=re.compile(r"<p>.*?</p>")
    @staticmethod
    def filterNILStr(s):
        filterFunc=lambda s: s and s.strip()
        s=' '.join(list(filter(filterFunc,s.split('\n'))))
        s=' '.join(list(filter(filterFunc,s.split())))

        return s

    def getPlainTxt(self,raw_txt,keep_emb_code=False):
        # return a list of paragraphs of plain text

        if keep_emb_code:
            txt=re.sub(self.code_snippet," ",raw_txt)
        else:
            txt=re.sub(self.code_insider," ",raw_txt)

        paragraphs=self.getParagraphs(txt)
        paragraphs=list(filter(self.filterNILStr,paragraphs))
        #print("filtered",paragraphs)

        texts=[]
        for txt in paragraphs:
            txt=re.sub(self.plain_text," ",txt)
            texts.append(txt)
        if len(paragraphs)==0:
            txt=re.sub(self.plain_text," ",txt)
            texts.append(txt)
        return texts

    def getEmCodes(self,raw_txt):
        txt=re.sub(self.code_snippet,'',raw_txt)
        emcodes=self.code_insider.findall(txt)

        return emcodes

    def getCodeSnippets(self,raw_txt):
        snippets=self.code_snippet.findall(raw_txt)

        return snippets

    def getParagraphs(self,raw_txt):
        return re.findall(self.paragraph,raw_txt)


class QuestionTextInformationExtractor(object):
    def __init__(self,maxClip,tokenizer):
        self.maxClip=maxClip

        self.informationEnt=ent.shannon_entropy
        self.dela_min=0.1

        self.processor=PreprocessPostContent()
        self.tokenizer=tokenizer

    def keepStrategy(self,paragraphs):
        texts=[]
        if len(paragraphs)<3:
            return paragraphs

        texts.append(paragraphs[0])
        texts.append(paragraphs[1])
        texts.append(paragraphs[-1])
        count=len(texts[0])+len(texts[1])+len(texts[2])

        if count > self.maxClip:
            return texts

        #encoding
        encoder={}
        for sent in paragraphs:
            for tok in sent:
                encoder[tok]=len(encoder)
        Delta=float('inf')
        seqs=[encoder[tok] for sent in texts for tok in sent]
        entSum=self.informationEnt(seqs)

        exploreSet=set([i for i in range(len(paragraphs))])
        exploreSet.remove(0)
        exploreSet.remove(1)
        exploreSet.remove(len(paragraphs)-1)


        while count<self.maxClip and Delta> self.dela_min and len(exploreSet)>0:
            selected_sent=None
            maxDela=float('-inf')
            for i in exploreSet:
                sent=paragraphs[i]
                newSeqs=seqs+[encoder[tok] for tok in sent]
                newEnt=self.informationEnt(newSeqs)
                if entSum==0:
                    entSum=1e-4
                Delta=(newEnt-entSum)/entSum
                if Delta >maxDela:
                    selected_sent=i
                    maxDela=Delta


            if selected_sent is not None:
                Delta=maxDela
                texts.append(paragraphs[selected_sent])
                exploreSet.remove(selected_sent)
                entSum=newEnt
            else:
                break

        return texts


    def clipText(self,q):
        title, body =q["Title"],q["Body"]

        #print("body=>",body)
        #print("title=>",title)

        title=" ".join(self.processor.getPlainTxt(title))
        paragraphs=self.processor.getPlainTxt(body)
        paragraphs.insert(0,title)

        #print(len(paragraphs))

        count=0
        for i in range(len(paragraphs)):
            paragraphs[i]=self.tokenizer.tokenize(paragraphs[i])
            count+=len(paragraphs[i])

        if count>self.maxClip:
            #logger.info("max sequence length exceed {}, and paragrahs {}".format(count,len(paragraphs)))
            paragraphs=self.keepStrategy(paragraphs)
            #logger.info("after clipping strategy=>{},{}".format(sum(map(lambda seqs:len(seqs),paragraphs)),len(paragraphs)))

        for i in range(len(paragraphs)):
            paragraphs[i]=" ".join(paragraphs[i])

        return paragraphs

class AnswerTextInformationExtractor(object):
    def __init__(self,maxClip,tokenizer):
        self.maxClip=maxClip

        self.informationEnt=ent.shannon_entropy
        self.dela_min=0.1

        self.processor=PreprocessPostContent()
        self.tokenizer=tokenizer

        with open(programmingalpha.ConfigPath+"pattern-answer","r") as f:
            self.patterns=f.readlines()

        self.keepdOrder=False

    def keepStrategy(self,paragraphs):
        texts=[]
        if len(paragraphs)<3:
            return paragraphs
        texts.append(paragraphs[0])
        texts.append(paragraphs[1])
        texts.append(paragraphs[-1])
        count=len(texts[0])+len(texts[1])+len(texts[2])



        if count > self.maxClip:
            return texts

        #encoding
        encoder={}
        for sent in paragraphs:
            for tok in sent:
                encoder[tok]=len(encoder)
        Delta=float('inf')
        seqs=[encoder[tok] for sent in texts for tok in sent]
        entSum=self.informationEnt(seqs)

        exploreSet=set([i for i in range(len(paragraphs))])
        exploreSet.remove(0)
        exploreSet.remove(1)
        exploreSet.remove(len(paragraphs)-1)

        tagger=-1
        for tagger in exploreSet:
            find=False
            for pat in self.patterns:
                if pat in paragraphs[tagger]:
                    find=True
                    break
            if find:
                break
        if tagger>0:
            exploreSet.remove(tagger)
        texts.insert(2,paragraphs[tagger])


        while count<self.maxClip and Delta> self.dela_min and len(exploreSet)>0:
            selected_sent=None
            maxDela=float('-inf')
            for i in exploreSet:
                sent=paragraphs[i]
                newSeqs=seqs+[encoder[tok] for tok in sent]
                newEnt=self.informationEnt(newSeqs)
                if entSum==0:
                    entSum=1e-4
                Delta=(newEnt-entSum)/entSum
                if Delta >maxDela:
                    selected_sent=i
                    maxDela=Delta
            if selected_sent is not None:
                Delta=maxDela
                texts.append(paragraphs[selected_sent])
                exploreSet.remove(selected_sent)
                entSum=newEnt
            else:
                break

        if self.keepdOrder==True:
            texts.clear()
            for i in range(len(paragraphs)):
                if i not in exploreSet:
                    texts.append(paragraphs[i])

        return texts


    def clipText(self,ans):

        paragraphs=self.processor.getPlainTxt(ans)

        count=0
        for i in range(len(paragraphs)):
            paragraphs[i]=self.tokenizer.tokenize(paragraphs[i])
            count+=len(paragraphs[i])

        if count>self.maxClip:
            #logger.info("max sequence length exceed {}, and paragrahs {}".format(count,len(paragraphs)))
            paragraphs=self.keepStrategy(paragraphs)
            #logger.info("after clipping strategy=>{},{}".format(sum(map(lambda seqs:len(seqs),paragraphs)),len(paragraphs)))

        for i in range(len(paragraphs)):
            paragraphs[i]=" ".join(paragraphs[i])

        return paragraphs

if __name__ == '__main__':
    s="<neural-networks><backpropagation><terminology><definitions>"
    pros=PreprocessPostContent()
    print(pros.getPlainTxt(s))
    print(", ".join(s.replace("<","").replace(">"," ").strip().split(" ")))
    print(s)

    '''
    <p>In particular, an embedded computer (with limited resources) analyzes live video stream from a traffic camera, trying to pick good frames that contain license plate numbers of passing cars. Once a plate is located, the frame is handed over to an OCR library to extract the registration and use it further.</p>

    <p>In my country two types of license plates are in common use - rectangular (the typical) and square - actually, somewhat rectangular but "higher than wider", with the registration split over two rows.</p>
    
    <p>(there are some more types, but let us disregard them; they are a small percent and usually belong to vehicles that lie outside our interest.)</p>
    
    <p>Due to the limited resources and need for rapid, real-time processing, the maximum size of the network (number of cells and connections) the system can handle is fixed.</p>
    
    <p>Would it be better to split this into two smaller networks, each recognizing one type of registration plates, or will the larger single network handle the two types better?</p>

    '''
    print(s)

    ss=pros.getPlainTxt(s)
    print(len(ss))
    [print(s1) for s1 in ss]

    texts=["","<p>hi ,you</p>","<p>what is it?</p>","","","<p>good</p>"]

    newtexts=pros.getPlainTxt(" ".join(texts))
    print(newtexts)
