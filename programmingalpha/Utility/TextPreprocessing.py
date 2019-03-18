import regex as re
from pyentrp import entropy as ent
import programmingalpha
from textblob import TextBlob
from bert_serving.client import BertClient

class PreprocessPostContent(object):
    code_snippet=re.compile(r"<pre><code>.*?</code></pre>")
    code_insider=re.compile(r"<code>.*?</code>")
    html_tag=re.compile(r"<.*?>")
    comment_quote=re.compile(r"\(.*?\)")
    href_resource=re.compile(r"<a.*?>.*?</a>")
    paragraph=re.compile(r"<p>.*?</p>")
    equation1=re.compile(r"\$.*?\$")
    equation2=re.compile(r"\$\$.*?\$\$")
    integers=re.compile(r"^-?[1-9]\d*$")
    floats=re.compile(r"^-?([1-9]\d*\.\d*|0\.\d*[1-9]\d*|0?\.0+|0)$")
    operators=re.compile(r"[><\+\-\*/=]")
    email=re.compile(r"\w+([-+.]\w+)*@\w+([-.]\w+)*\.\w+([-.]\w+)*")
    web_url=re.compile(r"[a-zA-z]+://[^\s]*")

    @staticmethod
    def filterNILStr(s):
        filterFunc=lambda s: s and s.strip()
        s=' '.join( filter( filterFunc, s.split() ) ).strip()

        return s

    def __init__(self):
        self.max_quote_rate=1.5
        self.max_quote_diff=5
        self.min_words_sent=5
        self.min_words_paragraph=10
        self.max_number_pragraph=5
        self.num_token="[NUM]"
        self.code_token="[CODE]"
        self.max_code=5

    def remove_quote(self,txt):
        cleaned=[]
        for sent in TextBlob(txt).sentences:
            s=sent.string
            s_c=re.sub(self.comment_quote,"",s)
            s_c_words=TextBlob(s_c).words
            if len(s_c_words)==0 or len(sent.words)/len(s_c_words)>self.max_quote_rate or \
                    len(sent.words)-len(s_c_words)>self.max_quote_diff:
                continue
            cleaned.append(s_c)

        return " ".join(cleaned)

    def remove_href(self,txt):
        cleaned=[]
        for sent in TextBlob(txt).sentences:
            s=sent.string
            s_c=re.sub(self.href_resource,"",s)
            if sent.words!=TextBlob(s_c).words:
                continue
            cleaned.append(s)

        return " ".join(cleaned)


    def remove_code(self,txt):
        cleaned=re.sub(self.code_snippet,"",txt)
        return cleaned

    def remove_emb_code(self,txt):
        cleand=[]

        txt=re.sub(self.code_insider," %s "%self.code_token,txt)

        for sent in TextBlob(txt).sentences:
            s_c=re.sub(r"[CODE]","",sent.string)
            s_c_words=TextBlob(s_c).words
            if  len(s_c_words)==0 or len(sent.words)/len(s_c_words)>self.max_quote_rate or \
                len(sent.words)/len(s_c_words)>self.max_code:
                continue


            cleand.append(sent.string)

        return " ".join(cleand)

    def remove_equation(self,txt):
        cleaned=[]
        for sent in TextBlob(txt).sentences:
            s=sent.string
            s_c=re.sub(self.equation2,"",s)
            s_c=re.sub(self.equation1,"",s_c)
            if sent.words!=TextBlob(s_c).words:
                continue
            cleaned.append(s)

        return " ".join(cleaned)

    def remove_numbers(self,txt):
        cleaned=[]
        for sent in TextBlob(txt).sentences:

            s=sent.string
            s_tokens=s.split()
            s_tokens=map(lambda t:re.sub(self.floats,"",t),s_tokens)
            s_tokens=map(lambda t:re.sub(self.integers,"",t),s_tokens)
            s_c=" ".join(s_tokens)
            #print("in number=>",s,"--vs--",s_c)

            if len(sent.words)-len(TextBlob(s_c).words)>self.max_number_pragraph:
                continue

            s_tokens=s.split()
            s_tokens=map(lambda t:re.sub(self.floats," %s "%self.num_token,t),s_tokens)
            s_tokens=map(lambda t:re.sub(self.integers," %s "%self.num_token,t),s_tokens)
            s_c=" ".join(s_tokens)

            cleaned.append(s_c)

        cleaned=" ".join(cleaned)

        return cleaned

    def remove_operators(self,txt):
        cleaned=[]
        for sent in TextBlob(txt).sentences:
            s=sent.string
            s_c=re.findall(self.operators,s)
            if len(s_c)>3:
                continue
            cleaned.append(s)

        return " ".join(cleaned)

    def remove_hmtltag(self,txt):
        cleaned=re.sub(self.html_tag, "", txt)
        return cleaned

    def remove_email(self,txt):
        cleaned=[]
        for sent in TextBlob(txt).sentences:
            s=sent.string
            s_c=re.sub(self.email,"",s)
            if sent.words!=TextBlob(s_c).words:
                continue
            cleaned.append(s)
        return " ".join(cleaned)

    def remove_url(self,txt):
        cleaned=[]
        for sent in TextBlob(txt).sentences:
            s=sent.string
            s_c=re.sub(self.web_url,"",s)
            if sent.words!=TextBlob(s_c).words:
                continue
            cleaned.append(s)
        return " ".join(cleaned)

    def remove_useless(self,txt):
        cleaned=[]

        for sent in TextBlob(txt).sentences:
            #print("rm sent=>",sent)
            if len(sent.words)<self.min_words_sent:
                continue
            if sent[-1] not in ('.','?','!') and len(sent.words)<2*self.min_words_sent:
                continue
            cleaned.append(sent.string)

        return " ".join(cleaned)

    def __process(self,txt):
        print("\nprocess=>",txt)

        txt=self.remove_emb_code(txt)
        print("after rm codecs=>",txt)


        txt=self.remove_href(txt)
        print("after rm href=>",txt)

        txt=self.remove_hmtltag(txt)
        print("after rm tag=>",txt)

        txt=self.remove_email(txt)
        print("atfer rm email=>",txt)

        txt=self.remove_url(txt)
        print("atfer rm url=>",txt)

        txt=self.remove_equation(txt)
        print("atfer rm eq=>",txt)

        txt=self.remove_quote(txt)
        print("after rm quote=>",txt)

        txt=self.remove_numbers(txt)
        print("after rm num=>",txt)

        txt=self.remove_operators(txt)
        print("after rm ops=>",txt)

        txt=self.remove_useless(txt)
        print("after rm use=>",txt)

        return txt

    def getPlainTxt(self,raw_txt):
        # return a list of paragraphs of plain text
        #filter code

        txt=self.remove_code(raw_txt)

        paragraphs=self.getParagraphs(txt)

        texts=[]
        for p in paragraphs:
            cleaned=self.__process(p)
            if len(cleaned.split())<self.min_words_paragraph:
                continue
            texts.append(self.filterNILStr(cleaned))

        return texts

    def getParagraphs(self,raw_txt):
        raw_txt=self.filterNILStr(raw_txt)
        paragraphs_candiates=re.findall(self.paragraph,raw_txt)
        paragraphs_candiates=[p[3:-4] for p in paragraphs_candiates if len(p[3:-4])>0]
        paragraphs=[]
        for p in paragraphs_candiates:
            if len(TextBlob(p).words)<self.min_words_paragraph:
                continue
            paragraphs.append(p)
        return paragraphs


class QuestionTextInformationExtractor(object):
    def __init__(self,maxClip,tokenizer):
        self.maxClip=maxClip

        self.informationEnt=ent.shannon_entropy
        self.dela_min=0.1

        self.processor=PreprocessPostContent()
        self.encoder=BertClient(ip="sugon-gpu-3",port=5555)
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


    def clipText(self,question_txts):

        paragraphs=self.processor.getPlainTxt(question_txts)

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

    ans='''
    <p>In linear regression, the word "linear" applies to the coefficients: the dependence between $Y$ and the coefficients is linear. This does not mean the dependence between $Y$ and $X$ is linear.</p>
    <p>i think a linear regression is handling the regression problem using math:</p>
    <p>Assume $X$ is a one dimensional variable. Basic linear regression is (I omit the noise and intercept for simplicity):
    $$Y=\beta X$$</p>
    <p> as a result of this, you better visit www.baidu.com to get full answer</p>
        <p> as a result of this, you better visit https://www.baidu.com to get full answer</p>

    <p>But this is still linear regression:</p>
    
    <p>$$Y=\beta_1 X+\beta_2X^2+\beta_3\log(X)$$</p>
    
    <p>The latter is the same as basic linear regression with feature vector $(X,X^2,\log(X))$ instead of $X$.</p>
    
    <p>By linear regression I assume that you mean simple linear regression. The difference is in the number of independent explanatory variables you use to model your dependent variable.</p>

    <p>Simple linear regression</p>
    
    <p>$Y=\beta X+\beta_0$</p>    
    
    <p>I have been given 65 values. 57 of these data values are quarterly results and 8 are the holdback data to be used. </p>
    
    <p>I have to do: 
    - Regression with Dummy variables with a linear trend cycle component </p>
    
    <p>Does anyone know what to do as my results aren't making much sense? </p>
    
    <p>For the first part - I obviously split the data into dummy variables for the relevant quarters (Q1-Q4). </p>
    
    <p>I then performed regression analysis - linear. But all my values are extremely large and not significant. Also Q2 has been listed as 'excluded variables' in the results? I have followed the steps and I am unsure why this has happened. </p>
    
    <p>Then I thought of removing Q4, due to multi-collinearity but again the values are still quite large (>.450). </p>
    
    <p>Not sure if I am doing something wrong at the start (especially with the excluded variables aspect) </p>
    
    <p>Anybody got any idea? This is driving me nuts</p>
    
    <p>Update: It won't let me comment back on the main page for some reason. </p>
    
    <p>The data set was given to us:
    "It is a quarterly series of total consumer lending. It is not seasonally adjusted.
    The first 57 data values for modelling and choose the remaining 8 data values as holdback data to test your models."</p>
    
    <p>The data is: 
    16180
    17425
    43321
    3214 4324 5435 41 143221 545 45</p>
    
    <p>It has to be SPSS generated: as it is not like.</p>
    
    <p>Email primarybeing12@hotmail.co.uk - not letting me respond to people. Thanks for any help!</p>
    
    <p>Doing the ARIMA forecasting is the next step (which I understand). I have to do regression on the linear/non-linear for this question</p>
    
    <p>If I was to use time, time^2, Q1, Q2, Q3 + lagged variables.</p>
    
    <p>Would I use lagged variables 1-3? Also, I understand the rest, but what benefit does using lagged variables do? As I said, feel free to e-mail me if you can.</p>
    
    <p>(I'm not positive about this, but...)</p>

    <p>AS3 uses a non-deterministic garbage collection. Which means that unreferenced memory will be freed up whenever the runtime feels like it (typically not unless there's a reason to run, since it's an expensive operation to execute). This is the same approach used by most modern garbage collected languages (like C# and Java as well).</p>
    
    <p>Assuming there are no other references to the memory pointed to by <code>byteArray</code> or the items within the array itself, the memory will be freed at some point after you exit the scope where <code>byteArray</code> is declared.</p>
    
    <p>You can force a garbage collection, though you really shouldn't. If you do, do it only for testing... if you do it in production, you'll hurt performance much more than help it.</p>
    
    <p>To force a GC, try (yes, twice):</p>
    
    <pre><code>flash.system.System.gc();
    flash.system.System.gc();
    </code></pre>
    
    <p><a href="http://www.craftymind.com/2008/04/09/kick-starting-the-garbage-collector-in-actionscript-3-with-air/" rel="noreferrer">You can read more here</a>.</p>

    '''

    s='''<p><code>(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*|"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])</code></p>'''
    ans+=s
    print(TextBlob(" ".join(s.split())).sentences)
    print(TextBlob(" ".join(s.split())).sentences)
    pros=PreprocessPostContent()
    texts=pros.getPlainTxt(ans)
    [print(text) for text in texts]
    exit(10)

    ss=pros.getPlainTxt(ans)
    print(len(ss))
    for s1 in ss:
        print(s1.sentences)

    print("clipped")
    from programmingalpha.tokenizers import BertTokenizer
    tokenizer=BertTokenizer.from_pretrained(programmingalpha.BertBasePath)
    ansExt=AnswerTextInformationExtractor(50,tokenizer)
    clipped=ansExt.clipText(ans)
    for p in clipped:
        print(p)
