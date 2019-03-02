import regex as re

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
        texts=[]
        for txt in paragraphs:
            txt=self.filterNILStr(txt)
            txt=re.sub(self.plain_text," ",txt)
            texts.append(txt)
        if len(paragraphs)==0:
            txt=self.filterNILStr(txt)
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

if __name__ == '__main__':
    s="<neural-networks><backpropagation><terminology><definitions>"
    pros=PreprocessPostContent()
    print(pros.getPlainTxt(s))
    print(", ".join(s.replace("<","").replace(">"," ").strip().split(" ")))
    print(s)

    #s='     with help of(jssie), <code>jet.netty().take(123)</code>'
    s='''#1: <p>What is "backprop"?</p>  <p>What does "backprop" mean? I've Googled it, but it's showing backpropagation.</p>
    <p>Is the "backprop" term basically the same as "backpropagation" <code>bp.loss.backward()</code> or does it have a different meaning?</p> '''
    print(pros.getPlainTxt(s))
    print(pros.getParagraphs(s))
