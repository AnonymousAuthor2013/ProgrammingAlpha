import regex as re

class PreprocessPostContent(object):
    code_snippet=re.compile(r"<pre><code>.*?</code></pre>")
    code_insider=re.compile(r"<code>.*?</code>")
    plain_text=re.compile(r"<.*?>")

    @staticmethod
    def filterNILStr(s):
        filterFunc=lambda s: s and s.strip()
        s=' '.join(list(filter(filterFunc,s.split())))
        s=list(filter(filterFunc,s.split('\n')))
        return s

    def getPlainTxt(self,raw_txt,keep_emb_code=False):
        if keep_emb_code:
            txt=re.sub(self.code_snippet," ",raw_txt)
        else:
            txt=re.sub(self.code_insider," ",raw_txt)

        txt=re.sub(self.plain_text," ",txt)
        return txt

    def getEmCodes(self,raw_txt):
        emcodes=self.code_insider.findall(raw_txt)

        return emcodes

    def getCodeSnippets(self,raw_txt):
        snippets=self.code_snippet.findall(raw_txt)

        return snippets

if __name__ == '__main__':
    s="<neural-networks><backpropagation><terminology><definitions>"
    pros=PreprocessPostContent()
    print(pros.getPlainTxt(s))
    print(", ".join(s.replace("<","").replace(">"," ").strip().split(" ")))
    print(s)
