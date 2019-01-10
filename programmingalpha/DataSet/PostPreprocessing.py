import regex as re
import json

class PreprocessPostContent(object):
    code_snippet=re.compile(r"<pre><code>.*?</code></pre>")
    code_insider=re.compile(r"<code>.*?</code>")
    plain_text=re.compile(r"<.*?>")

    def __init__(self):
        self.raw_txt=""

    def getPlainTxt(self):
        txt=re.sub(self.code_insider," ",self.raw_txt)
        txt=re.sub(self.plain_text," ",txt)
        return txt

    def getEmCodes(self):
        emcodes=self.code_insider.findall(self.raw_txt)

        return emcodes

    def getCodeSnippets(self):
        snippets=self.code_snippet.findall(self.raw_txt)

        return snippets

