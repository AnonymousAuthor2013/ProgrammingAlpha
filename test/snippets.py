import numpy as np

x=np.array([
    [1,2],
    [3,4],
    [5,6],
])


y=np.array([
    [0,1,2],
    [2,3,4]
])

print(x,'\n',y)

print('-*'*50)

z=y.dot(x)

print(z)

xm=np.sum(np.square(x),0)
ym=np.sum(np.square(y),1)

print(np.sqrt(xm))
print(np.sqrt(ym))

print(xm)
print(ym)

xm=np.linalg.norm(x,axis=0)
ym=np.linalg.norm(y,axis=1)

print(xm)
print(ym)

print(np.divide(x,xm))
print((y.T/ym).T)
print(
    ((y.T/ym).T).dot(np.divide(x,xm))
)

from programmingalpha.DataSet.PostPreprocessing import PreprocessPostContent
extractor=PreprocessPostContent()
txt=extractor.getPlainTxt('<p> is is it good?</p><code></code>')
print(txt)

'How to compute machine learning training computation time and what are reference values? Ask Question in many forums and documents on the internet we hear about "short" and "long" learning and prediction computation time for machine learning algorithms. For example the Decision Tree Algorithm has a short computation time as compared to Neural Networks. But what it is never mentioned is what is "short" and what is "long".Could you please clarify which unit you would use to measure computation time? Maybe \"seconds per sample"? And what are reference values, so that I can predict if it takes 1h, 1day or 1Week?'
