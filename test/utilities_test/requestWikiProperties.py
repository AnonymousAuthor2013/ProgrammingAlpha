import requests

def requestData():
    S=requests.Session()
    url="https://en.wikipedia.org/w/api.php"
    params={
        "action":"query",
        "format":"json",
        "pageids":[201489,238493],
        #"titles":"Gradient descent",
        "prop":"categories"
    }

    R=S.get(url=url,params=params)
    data=R.json()
    print(data)

requestData()

