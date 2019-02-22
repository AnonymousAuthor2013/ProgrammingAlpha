import json

#bert model
BertBasePath="/home/zhangzy/BertModels/uncased_L-12_H-768_A-12/"
BertLargePath="/home/zhangzy/BertModels/uncased_L-24_H-1024_A-16/"
#openai GPT model
openAIGPTPath="/home/zhangzy/BertModels/openAIGPT/"
#transformer-XL
transformerXL="/home/zhangzy/BertModels/transformerXL/"
#gpt-2 model
GPT2Path="/home/zhangzy/BertModels/GPT2/"

#global project path
ConfigPath="/home/zhangzy/ProgrammingAlpha/ConfigData/"
DataPath="/home/zhangzy/ProgrammingAlpha/data/"
ModelPath="/home/zhangzy/ProgrammingAlpha/modelData/"



def loadConfig(filename):
    with open(filename,"r") as f:
        config=json.load(f)
    return config

def saveConfig(filename,config):
    with open(filename, "w") as f:
        json.dump(config,f)


