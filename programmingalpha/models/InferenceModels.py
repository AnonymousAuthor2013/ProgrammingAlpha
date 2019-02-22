from pytorch_pretrained_bert import BertModel
from pytorch_pretrained_bert.modeling import BertPreTrainedModel
from torch.nn import CrossEntropyLoss,MSELoss
from torch import nn

class BertForSemanticPrediction(BertPreTrainedModel):

    def __init__(self, config, num_labels):
        super(BertForSemanticPrediction, self).__init__(config)
        self.regression_act_fn=nn.Sigmoid()
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.regressor=nn.Linear(config.hidden_size,1)

        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,values=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        simValues=self.regression_act_fn(self.regressor(pooled_output))

        if labels is not None:
            loss_fct_classification = CrossEntropyLoss()
            loss_fct_regression=MSELoss()
            classfication_loss = loss_fct_classification(logits.view(-1, self.num_labels), labels.view(-1))
            regression_loss=loss_fct_regression(simValues.view(-1,1),values.view(-1))
            return classfication_loss,regression_loss
        else:
            return logits,simValues
