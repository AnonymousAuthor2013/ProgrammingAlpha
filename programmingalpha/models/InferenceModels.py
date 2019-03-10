from pytorch_pretrained_bert import BertModel
from pytorch_pretrained_bert.modeling import BertPreTrainedModel
from torch.nn import CrossEntropyLoss
from torch import nn

class BertForLinkRelationPrediction(BertPreTrainedModel):

    def __init__(self, config, num_labels):
        super(BertForLinkRelationPrediction, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct_classification = CrossEntropyLoss()
            classfication_loss = loss_fct_classification(logits.view(-1, self.num_labels), labels.view(-1))
            return classfication_loss
        else:
            return logits
