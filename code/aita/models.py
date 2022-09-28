from torch import nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertForSequenceClassification, BertModel
from transformers.modeling_outputs import SequenceClassifierOutput


class LSTMW2V(nn.Module):
    """
    LSTM model code. Obtains text embedings and the applies a linear layer on top of them,
    in order to later on classify the post in being yta or nta.
    """
    def __init__(self, dimension, vocab_num, embedding_length) -> None:
        super(LSTMW2V, self).__init__()

        self.embedding_layer = nn.Embedding(vocab_num, embedding_dim=embedding_length)
        self.dimension = dimension
        self.lstm = nn.LSTM(input_size=300,
                            hidden_size=self.dimension,
                            num_layers=2,
                            batch_first=True)
        self.fc = nn.Linear(self.dimension, 1)

    def forward(self, text, text_len):
        text_embed = self.embedding_layer(text)
        # pad the text if it is shorter or longer
        packed_input = pack_padded_sequence(text_embed, text_len, batch_first=True, enforce_sorted=False)
        packed_output, (hidden_state, cell_state) = self.lstm(packed_input)
        features = self.fc(hidden_state[-1, :, :])
        binary_features = torch.nn.functional.sigmoid(features)
        return binary_features


class BERTJoint():
    """
    Joint BERT model uses two BERT models, one fined-tuned for post classification,
    and one fined-tuned for comments classification. After obtaining hidden feature space
    for both comments and posts, we perform an average function over the comments'
    representation, and then concatenate the average comments' representation with the
    comments hidden representation, to later on feed into a benchmark classifier.
    """
    def __init__(self, posts_weights, comments_weights, device="cpu"):
        self.post_bert = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels=2,
            output_attentions=False,
            output_hidden_states=True)

        self.post_bert.load_state_dict(torch.load(posts_weights))
        self.post_bert.to(device)

        self.comments_bert = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels=2,
            output_attentions=False,
            output_hidden_states=True)

        self.comments_bert.load_state_dict(torch.load(comments_weights))
        self.comments_bert.to(device)

    def __call__(self, post_id, post_attention, comments_ids, comments_attentions):
        with torch.no_grad():
            vals_post = self.post_bert(post_id, post_attention)
            final_layer = vals_post[1][-1]
            cls_token_post = final_layer[:, 0, :]
            cls_token_post = cls_token_post.squeeze()

            comments_ids = torch.squeeze(comments_ids)
            comments_attentions = torch.squeeze(comments_attentions)
            vals_comments = self.comments_bert(comments_ids, comments_attentions)
            final_layer = vals_comments[1][-1]
            cls_token_comments = final_layer[:, 0, :]
            averaged_comments = torch.mean(cls_token_comments, axis=0)

            concat_embeddings = torch.cat([averaged_comments, cls_token_post], dim=0)
            concat_embeddings = concat_embeddings.cpu().numpy()

            return concat_embeddings


class WeightedBCEBert(nn.Module):
    """
    The weighted BERT copies exactly the architecture of the usual classification BERT,
    with the difference that it's loss function is Weighted Binary Cross Entropy, where the
    weights represent the ration of the label classes.
    """
    def __init__(self):
        super(WeightedBCEBert, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768, 2)

    def forward(self, ids, mask, token_type_ids, labels=None, pos_weight=torch.ones([2])):
        outputs = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)

        output = self.dropout(outputs.pooler_output)
        logits = self.classifier(output)
        loss = None

        # you can define any loss function here yourself
        # see https://pytorch.org/docs/stable/nn.html#loss-functions for an overview
        loss_fct = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        # next, compute the loss based on logits + ground-truth labels
        loss = loss_fct(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            # attentions=outputs.attentions,
        )