import pdb
import copy
import torch
from torch import nn
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import BertEncoder, BertPooler, BertLMPredictionHead, BertPredictionHeadTransform
from transformers import BertConfig
from ovr.modeling.logged_module import LoggedModule
from ovr.modeling.mmss_heads.grounding_head import MMSS_HEADS_REGISTRY
from ovr.modeling.mmss_heads.transformer_head import VisualEmbedding, MMPreTrainingHeads
BertLayerNorm = nn.LayerNorm

__all__ = ["MLPHead"]

@MMSS_HEADS_REGISTRY.register()
class MLPHead(LoggedModule):
    def __init__(self, config, v_dim, l_dim, loc_dim, backbone, *args, **kwargs):
        super(MLPHead, self).__init__()
        self.config = config.MODEL.MMSS_HEAD.TRANSFORMER
        self.v_dim = v_dim
        self.l_dim = l_dim
        self.loc_dim = loc_dim
        self.backbone = backbone

        self.bert_config = BertConfig(**self.config.BERT_CONFIG)
        self.v2l_projection = nn.Linear(self.v_dim, self.l_dim)
        self.visual_emb = VisualEmbedding(self.bert_config, self.l_dim, self.loc_dim)
        self.encoder = BertEncoder(self.bert_config)
        self.pooler = BertPooler(self.bert_config)
        self.heads = MMPreTrainingHeads(self.bert_config, self.v_dim)

        self.encoder.apply(self._init_weights)
        self.pooler.apply(self._init_weights)
        self.heads.apply(self._init_weights)

        self._tie_weights()

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)

        # weather to return the pair scores for distillation
        self.return_dist = config.MODEL.MMSS_HEAD.DISTILLATION_LOSS
        import ipdb; ipdb.set_trace()

    def _tie_weights(self):
        assert(self.heads.predictions.decoder.weight.shape[0] == 
               self.backbone.embeddings.shape[0])
        assert(self.heads.predictions.decoder.weight.shape[1] == 
               self.backbone.embeddings.shape[1])
        self.heads.predictions.decoder.weight = self.backbone.embeddings

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.bert_config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

        if self.config.pretrained_weights and isinstance(module, BertEncoder):
            pretrained_weights = copy.deepcopy(self.backbone.state_dict())
            actual_weights = module.state_dict()
            for name, val in actual_weights.items():
                if 'bert_model.encoder.'+name in pretrained_weights.keys():
                    if pretrained_weights['bert_model.encoder.'+name].shape == val.shape:
                        actual_weights[name] = pretrained_weights['bert_model.encoder.'+name].to(val.device)
            module.load_state_dict(actual_weights)
            del pretrained_weights, actual_weights

    def forward(self, input_image, input_caption):
        '''
        Mapping between my terminology and the vilbert codebase:
        input_ids => Not needed. Instead I have caption_emb which is the output of BERT
        image_feat => region_features
        image_loc => region_loc
        token_type_ids => Not needed. All zero anyway
        attention_mask => caption_mask,
        image_attention_mask => region mask,
        masked_lm_labels => combination of target_caption_ids, mlm_mask,
        image_label => mvm_mask,
        image_target => target_region_features,
        next_sentence_label => Not needed for now (image_caption_match_label).
        '''

        caption_emb = input_caption['encoded_tokens']
        caption_mask = input_caption['attention_mask']
        mlm_mask = input_caption['mlm_mask']
        target_caption_ids = input_caption['target_ids']

        region_features = input_image['region_features']
        region_mask = input_image['region_mask']
        region_loc = input_image['region_loc']
        mvm_mask = input_image['mvm_mask']
        target_region_features = input_image['target_region_features']

        target_caption_ids = torch.where(
            mlm_mask > 0,
            target_caption_ids,
            torch.ones_like(target_caption_ids) * (-1)
        )
        caption_mask = caption_mask.to(torch.float32)
        region_mask = region_mask.to(torch.float32)
        mlm_mask = mlm_mask.to(torch.float32)
        mvm_mask = mvm_mask.to(torch.float32)
        num_words = caption_mask.sum(dim=1)
        _, max_num_words = caption_mask.shape
        batch_size, max_num_regions, _ = region_features.shape

        image_emb = self.v2l_projection(region_features)
        image_emb = self.visual_emb(image_emb, region_loc)

        if self.mmm_loss == 'cross_entropy':
            image_emb = image_emb[None, :, :, :].repeat(batch_size, 1, 1, 1).reshape(
                batch_size**2, max_num_regions, self.l_dim)
            caption_emb = caption_emb[:, None, :, :].repeat(1, batch_size, 1, 1).reshape(
                batch_size**2, max_num_words, self.l_dim)
            region_mask = region_mask[None, :, :].repeat(batch_size, 1, 1).reshape(
                batch_size**2, max_num_regions)
            caption_mask = caption_mask[:, None, :].repeat(1, batch_size, 1).reshape(
                batch_size**2, max_num_words)

        embedded_tokens = torch.cat([caption_emb, image_emb], dim=1)
        attention_mask = torch.cat([caption_mask, region_mask], dim=1)

        sequence_output = self.encoder(
            embedded_tokens,
            attention_mask[:, None, None, :],
            head_mask=[None] * self.bert_config.num_hidden_layers,
            output_attentions=False,
            output_hidden_states=False,
        )
        sequence_output = sequence_output[0]
        pooled_output = self.pooler(sequence_output)
        sequence_output_t, sequence_output_v = torch.split(
            sequence_output,
            [max_num_words, max_num_regions],
            dim=1
        )
        prediction_scores_t, prediction_scores_v, seq_relationship_score = self.heads(
            sequence_output_t, sequence_output_v, pooled_output
        )
        # prediction_scores_v = prediction_scores_v[:, 1:]

        if self.mmm_loss == 'cross_entropy':
            prediction_scores_t = torch.diagonal(prediction_scores_t.reshape(
                batch_size, batch_size, max_num_words, self.bert_config.vocab_size),
                dim1=0, dim2=1).permute(2, 0, 1)
            prediction_scores_v = torch.diagonal(prediction_scores_v.reshape(
                batch_size, batch_size, max_num_regions, self.v_dim),
                dim1=0, dim2=1).permute(2, 0, 1)

        masked_lm_loss = self.loss_fct(
            prediction_scores_t.reshape(-1, self.bert_config.vocab_size),
            target_caption_ids.reshape(-1),
        )

        if self.mmm_loss == 'binary':
            raise NotImplementedError
        elif self.mmm_loss == 'cross_entropy':
            global_dist = seq_relationship_score[:, 0]
            pw_cost = global_dist.reshape(batch_size, batch_size)
            pw_logits_c_cap = torch.log_softmax(- pw_cost, dim=0)
            pw_logits_c_img = torch.log_softmax(- pw_cost, dim=1)
            next_sentence_loss_c_cap = torch.diag(- pw_logits_c_cap).mean()
            next_sentence_loss_c_img = torch.diag(- pw_logits_c_img).mean()
            next_sentence_loss = next_sentence_loss_c_cap + next_sentence_loss_c_img
        elif self.mmm_loss == '':
            next_sentence_loss = torch.tensor(0.0).cuda()
        else:
            raise NotImplementedError

        if self.mvm_loss == '':
            masked_img_loss = torch.tensor(0.0).cuda()
        else:
            raise NotImplementedError

        losses = {
            'Masked Language Modeling Loss': masked_lm_loss,
            'Masked Visual Modeling Loss': masked_img_loss,
            'Image Caption Matching Loss': next_sentence_loss,
        }
        acc_num = (prediction_scores_t.argmax(dim=-1) == target_caption_ids).to(torch.float32).sum()
        acc_denom = (target_caption_ids >= 0).to(torch.float32).sum()
        acc = torch.where(acc_denom > 0, acc_num / acc_denom, acc_denom)
        other_info = {
            'Masked Language Modeling Accuracy': acc,
        }
        if self.mmm_loss == 'cross_entropy':
            other_info['Batch Accuracy (Choose Caption)'] = torch.mean(
                (pw_cost.argmin(dim=0) ==torch.arange(batch_size).to('cuda')).to(torch.float32))
            other_info['Batch Accuracy (Choose Image)'] = torch.mean(
                (pw_cost.argmin(dim=1) == torch.arange(batch_size).to('cuda')).to(torch.float32))

        self.log_dict(losses)
        self.log_dict(other_info)

        if self.return_dist:
            distributions = {'trans':pw_cost}
            return other_info, losses, distributions

        return other_info, losses

def build_mlp_head(name, cfg, v_dim, l_dim, loc_dim, backbone, *args, **kwargs):
    return MMSS_HEADS_REGISTRY.get(name)(cfg, v_dim, l_dim, loc_dim, backbone)