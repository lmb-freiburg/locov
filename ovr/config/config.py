from detectron2.config import CfgNode as CN


def add_ovr_config(cfg):
    _C = cfg

    # object projection weights
    _C.MODEL.PROJECTION_WEIGHTS = ""
    # List of the prefix of the checkpoint parameter names so they can be correctly loaded
    _C.MODEL.BACKBONE_PREFIX = ("backbone.body.",)
    # Whether to load the vl_projection layer from the multimedia self-supervised learning head
    # If true, it loads it from the default mmss head defined by MODEL.MMSS_HEAD.DEFAULT_HEAD
    _C.MODEL.LOAD_EMB_PRED_FROM_MMSS_HEAD = False
    _C.MODEL.LOAD_OBJ_PROPOSALS = False

    # Add type of data if caption or detection
    _C.DATASETS.DATASET_CLASS = ""
    _C.DATASETS.NUM_TRAINIG_SAMPLES = 0

    # ---------------------------------------------------------------------------- #
    # Language Backbone options
    # ---------------------------------------------------------------------------- #
    _C.MODEL.LANGUAGE_BACKBONE = CN()
    _C.MODEL.LANGUAGE_BACKBONE.TYPE = "build_bert_backbone"
    _C.MODEL.LANGUAGE_BACKBONE.FREEZE = True
    _C.MODEL.LANGUAGE_BACKBONE.EMBEDDING_PATH = ""
    _C.MODEL.LANGUAGE_BACKBONE.ADD_POSITION_EMBEDDING = False
    _C.MODEL.LANGUAGE_BACKBONE.PRETRAINED = True

    # ---------------------------------------------------------------------------- #
    # MMSS Head options
    # ---------------------------------------------------------------------------- #
    # Config for Multimedia Self-Supervised training heads
    _C.MODEL.MMSS_HEAD = CN()

    _C.MODEL.MMSS_HEAD.TYPES = ("GroundingHead",)
    _C.MODEL.MMSS_HEAD.DEFAULT_HEAD = "GroundingHead"
    _C.MODEL.MMSS_HEAD.TIE_VL_PROJECTION_WEIGHTS = False
    _C.MODEL.MMSS_HEAD.IN_FEATURES = "res5"
    # How many visual regions to keep at most? -1 or 0 means keep all.
    _C.MODEL.MMSS_HEAD.SPATIAL_DROPOUT = -1
    # Activates distillation loss between the two branches
    _C.MODEL.MMSS_HEAD.DISTILLATION_LOSS = False
    _C.MODEL.MMSS_HEAD.DISTILLATION_LOSS_TYPE = "KD"
    _C.MODEL.MMSS_HEAD.DISTILLATION_TEMPERATURE = 1.0
    _C.MODEL.MMSS_HEAD.DISTILLATION_LOSS_WEIGHT = 1.0
    _C.MODEL.MMSS_HEAD.DISTILLATION_DETACH_TEACHER = False
    _C.MODEL.MMSS_HEAD.DISTILLATION_TEACHER_TRANSFORMER = True

    # Config for Grounding Head (default)
    _C.MODEL.MMSS_HEAD.GROUNDING = CN()
    _C.MODEL.MMSS_HEAD.GROUNDING.LOCAL_METRIC = "dot"
    _C.MODEL.MMSS_HEAD.GROUNDING.GLOBAL_METRIC = "aligned_local"
    _C.MODEL.MMSS_HEAD.GROUNDING.ALIGNMENT = "softmax"
    _C.MODEL.MMSS_HEAD.GROUNDING.ALIGNMENT_TEMPERATURE = 10.0
    _C.MODEL.MMSS_HEAD.GROUNDING.LOSS = "cross_entropy"
    _C.MODEL.MMSS_HEAD.GROUNDING.NEGATIVE_MINING = "random"
    _C.MODEL.MMSS_HEAD.GROUNDING.TRIPLET_MARGIN = 1.0
    _C.MODEL.MMSS_HEAD.GROUNDING.ALIGN_WORDS_TO_REGIONS = True
    _C.MODEL.MMSS_HEAD.GROUNDING.ALIGN_REGIONS_TO_WORDS = True
    _C.MODEL.MMSS_HEAD.GROUNDING.CONV_EMB = (1, 2, 3)
    # Which output of the language model to use: {"input_embeddings", "encoded_tokens"}
    _C.MODEL.MMSS_HEAD.GROUNDING.TEXT_INPUT = "input_embeddings"

    # Config for Transformer Head
    _C.MODEL.MMSS_HEAD.TRANSFORMER = CN()
    _C.MODEL.MMSS_HEAD.TRANSFORMER.MASKED_LANGUAGE_MODELING = False
    # Probability of selecting each token for masking
    _C.MODEL.MMSS_HEAD.TRANSFORMER.MASKED_LANGUAGE_MODELING_PROB = 0.15
    # Probability of replacing a selected token with mask
    _C.MODEL.MMSS_HEAD.TRANSFORMER.MASKED_LANGUAGE_MODELING_PROB_MASK = 0.9
    # Probability of replacing a selected token with another random token
    # Note the sum of this and the previous one should be less than or equal to one.
    # If the sum is less than one, the rest of the selected tokens will remain intact.
    _C.MODEL.MMSS_HEAD.TRANSFORMER.MASKED_LANGUAGE_MODELING_PROB_NOISE = 0.0
    # If true, it will do MLM even during validation, so we can compute MLM accuracy.
    # But this affects other tasks like image-caption matching, so we have the option to disable it.
    _C.MODEL.MMSS_HEAD.TRANSFORMER.MASKED_LANGUAGE_MODELING_VALIDATION = True
    _C.MODEL.MMSS_HEAD.TRANSFORMER.MASKED_VISUAL_MODELING = False
    # Can be either contrastive_cross_entropy or reconstruction_error
    _C.MODEL.MMSS_HEAD.TRANSFORMER.MVM_LOSS = ""
    # Used when MVM_LOSS is contrastive_cross_entropy
    _C.MODEL.MMSS_HEAD.TRANSFORMER.MVM_LOSS_NUM_NEGATIVE = 128
    # Can be either binary or cross_entropy
    _C.MODEL.MMSS_HEAD.TRANSFORMER.MMM_LOSS = ""
    # BERT configuration for language model
    _C.MODEL.MMSS_HEAD.TRANSFORMER.BERT_CONFIG = CN()
    _C.MODEL.MMSS_HEAD.TRANSFORMER.BERT_CONFIG.vocab_size = 30522
    _C.MODEL.MMSS_HEAD.TRANSFORMER.BERT_CONFIG.hidden_size = 768
    _C.MODEL.MMSS_HEAD.TRANSFORMER.BERT_CONFIG.num_hidden_layers = 12
    _C.MODEL.MMSS_HEAD.TRANSFORMER.BERT_CONFIG.num_attention_heads = 12
    _C.MODEL.MMSS_HEAD.TRANSFORMER.BERT_CONFIG.intermediate_size = 3072
    _C.MODEL.MMSS_HEAD.TRANSFORMER.BERT_CONFIG.hidden_act = "gelu"
    _C.MODEL.MMSS_HEAD.TRANSFORMER.BERT_CONFIG.hidden_dropout_prob = 0.1
    _C.MODEL.MMSS_HEAD.TRANSFORMER.BERT_CONFIG.attention_probs_dropout_prob = 0.1
    _C.MODEL.MMSS_HEAD.TRANSFORMER.BERT_CONFIG.max_position_embeddings = 512
    _C.MODEL.MMSS_HEAD.TRANSFORMER.BERT_CONFIG.type_vocab_size = 2
    _C.MODEL.MMSS_HEAD.TRANSFORMER.BERT_CONFIG.initializer_range = 0.02
    _C.MODEL.MMSS_HEAD.TRANSFORMER.BERT_CONFIG.layer_norm_eps = 1e-12
    _C.MODEL.MMSS_HEAD.TRANSFORMER.BERT_CONFIG.pad_token_id = 0
    _C.MODEL.MMSS_HEAD.TRANSFORMER.BERT_CONFIG.gradient_checkpointing = False
    _C.MODEL.MMSS_HEAD.TRANSFORMER.pretrained_weights = False

    # CLIP configuration for language model
    _C.MODEL.MMSS_HEAD.TRANSFORMER.CLIP_CONFIG = CN()
    _C.MODEL.MMSS_HEAD.TRANSFORMER.CLIP_CONFIG.TYPE = "RN50_text"
    _C.MODEL.MMSS_HEAD.TRANSFORMER.CLIP_CONFIG.EMBED_DIM = 1024
    _C.MODEL.MMSS_HEAD.TRANSFORMER.CLIP_CONFIG.CONTEXT_LENGHT = 77
    _C.MODEL.MMSS_HEAD.TRANSFORMER.CLIP_CONFIG.VOCAB_SIZE = 49408
    _C.MODEL.MMSS_HEAD.TRANSFORMER.CLIP_CONFIG.TRANSFORMER_WIDTH = 512
    _C.MODEL.MMSS_HEAD.TRANSFORMER.CLIP_CONFIG.TRANSFORMER_HEADS = 8
    _C.MODEL.MMSS_HEAD.TRANSFORMER.CLIP_CONFIG.TRANSFORMER_LAYERS = 12
    _C.MODEL.MMSS_HEAD.TRANSFORMER.CLIP_CONFIG.WEIGHTS_PRETRAINED = True

    # Word base embeddings config (e.g. glove, vico)
    _C.MODEL.MMSS_HEAD.TRANSFORMER.WORD_EMBEDDING_CONFIG = CN()
    _C.MODEL.MMSS_HEAD.TRANSFORMER.WORD_EMBEDDING_CONFIG.VOCAB_PATH = "datasets_data/word_vec_emb/vico/glove_300_vico_linear_200/visual_word_vecs_idx.json"
    _C.MODEL.MMSS_HEAD.TRANSFORMER.WORD_EMBEDDING_CONFIG.EMBEDDING_WORD_VECS_PATH = "datasets_data/word_vec_emb/vico/glove_300_vico_linear_200/visual_word_vecs.h5py"

    # Config for MLP Head
    _C.MODEL.MMSS_HEAD.MLP = CN()

    # If true, classification is done via dot product with loaded embeddings
    _C.MODEL.ROI_BOX_HEAD.EMBEDDING_BASED = False
    # Dimension of embeddings that will be loaded (300 for Glove, 768 for Bert)
    _C.MODEL.ROI_BOX_HEAD.EMB_DIM = 768
    # Whether or not to freeze the vl_projection layer. True is better. Only works if
    # MODEL.LOAD_EMB_PRED_FROM_MMSS_HEAD is true
    _C.MODEL.ROI_BOX_HEAD.FREEZE_EMB_PRED = False
    # Weather or not to normalize the embedding before the dot product
    _C.MODEL.ROI_BOX_HEAD.NORMALIZE_EMB_PRED = False
    # Weather or not to standardize the embedding before the dot product
    _C.MODEL.ROI_BOX_HEAD.STANDARDIZE_EMB_PRED = False

    # This excludes training the last class predictor layer
    _C.MODEL.ROI_HEADS.DETACH_CLASS_PREDICTOR = False

    # Logging options
    _C.SOLVER.LOG_PERIOD = 20
    _C.SOLVER.MAX_EPOCHS = 0

    _C.SOLVER.EPOCH_ITER_SIZE = 1000
    _C.SOLVER.CHECKPOINT_EPOCH = 1

    # To do evaluation not only calculate loss for first test dataset
    _C.TEST.DO_EVAL = True
    _C.TEST.IMS_PER_BATCH = 16
    _C.TEST.EVAL_INIT = False
    _C.TEST.SAVE_MODEL_BEST_METRIC = "val/bbox/AP50"  #

    # ADD NOISE TO SEE HOW ROBUST IS THE DETECTION MODEL
    _C.INPUT.NOISE_OFFLINE = False
    # Percentage of noisy bounding boxes added in every training image
    _C.INPUT.NOISE_BBOX = 0.0
    # Percentage of noisy classify examples in every training image
    _C.INPUT.NOISE_CLS = 0.0
    # Percentage of bounding boxes removed in every training image
    _C.INPUT.NOISE_RM_BBOX = 0.0
    # Percentage of bounding boxes shifted in every training image
    _C.INPUT.NOISE_LOC = 0.0
    # Percentage of bounding boxes whose class label is set to -1 to be ignored
    _C.INPUT.NOISE_IGN = 0.0

    # Mode for flipping images used in data augmentation during training
    # choose one of ["horizontal, "vertical", "none"]
    _C.INPUT.RANDOM_FLIP = "horizontal"
    # Mode for color jittering, SimCLR uses 0.4
    _C.INPUT.COLOR_JITTER = 0.0
    # Mode for random gray scale
    _C.INPUT.RANDOM_GRAY_SCALE = False
    # Mode for gaussian blur
    _C.INPUT.GAUSSIAN_BLUR = False
    # Mode for random erase
    _C.INPUT.RANDOM_ERASE = False
