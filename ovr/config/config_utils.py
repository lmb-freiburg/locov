from detectron2.config.config import CfgNode
from detectron2.utils.file_io import PathManager


def edit_output_dir_exp_specific(cfg: CfgNode):
    base_dir = cfg.OUTPUT_DIR
    if PathManager.isdir(base_dir):
        print("continue from existing folder")
        return cfg

    # Visual model params
    visual_text = "V-" + cfg.MODEL.BACKBONE.NAME.replace("build_", "").replace(
        "_backbone", ""
    )
    if "resnet" in visual_text:
        visual_text.replace("resnet", "resnet" + str(cfg.MODEL.RESNETS.DEPTH))
    visual_text += "_frz" + str(cfg.MODEL.BACKBONE.FREEZE_AT)

    lang_text = ""

    if "MMSS" in cfg.MODEL.META_ARCHITECTURE:
        # In features
        visual_text += "_infeat-" + cfg.MODEL.MMSS_HEAD.IN_FEATURES
        # distill
        if cfg.MODEL.MMSS_HEAD.DISTILLATION_LOSS:
            visual_text += (
                "_distill"
                + str(cfg.MODEL.MMSS_HEAD.DISTILLATION_TEMPERATURE)
                + "w"
                + str(cfg.MODEL.MMSS_HEAD.DISTILLATION_LOSS_WEIGHT)
                + (
                    "_detachteacher"
                    if cfg.MODEL.MMSS_HEAD.DISTILLATION_DETACH_TEACHER
                    else ""
                )
                + (
                    "_teachergrounding"
                    if not cfg.MODEL.MMSS_HEAD.DISTILLATION_TEACHER_TRANSFORMER
                    else ""
                )
            )
        # embd normalization
        if (
            cfg.MODEL.ROI_BOX_HEAD.EMBEDDING_BASED
            and cfg.MODEL.ROI_BOX_HEAD.NORMALIZE_EMB_PRED
        ):
            visual_text += "_normembd"
        # embd standardization
        if (
            cfg.MODEL.ROI_BOX_HEAD.EMBEDDING_BASED
            and cfg.MODEL.ROI_BOX_HEAD.STANDARDIZE_EMB_PRED
        ):
            visual_text += "_standembd"

        # Language model params
        lang_text = "L-" + cfg.MODEL.LANGUAGE_BACKBONE.TYPE.replace(
            "build_", ""
        ).replace("_backbone", "")
        lang_text += "_frz" if cfg.MODEL.LANGUAGE_BACKBONE.FREEZE else ""
    else:
        # RoI head
        if cfg.MODEL.ROI_BOX_HEAD.NAME != "":
            visual_text += (
                "_"
                + cfg.MODEL.ROI_BOX_HEAD.NAME
                + ("-emb" if cfg.MODEL.ROI_BOX_HEAD.EMBEDDING_BASED else "")
            )
            visual_text += (
                "-cls_agnostic" if cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG else ""
            )

        # embd normalization
        if (
            cfg.MODEL.ROI_BOX_HEAD.EMBEDDING_BASED
            and cfg.MODEL.ROI_BOX_HEAD.NORMALIZE_EMB_PRED
        ):
            visual_text += "_normembd"
        # embd standardization
        if (
            cfg.MODEL.ROI_BOX_HEAD.EMBEDDING_BASED
            and cfg.MODEL.ROI_BOX_HEAD.STANDARDIZE_EMB_PRED
        ):
            visual_text += "_standembd"
        # Matching head
        # if cfg.MODEL.MATCH_HEAD.NAME != '':
        #     visual_text += '-emb:'+str(cfg.MODEL.MATCH_HEAD.EMB_DIM) +\
        #                    ('_'+cfg.MODEL.MATCH_HEAD.BACKBONE if cfg.MODEL.MATCH_HEAD.BACKBONE=='linear' else \
        #                    '_'+cfg.MODEL.MATCH_HEAD.BACKBONE+'_h'+str(cfg.MODEL.MATCH_HEAD.NUM_HEADS)) +\
        #                    '_local:'+cfg.MODEL.MATCH_HEAD.LOCAL_FEAT+('_norm' if cfg.MODEL.MATCH_HEAD.NORMALIZE else '') +\
        #                    '_sim:'+cfg.MODEL.MATCH_HEAD.SIMILARITY_METRIC +\
        #                    '_align:'+cfg.MODEL.MATCH_HEAD.ALIGNMENT +\
        #                    '_loss:'+cfg.MODEL.MATCH_HEAD.LOSS

        # Language model params
        # lang_text = 'L:'+cfg.LANGUAGE_MODEL.PRETRAINED +\
        # lang_text = ('L:'+cfg.LANGUAGE_MODEL.PRETRAINED +\
        #             ('-frz:'+str(cfg.LANGUAGE_MODEL.FREEZE_AT) if cfg.LANGUAGE_MODEL.FREEZE else '')+\
        #             ('-l'+str(cfg.LANGUAGE_MODEL.HIDDEN_LAYERS) if cfg.LANGUAGE_MODEL.HIDDEN_LAYERS>0 else '') \
        #             if cfg.MODEL.META_ARCHITECTURE not in {'VisGeneralizedRCNN', 'GeneralizedRCNN'} else '')

    # Optimization params
    opt_text = "S-" + "bs" + str(cfg.SOLVER.IMS_PER_BATCH)
    opt_text += "_lr" + str(cfg.SOLVER.BASE_LR)
    opt_text += "_sch-" + cfg.SOLVER.LR_SCHEDULER_NAME.lower()

    # Data augmentations
    # data_text = 'D:'
    # data_text += 'match' if cfg.INPUT.MATCHING_ONLY else ''
    # data_text += '-noise:' + ('offline' if cfg.INPUT.NOISE_OFFLINE else '') +\
    #             ('_box'+str(cfg.INPUT.NOISE_BBOX) if cfg.INPUT.NOISE_BBOX>0 else '') +\
    #             ('_cls'+str(cfg.INPUT.NOISE_CLS) if cfg.INPUT.NOISE_CLS>0 else '') +\
    #             ('_rm'+str(cfg.INPUT.NOISE_RM_BBOX) if cfg.INPUT.NOISE_RM_BBOX>0 else '') +\
    #             ('_loc'+str(cfg.INPUT.NOISE_LOC) if cfg.INPUT.NOISE_LOC>0 else '') +\
    #             ('_ign'+str(cfg.INPUT.NOISE_IGN) if cfg.INPUT.NOISE_IGN>0 else '')
    # data_text += '-aug:' + ('_cjit'+str(cfg.INPUT.COLOR_JITTER) if cfg.INPUT.COLOR_JITTER>0 else '') +\
    #             ('_grey' if cfg.INPUT.RANDOM_GRAY_SCALE else '') + \
    #             ('_blur' if cfg.INPUT.GAUSSIAN_BLUR else '') + ('_erase' if cfg.INPUT.RANDOM_ERASE else '')
    # if len(data_text)>2:
    #     opt_text += '-'+data_text

    # reduce training samples
    # if cfg.DATASETS.PERCENT_TRAINING_SAMPLES < 1.0:
    #     cfg.DATASETS.NUM_TRAINIG_SAMPLES = int(cfg.DATASETS.PERCENT_TRAINING_SAMPLES*cfg.DATASETS.NUM_TRAINIG_SAMPLES)
    #     base_dir += '_train'+str(cfg.DATASETS.PERCENT_TRAINING_SAMPLES)

    base_dir += "-" + cfg.MODEL.META_ARCHITECTURE
    base_dir += "-" + visual_text if len(visual_text) > 0 else ""
    base_dir += "-" + lang_text if len(lang_text) > 0 else ""
    base_dir += "-" + opt_text
    cfg.OUTPUT_DIR = base_dir

    # fix number of iterations if necessary to adjust to bs
    if cfg.SOLVER.MAX_EPOCHS != 0 and cfg.DATASETS.NUM_TRAINIG_SAMPLES != 0:
        epoch_iterations = cfg.DATASETS.NUM_TRAINIG_SAMPLES // cfg.SOLVER.IMS_PER_BATCH
        cfg.SOLVER.EPOCH_ITER_SIZE = epoch_iterations
        cfg.SOLVER.MAX_ITER = int(epoch_iterations * cfg.SOLVER.MAX_EPOCHS)
        if cfg.SOLVER.CHECKPOINT_PERIOD > 0:
            cfg.SOLVER.CHECKPOINT_PERIOD = (
                int(epoch_iterations) * cfg.SOLVER.CHECKPOINT_EPOCH
            )
        if cfg.SOLVER.STEPS_EPOCHS[0] != 0:
            list_steps = []
            for sepoch in cfg.SOLVER.STEPS_EPOCHS:
                list_steps.append(int(epoch_iterations * sepoch))
            cfg.SOLVER.STEPS = tuple(list_steps)
        if cfg.TEST.EVAL_EPOCH != 0:
            cfg.TEST.EVAL_PERIOD = int(epoch_iterations * cfg.TEST.EVAL_EPOCH)

    if cfg.SOLVER.CHECKPOINT_PERIOD == 0:
        cfg.SOLVER.CHECKPOINT_PERIOD = cfg.SOLVER.MAX_ITER + 10
    return cfg
