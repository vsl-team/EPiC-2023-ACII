from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

tf.get_logger().setLevel('ERROR')

from keras.callbacks import TensorBoard, TerminateOnNaN
from official.modeling.optimization import CosineDecayWithOffset, LinearWarmup

from core import utils, config
from core.config import cfg
from core.dataset import epic_dataloader
from core.models import EPiCModel

import os.path as osp
import glob
import shutil


def copyfiles(source_dir, dest_dir, ext='*.py'):
    # Copy source files or compress to zip
    files = glob.iglob(osp.join(source_dir, ext))
    for file in files:
        if osp.isfile(file):
            shutil.copy2(file, dest_dir)

    if osp.isdir(osp.join(source_dir, 'core')) and not osp.isdir(osp.join(dest_dir, 'core')):
        shutil.copytree(osp.join(source_dir, 'core'), osp.join(dest_dir, 'core'), copy_function=shutil.copy2)


def get_dataloader(run_cfg):
    win_size = run_cfg.MODEL.SEQ_LEN / 1000
    dataloader = epic_dataloader(data_dir=run_cfg.DATA_LOADER.DATA_DIR, scenario=run_cfg.DATA_LOADER.SCENARIO,
                                 fold=run_cfg.DATA_LOADER.FOLD, val_ratio=run_cfg.DATA_LOADER.VAL_RATIO,
                                 win_size=win_size, batch_size=run_cfg.TRAIN.BATCH_SIZE)
    return dataloader


def get_mode(run_cfg):
    seq_len = run_cfg.MODEL.SEQ_LEN

    if run_cfg.MODEL.PRETRAINED != "" and Path(run_cfg.MODEL.PRETRAINED).is_file():
        use_random_feature = True
        print('Using random features')
    else:
        use_random_feature = False

    model = EPiCModel(seq_len, num_stages=run_cfg.MODEL.N_STAGES, dim_ff=run_cfg.MODEL.DIM_FF,
                      n_heads=run_cfg.MODEL.N_HEADS, n_layers=run_cfg.MODEL.N_LAYERS,
                      hid_multiplier=run_cfg.MODEL.HID_MULT, num_outputs=2, random_feature=use_random_feature,
                      variant=run_cfg.MODEL.VARIANT)

    base_lr = run_cfg.OPTIM.BASE_LR
    wd = run_cfg.OPTIM.WEIGHT_DECAY
    steps_per_epoch = cfg.TRAIN.STEP_PER_EPOCH
    n_epochs = cfg.TRAIN.EPOCHS
    assert 0 <= cfg.OPTIM.WARMUP_STEPS < 1
    n_warmup_steps = int(cfg.OPTIM.WARMUP_STEPS * steps_per_epoch * n_epochs)

    if cfg.OPTIM.LR_SCHEDULER_DECAY_STEPS < 0:
        decay_steps = steps_per_epoch * n_epochs - n_warmup_steps
    else:
        decay_steps = int(cfg.OPTIM.LR_SCHEDULER_DECAY_STEPS * steps_per_epoch * n_epochs)

    if cfg.OPTIM.LR_SCHEDULER == 'cosine':
        lr_scheduler = CosineDecayWithOffset(offset=n_warmup_steps, initial_learning_rate=base_lr,
                                             decay_steps=decay_steps, alpha=cfg.OPTIM.LR_SCHEDULER_ALPHA)
    else:
        lr_scheduler = base_lr
    if n_warmup_steps > 0:
        lr_opt = LinearWarmup(after_warmup_lr_sched=lr_scheduler, warmup_steps=n_warmup_steps,
                              warmup_learning_rate=0.)
    else:
        lr_opt = base_lr if lr_scheduler is None else lr_scheduler

    if run_cfg.OPTIM.NAME == 'adam':
        print('Use Adam optimizer')
        opt = tf.optimizers.Adam(learning_rate=lr_opt)
    elif run_cfg.OPTIM.NAME == 'adamw':
        print('Use AdamW optimizer')
        opt = tf.optimizers.AdamW(learning_rate=lr_opt, weight_decay=wd,) #  clipnorm=2.
        opt.exclude_from_weight_decay(var_names=['LayerNorm', 'layer_norm', 'bias'])
    else:
        print('Only support Adam, AdamW, and SGD. Use SGD by default')
        opt = tf.optimizers.SGD(learning_rate=lr_opt)

    loss_func = tf.losses.MeanSquaredError()
    metrics = [tf.metrics.MeanSquaredError(name='mse'), tf.metrics.RootMeanSquaredError(name='rmse')]

    model.compile(optimizer=opt, loss=loss_func, metrics=metrics, run_eagerly=run_cfg.DEBUG)

    if run_cfg.MODEL.PRETRAINED != "" and Path(run_cfg.MODEL.PRETRAINED).is_file():
        print('Loading pretrained weights ', run_cfg.MODEL.PRETRAINED)
        model.load_weights(run_cfg.MODEL.PRETRAINED, skip_mismatch=True, by_name=True)
        # Freeze all, except last dense layer
        print("trainable_weights:", len(model.trainable_weights))
        print('Freezing all, except last dense layer.')
        for idx in range(len(model.layers)):
            if model.layers[idx].name != 'regression_head':
                model.layers[idx].trainable = False

        print("after frozen, trainable_weights:", len(model.trainable_weights))

    return model


def run_experiments(run_cfg, num_gpus):
    n_epoch = cfg.TRAIN.EPOCHS
    steps_per_epoch = cfg.TRAIN.STEP_PER_EPOCH

    tsb_logdir = Path(cfg.OUT_DIR) / 'logs'
    tsb_logdir.mkdir(exist_ok=True)

    dataloader = get_dataloader(run_cfg)
    model = get_mode(run_cfg)

    print(f'Logging to {tsb_logdir.__str__()}')
    callbacks = [TensorBoard(log_dir=tsb_logdir), TerminateOnNaN()]

    model.fit(dataloader['train'], validation_data=dataloader['val'], epochs=n_epoch, callbacks=callbacks,
              steps_per_epoch=steps_per_epoch)

    # Saving last checkpoint to file
    print(f'Saving last checkpoint to {(Path(cfg.OUT_DIR) / "ckpt_last.h5").__str__()}')
    ckpt_path = Path(cfg.OUT_DIR) / 'ckpt_last.h5'
    model.save_weights(ckpt_path)

    # Do prediction
    test_arr, test_name, test_prediction = model.predict(dataloader['test'])

    prediction_group = pd.DataFrame({'name': test_name, 'time': test_arr[:, 0], 'valence': test_prediction[:, 0],
                                     'arousal': test_prediction[:, 1]}).groupby(by='name')

    write_folder = Path(cfg.OUT_DIR) / 'test/annotations'
    write_folder.mkdir(exist_ok=True, parents=True)
    print('Writing prediction to ', write_folder.__str__())
    for name, pred in prediction_group:
        pred.drop(columns=['name']).set_index('time').to_csv(write_folder / f"{name.decode('utf-8')}")

    print('Finished')


if __name__ == '__main__':
    config.load_cfg_fom_args("EPiC - ACII 2023")
    config.assert_and_infer_cfg()
    cfg.freeze()

    cfg_file = config.dump_cfg(cfg.OUT_DIR)

    # Copy source code
    out_dir_src = Path(cfg.OUT_DIR) / 'src'
    out_dir_src.mkdir(exist_ok=True, parents=True)
    copyfiles(source_dir='./', dest_dir=out_dir_src.__str__())

    # Set tensorflow GPU parameters
    utils.set_seed(seed=cfg.RNG_SEED)
    n_gpus = utils.set_gpu_growth()
    utils.set_mixed_precision(mixed_precision=cfg.TRAIN.MIXED_PRECISION)

    run_experiments(cfg, n_gpus)
