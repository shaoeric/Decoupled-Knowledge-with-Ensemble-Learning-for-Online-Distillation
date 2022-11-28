import torch
from configs import ConfigLoader
from datetime import datetime
from src import DatasetBuilder, TransformBuilder, ModelBuilder, LossBuilder, LossWrapper, NetIO, Trainer
import argparse
import numpy as np
import os
from torch.utils.data import DataLoader
from copy import deepcopy

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_environment(args):
    config  = ConfigLoader.load(args.config_path.replace('\n', '').replace('\r', ''))
    date = datetime.now().strftime("%Y%m%d")
    if args.save_dir is not None:
        config.output["save_dir"] = args.save_dir
    config.output["save_dir"] = "{}_{}".format(date, config.output["save_dir"])

    config.model['name'] = args.model
    config.train['init'] = args.init
    config.train['exp'] = args.exp
    
    if args.seed is not None:
        seed = args.seed
        config.environment['seed'] = seed
    seed = config.environment['seed']
    set_seed(seed)


    if config.environment.cuda.flag:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
    return config

def build_dataloader(config):
    batch_size = config.train['batch_size']
    transform_name = config.dataset['transform_name']
    dataset_name = config.dataset['name']
    train_transform, val_transform = TransformBuilder.load(transform_name)
    if dataset_name in ['CIFAR10', 'CIFAR100']:
        trainset, trainset_config = DatasetBuilder.load(dataset_name=dataset_name, transform=train_transform, train=True)
        valset, valset_config = DatasetBuilder.load(dataset_name=dataset_name, transform=val_transform, train=False)
    else:
        trainset, trainset_config = DatasetBuilder.load(dataset_name=dataset_name, transform=train_transform, train=True, model_num=3)
        valset, valset_config = DatasetBuilder.load(dataset_name=dataset_name, transform=val_transform, train=False, class2label=trainset.class_to_idx, model_num=3)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return (train_loader, trainset_config), (val_loader, valset_config)


def build_trainer(config):
    netio = NetIO(config)

    model = ModelBuilder.load(config.model['name'], num_classes=config.model['num_classes'], ema=False)
    mean_model = deepcopy(model)
    mean_model.ema = True

    if config.model['resume']:
        model = netio.load_file(model, config.model['ckpt'])

    loss_funcs = []
    loss_weights = []
    for loss_name, weight in zip(config.train.criterion.names, config.train.criterion.loss_weights):
        loss_fun = LossBuilder.load(loss_name)
        loss_funcs.append(loss_fun)
        loss_weights.append(weight)
    
    loss_wrapper = LossWrapper(loss_funcs, loss_weights, exp=config.train['exp'])

    if config.environment.cuda.flag:
        model = model.cuda()
        mean_model = mean_model.cuda()
        loss_wrapper = loss_wrapper.cuda()
    
    trainer = Trainer(config=config, model=model, mean_model=mean_model, wrapper=loss_wrapper, ioer=netio)
    return trainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default="./configs/20220223_cifar100.yml")
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--init', type=int, default=1)
    parser.add_argument('--exp', type=float, default=0.5)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()

    config = prepare_environment(args)

    start_epoch = config.train['start_epoch']
    max_epoch = config.train['epochs']

    trainer = build_trainer(config)
    (train_loader, trainset_config), (val_loader, valset_config) = build_dataloader(config)
    trainer.init_mean_model(deepcopy(train_loader))

    if trainer.logger is not None:
        trainer.logger.info(config.environment)
        trainer.logger.info(trainset_config)
        trainer.logger.info(valset_config)
        trainer.logger.info(config.model)
        trainer.logger.info(config.train)
        trainer.logger.info(config.output)

    for epoch in range(start_epoch, max_epoch):
        trainer.train(epoch, train_loader)
        trainer.validate(epoch, val_loader)

    trainer.logger.info("best metric: {}".format(trainer.ioer.get_best_score()))

if __name__ == '__main__':
    main()

