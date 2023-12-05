import os
import argparse
import logging
import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from models.unimo_model import UnimoREModel
from processor.dataset import MOREProcessor, MOREDataset
from modules.train import BertTrainer
import warnings

# from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter

warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO,
                    )
logger = logging.getLogger(__name__)

DATA_PATH = {
    'MORE': {
        'train': 'data/MORE/txt/train.txt',
        'valid': 'data/MORE/txt/valid.txt',
        'test': 'data/MORE/txt/test.txt',
        'train_ent_dict': 'data/MORE/ent_train_dict.pth',
        'valid_ent_dict': 'data/MORE/ent_valid_dict.pth',
        'test_ent_dict': 'data/MORE/ent_test_dict.pth',
    }
}

IMG_PATH = {
    'MORE': {
        'train': 'data/MORE/img_org/train',
        'valid': 'data/MORE/img_org/valid',
        'test': 'data/MORE/img_org/test',
    }
}

DEP_PATH = {
    'MORE': {
        'train': 'data/MORE/img_dep/',
        'valid': 'data/MORE/img_dep/',
        'test': 'data/MORE/img_dep/',
    }
}

CAP_PATH = {
    'MORE': {
        'train': 'data/MORE/caption_dict.json',
        'valid': 'data/MORE/caption_dict.json',
        'test': 'data/MORE/caption_dict.json',
    }
}

re_path = 'data/MORE/rel2id.json'


def set_seed(seed=2021):
    """set random seed"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vit_name', default='vit', type=str, help="The name of vit.")
    parser.add_argument('--dataset_name', default='MORE', type=str, help="The name of dataset.")
    parser.add_argument('--bert_name', default='bert-base', type=str,
                        help="Pretrained language model name, bart-base or bart-large")
    parser.add_argument('--num_epochs', default=30, type=int, help="Training epochs")
    parser.add_argument('--device', default='cuda', type=str, help="cuda or cpu")
    parser.add_argument('--batch_size', default=16, type=int, help="batch size")
    parser.add_argument('--lr', default=2e-5, type=float, help="learning rate")
    parser.add_argument('--warmup_ratio', default=0.01, type=float)
    parser.add_argument('--eval_begin_epoch', default=16, type=int)
    parser.add_argument('--seed', default=1, type=int, help="random seed, default is 1")
    parser.add_argument('--load_path', default=None, type=str, help="Load model from load_path")
    parser.add_argument('--save_path', default=None, type=str, help="save model at save_path")
    parser.add_argument('--notes', default="", type=str, help="input some remarks for making save path dir.")
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--do_predict', action='store_true')
    parser.add_argument('--max_seq', default=128, type=int)
    parser.add_argument('--use_box', action='store_true')
    parser.add_argument('--use_cap', action='store_true')
    parser.add_argument('--use_dep', action='store_true')

    args = parser.parse_args()
    set_seed(args.seed)  # set seed, default is 1

    data_path, img_path, dep_path, cap_path = DATA_PATH[args.dataset_name], IMG_PATH[args.dataset_name], DEP_PATH[args.dataset_name], CAP_PATH[args.dataset_name]
    data_process, dataset_class = MOREProcessor, MOREDataset
    logger.info(data_path)

    if args.save_path is not None:  # make save_path dir
        args.save_path = os.path.join(args.save_path, args.dataset_name + "_" + str(args.batch_size) + "_" + str(args.lr) + "_" + args.notes)
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path, exist_ok=True)
    logger.info(args)

    # logdir = "logs/" + args.dataset_name + "_" + str(args.batch_size) + "_" + str(args.lr) + args.notes
    # writer = SummaryWriter(logdir=logdir),
    writer = None
    if args.do_train:
        processor = data_process(data_path, re_path, args.bert_name, args.vit_name)

        train_dataset = dataset_class(processor, img_path, dep_path, cap_path, args, mode='train')
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

        valid_dataset = dataset_class(processor, img_path, dep_path, cap_path, args, mode='valid')
        valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

        test_dataset = dataset_class(processor, img_path, dep_path, cap_path, args, mode='test')
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

        re_dict = processor.get_relation_dict()
        num_labels = len(re_dict)
        tokenizer = processor.tokenizer

        # train
        model = UnimoREModel(num_labels, tokenizer, args)
        model = torch.nn.DataParallel(model)
        model = model.to(args.device)
        trainer = BertTrainer(train_data=train_dataloader, dev_data=valid_dataloader, test_data=test_dataloader,
                              re_dict=re_dict, model=model, args=args, logger=logger, writer=writer)
        trainer.train()
        torch.cuda.empty_cache()
        # writer.close()


if __name__ == "__main__":
    main()
