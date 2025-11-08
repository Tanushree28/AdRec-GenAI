import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SRC = PROJECT_ROOT / "train"

if str(TRAIN_SRC) not in sys.path:
    sys.path.insert(0, str(TRAIN_SRC))

from dataset import MyDataset  # noqa: E402

from .model import KuaiRecBaselineModel


def get_args():
    parser = argparse.ArgumentParser()

    # Train params
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--maxlen', default=101, type=int)

    # Baseline Model construction
    parser.add_argument('--hidden_units', default=32, type=int)
    parser.add_argument('--num_blocks', default=1, type=int)
    parser.add_argument('--num_epochs', default=3, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--l2_emb', default=0.0, type=float)
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--inference_only', action='store_true')
    parser.add_argument('--state_dict_path', default=None, type=str)
    parser.add_argument('--norm_first', action='store_true')

    # MMemb Feature ID
    parser.add_argument('--mm_emb_id', nargs='+', default=['82'], type=str, choices=[str(s) for s in range(81, 87)])

    args = parser.parse_args()

    return args



if __name__ == '__main__':
    Path(os.environ.get('TRAIN_LOG_PATH', './logs')).mkdir(parents=True, exist_ok=True)
    Path(os.environ.get('TRAIN_TF_EVENTS_PATH', './events')).mkdir(parents=True, exist_ok=True)
    log_file = open(Path(os.environ.get('TRAIN_LOG_PATH', './logs'), 'train.log'), 'w')
    writer = SummaryWriter(os.environ.get('TRAIN_TF_EVENTS_PATH'))
    # global dataset
    default_data = PROJECT_ROOT / 'kuairec' / 'data'
    data_path = os.environ.get('TRAIN_DATA_PATH', str(default_data))

    args = get_args()
    dataset = MyDataset(data_path, args)
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=dataset.collate_fn
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=dataset.collate_fn
    )
    usernum, itemnum = dataset.usernum, dataset.itemnum
    feat_statistics, feat_types = dataset.feat_statistics, dataset.feature_types

    model = KuaiRecBaselineModel(usernum, itemnum, feat_statistics, feat_types, args).to(args.device)

    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except Exception:
            pass

    model.pos_emb.weight.data[0, :] = 0
    model.item_emb.weight.data[0, :] = 0
    model.user_emb.weight.data[0, :] = 0

    for k in model.sparse_emb:
        model.sparse_emb[k].weight.data[0, :] = 0

    epoch_start_idx = 1

    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6 :]
            epoch_start_idx = int(tail[: tail.find('.')]) + 1
        except:
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            raise RuntimeError('failed loading state_dicts, pls check file path!')

    #bce_criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    best_val_ndcg, best_val_hr = 0.0, 0.0
    best_test_ndcg, best_test_hr = 0.0, 0.0
    T = 0.0
    t0 = time.time()
    global_step = 0
    print("Start training")
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        model.train()
        if args.inference_only:
            break

        train_loss_sum = 0.0
        train_total = 0
        train_correct = 0
        train_hit5 = 0
        train_hit10 = 0

        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = batch
            seq = seq.to(args.device)
            pos = pos.to(args.device)
            neg = neg.to(args.device)

            '''
            pos_logits, neg_logits = model(
                seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat
            )
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(
                neg_logits.shape, device=args.device
            )
            optimizer.zero_grad()
            indices = np.where(next_token_type == 1)
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])
            '''
            
            log_feats, pos_embs, neg_embs, loss_mask = model(
                seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat
            )
            
            optimizer.zero_grad()
            loss, metrics = model.compute_infonce_loss(
                log_feats, pos_embs, neg_embs, loss_mask, return_metrics=True
            )

            batch_total = metrics["total"] if metrics else 0
            batch_acc = (metrics["correct"] / batch_total) if batch_total else 0.0
            batch_hit5 = (metrics["hit_at_5"] / batch_total) if batch_total else 0.0
            batch_hit10 = (metrics["hit_at_10"] / batch_total) if batch_total else 0.0

            train_loss_sum += loss.item()
            train_total += batch_total
            train_correct += metrics.get("correct", 0)
            train_hit5 += metrics.get("hit_at_5", 0)
            train_hit10 += metrics.get("hit_at_10", 0)

            log_json = json.dumps(
                {
                    'global_step': global_step,
                    'loss': loss.item(),
                    'epoch': epoch,
                    'time': time.time(),
                    'top1_acc': batch_acc,
                    'hit_rate@5': batch_hit5,
                    'hit_rate@10': batch_hit10,
                }
            )
            log_file.write(log_json + '\n')
            log_file.flush()
            print(log_json)

            writer.add_scalar('Loss/train', loss.item(), global_step)
            writer.add_scalar('Metrics/train_top1', batch_acc, global_step)
            writer.add_scalar('Metrics/train_hit5', batch_hit5, global_step)
            writer.add_scalar('Metrics/train_hit10', batch_hit10, global_step)

            global_step += 1

            for param in model.item_emb.parameters():
                loss += args.l2_emb * torch.norm(param)
            loss.backward()
            optimizer.step()

        model.eval()
        valid_loss_sum = 0
        valid_total = 0
        valid_correct = 0
        valid_hit5 = 0
        valid_hit10 = 0
        for step, batch in tqdm(enumerate(valid_loader), total=len(valid_loader)):
            seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = batch
            seq = seq.to(args.device)
            pos = pos.to(args.device)
            neg = neg.to(args.device)
            
            '''
            pos_logits, neg_logits = model(
                seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat
            )
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(
                neg_logits.shape, device=args.device
            )
            indices = np.where(next_token_type == 1)
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])            
            valid_loss_sum += loss.item()
            '''
            
            with torch.no_grad():
                log_feats, pos_embs, neg_embs, loss_mask = model(
                    seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat
                )
                loss, metrics = model.compute_infonce_loss(
                    log_feats, pos_embs, neg_embs, loss_mask, return_metrics=True
                )
            valid_loss_sum += loss.item()
            batch_total = metrics["total"] if metrics else 0
            if batch_total:
                valid_total += batch_total
                valid_correct += metrics.get("correct", 0)
                valid_hit5 += metrics.get("hit_at_5", 0)
                valid_hit10 += metrics.get("hit_at_10", 0)

        valid_loader_len = len(valid_loader)
        valid_loss_avg = valid_loss_sum / valid_loader_len if valid_loader_len else 0.0

        train_loss_avg = train_loss_sum / len(train_loader) if len(train_loader) else 0.0
        train_top1 = (train_correct / train_total) if train_total else 0.0
        train_hr5 = (train_hit5 / train_total) if train_total else 0.0
        train_hr10 = (train_hit10 / train_total) if train_total else 0.0

        valid_top1 = (valid_correct / valid_total) if valid_total else 0.0
        valid_hr5 = (valid_hit5 / valid_total) if valid_total else 0.0
        valid_hr10 = (valid_hit10 / valid_total) if valid_total else 0.0

        writer.add_scalar('Loss/train_epoch', train_loss_avg, epoch)
        writer.add_scalar('Loss/valid', valid_loss_avg, epoch)
        writer.add_scalar('Metrics/train_top1_epoch', train_top1, epoch)
        writer.add_scalar('Metrics/train_hit5_epoch', train_hr5, epoch)
        writer.add_scalar('Metrics/train_hit10_epoch', train_hr10, epoch)
        writer.add_scalar('Metrics/valid_top1_epoch', valid_top1, epoch)
        writer.add_scalar('Metrics/valid_hit5_epoch', valid_hr5, epoch)
        writer.add_scalar('Metrics/valid_hit10_epoch', valid_hr10, epoch)

        summary = {
            'epoch': epoch,
            'phase': 'epoch_summary',
            'train_loss': train_loss_avg,
            'train_top1_acc': train_top1,
            'train_hit_rate@5': train_hr5,
            'train_hit_rate@10': train_hr10,
            'valid_loss': valid_loss_avg,
            'valid_top1_acc': valid_top1,
            'valid_hit_rate@5': valid_hr5,
            'valid_hit_rate@10': valid_hr10,
            'time': time.time(),
        }
        log_file.write(json.dumps(summary) + '\n')
        log_file.flush()
        print(
            f"Epoch {epoch}: train_loss={train_loss_avg:.4f}, valid_loss={valid_loss_avg:.4f}, "
            f"train_top1={train_top1:.4f}, valid_top1={valid_top1:.4f}, "
            f"valid_hit@5={valid_hr5:.4f}, valid_hit@10={valid_hr10:.4f}"
        )

        save_dir = Path(
            os.environ.get('TRAIN_CKPT_PATH','./ckpt_path'),
            f"global_step{global_step}.valid_loss={valid_loss_avg:.4f}"
        )
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_dir / "model.pt")

    print("Done")
    writer.close()
    log_file.close()

