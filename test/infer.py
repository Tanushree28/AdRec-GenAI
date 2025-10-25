import argparse
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT_DIR / "train"
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))

from dataset import MyTestDataset, save_emb
from model import BaselineModel


def get_required_env_path(var_name, expect_dir=False, create=False, description=None):
    """Resolve a filesystem path from an environment variable.

    Args:
        var_name: Environment variable to read.
        expect_dir: Whether the path should exist and be a directory.
        create: If True and expect_dir is True, create the directory when missing.
        description: Optional human readable description for error messages.

    Returns:
        pathlib.Path corresponding to the environment variable.
    """

    value = os.environ.get(var_name)
    if not value:
        human = description or var_name
        raise ValueError(f"{human} is not set; export {var_name} before running inference.")

    path = Path(value)

    if expect_dir:
        if path.exists():
            if not path.is_dir():
                raise NotADirectoryError(f"{var_name}={path} must be a directory, not a file.")
        else:
            if create:
                path.mkdir(parents=True, exist_ok=True)
            else:
                human = description or var_name
                raise FileNotFoundError(f"{human} directory does not exist: {path}")

    return path


def get_ckpt_path():
    ckpt_dir = get_required_env_path("MODEL_OUTPUT_PATH", expect_dir=True, description="Checkpoint directory")
    for item in sorted(os.listdir(ckpt_dir)):
        if item.endswith(".pt"):
            return os.path.join(ckpt_dir, item)
    raise FileNotFoundError(
        "No .pt files were found inside MODEL_OUTPUT_PATH. Ensure you point to a checkpoint folder containing model.pt."
    )


def validate_tencent_eval_dir(data_dir: Path) -> str:
    """Ensure the Tencent evaluation directory has the expected artifacts.

    Returns the name of the sequence file that will be used (either
    ``predict_seq.jsonl`` or ``seq.jsonl``).
    """

    base_required = ["indexer.pkl", "item_feat_dict.json", "predict_set.jsonl"]
    missing_base = [filename for filename in base_required if not (data_dir / filename).exists()]
    if missing_base:
        raise FileNotFoundError(
            "Tencent evaluation directory is missing the following files: "
            + ", ".join(missing_base)
            + ". Verify that EVAL_DATA_PATH points to the preprocessed dataset root."
        )

    sequence_pairs = [
        ("predict_seq.jsonl", "predict_seq_offsets.pkl"),
        ("seq.jsonl", "seq_offsets.pkl"),
    ]

    for seq_name, offsets_name in sequence_pairs:
        seq_path = data_dir / seq_name
        offsets_path = data_dir / offsets_name
        if seq_path.exists() and offsets_path.exists():
            return seq_name
        if seq_path.exists() ^ offsets_path.exists():
            missing = offsets_name if seq_path.exists() else seq_name
            raise FileNotFoundError(
                f"Found {seq_path if seq_path.exists() else offsets_path} but missing {missing}."
                " Ensure both files from the same preprocessing step are present."
            )

    raise FileNotFoundError(
        "Tencent evaluation directory is missing predict_seq.jsonl/predict_seq_offsets.pkl "
        "and seq.jsonl/seq_offsets.pkl. Provide at least one processed sequence set."
    )


def get_args():
    parser = argparse.ArgumentParser()

    # Train params
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--maxlen', default=101, type=int)

    # Baseline Model construction
    parser.add_argument('--hidden_units', default=32, type=int)
    parser.add_argument('--num_blocks', default=1, type=int)
    parser.add_argument('--num_epochs', default=3, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--l2_emb', default=0.0, type=float)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--inference_only', action='store_true')
    parser.add_argument('--state_dict_path', default=None, type=str)
    parser.add_argument('--norm_first', action='store_true')

    # MMemb Feature ID
    parser.add_argument('--mm_emb_id', nargs='+', default=['81'], type=str, choices=[str(s) for s in range(81, 87)])

    args = parser.parse_args()

    return args


def read_result_ids(file_path):
    with open(file_path, 'rb') as f:
        # Read the header (num_points_query and FLAGS_query_ann_top_k)
        num_points_query = struct.unpack('I', f.read(4))[0]  # uint32_t -> 4 bytes
        query_ann_top_k = struct.unpack('I', f.read(4))[0]  # uint32_t -> 4 bytes

        print(f"num_points_query: {num_points_query}, query_ann_top_k: {query_ann_top_k}")

        # Calculate how many result_ids there are (num_points_query * query_ann_top_k)
        num_result_ids = num_points_query * query_ann_top_k

        # Read result_ids (uint64_t, 8 bytes per value)
        result_ids = np.fromfile(f, dtype=np.uint64, count=num_result_ids)

        return result_ids.reshape((num_points_query, query_ann_top_k))


def process_cold_start_feat(feat):
    """
    处理冷启动特征。训练集未出现过的特征value为字符串，默认转换为0.可设计替换为更好的方法。
    """
    processed_feat = {}
    for feat_id, feat_value in feat.items():
        if type(feat_value) == list:
            value_list = []
            for v in feat_value:
                if type(v) == str:
                    value_list.append(0)
                else:
                    value_list.append(v)
            processed_feat[feat_id] = value_list
        elif type(feat_value) == str:
            processed_feat[feat_id] = 0
        else:
            processed_feat[feat_id] = feat_value
    return processed_feat


def get_candidate_emb(indexer, feat_types, feat_default_value, mm_emb_dict, model, data_root: Path, result_root: Path):
    """
    生产候选库item的id和embedding

    Args:
        indexer: 索引字典
        feat_types: 特征类型，分为user和item的sparse, array, emb, continual类型
        feature_default_value: 特征缺省值
        mm_emb_dict: 多模态特征字典
        model: 模型
    Returns:
        retrieve_id2creative_id: 索引id->creative_id的dict
    """
    EMB_SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
    candidate_path = data_root / 'predict_set.jsonl'
    if not candidate_path.exists():
        raise FileNotFoundError(
            f"predict_set.jsonl was not found under {data_root}. Export the Tencent candidate file before running inference."
        )
    print(f"[infer] Loading candidate items from {candidate_path}")
    item_ids, creative_ids, retrieval_ids, features = [], [], [], []
    retrieve_id2creative_id = {}

    with open(candidate_path, 'r') as f:
        for line in f:
            line = json.loads(line)
            # 读取item特征，并补充缺失值
            feature = line['features']
            creative_id = line['creative_id']
            retrieval_id = line['retrieval_id']
            item_id = indexer[creative_id] if creative_id in indexer else 0
            missing_fields = set(
                feat_types['item_sparse'] + feat_types['item_array'] + feat_types['item_continual']
            ) - set(feature.keys())
            feature = process_cold_start_feat(feature)
            for feat_id in missing_fields:
                feature[feat_id] = feat_default_value[feat_id]
            for feat_id in feat_types['item_emb']:
                if creative_id in mm_emb_dict[feat_id]:
                    feature[feat_id] = mm_emb_dict[feat_id][creative_id]
                else:
                    feature[feat_id] = np.zeros(EMB_SHAPE_DICT[feat_id], dtype=np.float32)

            item_ids.append(item_id)
            creative_ids.append(creative_id)
            retrieval_ids.append(retrieval_id)
            features.append(feature)
            retrieve_id2creative_id[retrieval_id] = creative_id

    # 保存候选库的embedding和sid
    print(f"[infer] Saving candidate embeddings to {result_root}")
    model.save_item_emb(item_ids, retrieval_ids, features, str(result_root))
    with open(result_root / "retrive_id2creative_id.json", "w") as f:
        json.dump(retrieve_id2creative_id, f)
    return retrieve_id2creative_id


def infer():
    args = get_args()
    data_path = get_required_env_path("EVAL_DATA_PATH", expect_dir=True, description="Evaluation dataset")
    sequence_file = validate_tencent_eval_dir(data_path)
    print(f"[infer] Using evaluation dataset at: {data_path} (sequence source: {sequence_file})")
    test_dataset = MyTestDataset(data_path, args)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=test_dataset.collate_fn
    )
    usernum, itemnum = test_dataset.usernum, test_dataset.itemnum
    feat_statistics, feat_types = test_dataset.feat_statistics, test_dataset.feature_types
    model = BaselineModel(usernum, itemnum, feat_statistics, feat_types, args).to(args.device)
    model.eval()

    ckpt_path = get_ckpt_path()
    print(f"[infer] Loading checkpoint: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=torch.device(args.device)))
    all_embs = []
    user_list = []
    print(
        f"[infer] Generating user embeddings from {test_dataset.sequence_source} for {len(test_loader)} batches..."
    )
    for step, batch in tqdm(enumerate(test_loader), total=len(test_loader)):

        seq, token_type, seq_feat, user_id = batch
        seq = seq.to(args.device)
        logits = model.predict(seq, seq_feat, token_type)
        for i in range(logits.shape[0]):
            emb = logits[i].unsqueeze(0).detach().cpu().numpy().astype(np.float32)
            all_embs.append(emb)
        user_list += user_id

    # 生成候选库的embedding 以及 id文件
    result_root = get_required_env_path(
        "EVAL_RESULT_PATH", expect_dir=True, create=True, description="Evaluation result output"
    )
    print(f"[infer] Writing inference artifacts to: {result_root}")
    retrieve_id2creative_id = get_candidate_emb(
        test_dataset.indexer['i'],
        test_dataset.feature_types,
        test_dataset.feature_default_value,
        test_dataset.mm_emb_dict,
        model,
        data_path,
        result_root,
    )
    if not all_embs:
        raise ValueError(
            "No user embeddings were generated. Check that predict_seq.jsonl contains evaluation sequences and that "
            "EVAL_DATA_PATH is set to the evaluation dataset root."
        )
    all_embs = np.concatenate(all_embs, axis=0)
    print(f"[infer] Exporting {all_embs.shape[0]} user embeddings to {result_root / 'query.fbin'}")
    # 保存query文件
    save_emb(all_embs, result_root / 'query.fbin')
    # ANN 检索
    ann_cmd = (
        str(Path("/workspace", "faiss-based-ann", "faiss_demo"))
        + " --dataset_vector_file_path="
        + str(result_root / "embedding.fbin")
        + " --dataset_id_file_path="
        + str(result_root / "id.u64bin")
        + " --query_vector_file_path="
        + str(result_root / "query.fbin")
        + " --result_id_file_path="
        + str(result_root / "id100.u64bin")
        + " --query_ann_top_k=10 --faiss_M=64 --faiss_ef_construction=1280 --query_ef_search=640 --faiss_metric_type=0"
    )
    print("[infer] Running ANN retrieval command...")
    ann_status = os.system(ann_cmd)
    if ann_status != 0:
        raise RuntimeError(
            "ANN retrieval command failed. Ensure the faiss_demo binary is available and executable."
        )

    # 取出top-k
    ann_output = result_root / "id100.u64bin"
    if not ann_output.exists():
        raise FileNotFoundError(
            f"Expected ANN output {ann_output} was not created. Check the FAISS command output for errors."
        )
    print(f"[infer] Reading ANN results from {ann_output}")
    top10s_retrieved = read_result_ids(ann_output)
    top10s_untrimmed = []
    print("[infer] Converting retrieval ids to creative ids...")
    for top10 in tqdm(top10s_retrieved):
        for item in top10:
            top10s_untrimmed.append(retrieve_id2creative_id.get(int(item), 0))

    top10s = [top10s_untrimmed[i : i + 10] for i in range(0, len(top10s_untrimmed), 10)]
    print("[infer] Inference complete.")

    return top10s, user_list
