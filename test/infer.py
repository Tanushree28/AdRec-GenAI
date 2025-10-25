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


def normalise_device(device_str: str) -> str:
    """Validate the requested device string and fall back to CPU when unavailable."""

    requested = (device_str or "cpu").strip()

    try:
        device = torch.device(requested)
    except (TypeError, RuntimeError):
        print(
            f"[infer] WARNING: Unrecognised device '{requested}'; defaulting to CPU.",
            file=sys.stderr,
        )
        return "cpu"

    if device.type == "cuda":
        if not torch.cuda.is_available():
            print(
                f"[infer] WARNING: CUDA requested ('{requested}') but torch reports no CUDA support. Using CPU instead.",
                file=sys.stderr,
            )
            return "cpu"
    elif device.type == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            print(
                f"[infer] WARNING: MPS requested ('{requested}') but is not available. Using CPU instead.",
                file=sys.stderr,
            )
            return "cpu"

    return str(device)


def get_ckpt_path():
    ckpt_path = os.environ.get("MODEL_OUTPUT_PATH")
    if not ckpt_path:
        raise ValueError("MODEL_OUTPUT_PATH is not set")
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"MODEL_OUTPUT_PATH does not exist: {ckpt_path}")

    if ckpt_path.is_file():
        if ckpt_path.suffix != ".pt":
            raise ValueError(
                f"MODEL_OUTPUT_PATH points to a file that is not a .pt checkpoint: {ckpt_path}"
            )
        return str(ckpt_path)

    if not ckpt_path.is_dir():
        raise NotADirectoryError(f"MODEL_OUTPUT_PATH must be a directory or .pt file: {ckpt_path}")

    for item in sorted(ckpt_path.iterdir()):
        if item.suffix == ".pt":
            return str(item)
    raise FileNotFoundError("No .pt file found in MODEL_OUTPUT_PATH")


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
    parser.add_argument('--mm_emb_id', nargs='+', default=['82'], type=str, choices=[str(s) for s in range(81, 87)])

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


def get_candidate_emb(
    indexer,
    feat_types,
    feat_default_value,
    mm_emb_dict,
    model,
    data_root=None,
    result_root=None,
    item_feat_dict=None,
):
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
    data_root = Path(data_root) if data_root is not None else Path(os.environ.get('EVAL_DATA_PATH'))
    result_root = Path(result_root) if result_root is not None else Path(os.environ.get('EVAL_RESULT_PATH'))
    candidate_path = data_root / 'predict_set.jsonl'
    use_predict_set = candidate_path.exists()
    item_ids, creative_ids, retrieval_ids, features = [], [], [], []
    retrieve_id2creative_id = {}

    if use_predict_set:
        print(f"[infer] Loading candidate set from {candidate_path}")
        with open(candidate_path, 'r') as f:
            for line in f:
                line = json.loads(line)
                feature = line['features']
                creative_id = line['creative_id']
                retrieval_id = line['retrieval_id']
                item_id = indexer.get(creative_id, 0)

                feature = process_cold_start_feat(feature)
                missing_fields = set(
                    feat_types['item_sparse'] + feat_types['item_array'] + feat_types['item_continual']
                ) - set(feature.keys())
                for feat_id in missing_fields:
                    feature[feat_id] = feat_default_value[feat_id]

                for feat_id in feat_types['item_emb']:
                    emb_lookup = mm_emb_dict.get(feat_id, {})
                    if creative_id in emb_lookup:
                        feature[feat_id] = emb_lookup[creative_id]
                    else:
                        feature[feat_id] = np.zeros(EMB_SHAPE_DICT[feat_id], dtype=np.float32)

                item_ids.append(item_id)
                creative_ids.append(creative_id)
                retrieval_ids.append(retrieval_id)
                features.append(feature)
                retrieve_id2creative_id[retrieval_id] = creative_id
    else:
        if item_feat_dict is None:
            item_feat_path = data_root / 'item_feat_dict.json'
            if not item_feat_path.exists():
                raise FileNotFoundError(
                    "predict_set.jsonl not found under "
                    f"{data_root} and item_feat_dict.json is unavailable for fallback"
                )
            with open(item_feat_path, 'r') as f:
                item_feat_dict = json.load(f)

        if not isinstance(item_feat_dict, dict):
            item_feat_dict = {}

        print(
            "[infer] WARNING: predict_set.jsonl not found; generating candidate set from item_feat_dict"
        )

        def _resolve_feature_for_creative(raw_map, creative_key, item_id_value):
            """Return a feature dict for the given creative, trying common key variants."""

            if not isinstance(raw_map, dict):
                return None

            candidate_keys = []

            # Preserve original key as string
            candidate_keys.append(str(creative_key))

            # Some JSON dumps store numeric IDs without quotes; try the original
            candidate_keys.append(creative_key)

            # Include numeric conversions for both the creative ID and the retrieval ID
            try:
                candidate_keys.append(int(creative_key))
            except (TypeError, ValueError):
                pass

            candidate_keys.extend([item_id_value, str(item_id_value)])

            for key in dict.fromkeys(candidate_keys):
                if key in raw_map:
                    return raw_map[key]
            return None

        # ``indexer`` maps creative_id -> item_id. Iterate over it so we cover
        # every item observed during training even if the feature dict omits a
        # few entries (we will fill defaults in that case).
        for creative_id, item_id in sorted(indexer.items(), key=lambda kv: kv[1]):
            if item_id in (None, 0):
                continue
            feature_data = _resolve_feature_for_creative(
                item_feat_dict, creative_id, item_id
            )

            feature = process_cold_start_feat(feature_data or {})

            missing_fields = set(
                feat_types['item_sparse'] + feat_types['item_array'] + feat_types['item_continual']
            ) - set(feature.keys())
            for feat_id in missing_fields:
                feature[feat_id] = feat_default_value[feat_id]

            for feat_id in feat_types['item_emb']:
                emb_lookup = mm_emb_dict.get(feat_id, {})
                emb_value = None
                if creative_id in emb_lookup:
                    emb_value = emb_lookup[creative_id]
                elif str(creative_id) in emb_lookup:
                    emb_value = emb_lookup[str(creative_id)]
                elif item_id in emb_lookup:
                    emb_value = emb_lookup[item_id]
                elif str(item_id) in emb_lookup:
                    emb_value = emb_lookup[str(item_id)]

                if emb_value is None:
                    emb_value = np.zeros(EMB_SHAPE_DICT[feat_id], dtype=np.float32)

                feature[feat_id] = emb_value

            item_ids.append(item_id)
            creative_ids.append(creative_id)
            retrieval_ids.append(item_id)
            features.append(feature)
            retrieve_id2creative_id[item_id] = creative_id

    if not item_ids:
        raise ValueError(
            "No candidate items could be constructed. Ensure predict_set.jsonl or item_feat_dict.json "
            "contains entries matching the training indexer."
        )

    # 保存候选库的embedding和sid
    result_root.mkdir(parents=True, exist_ok=True)
    print(f"[infer] Encoding {len(item_ids)} candidate items")
    model.save_item_emb(item_ids, retrieval_ids, features, str(result_root))
    with open(result_root / "retrive_id2creative_id.json", "w") as f:
        json.dump(retrieve_id2creative_id, f)
    print(
        f"[infer] Candidate artifacts saved to {result_root} (retrive_id2creative_id.json, embedding/id files)"
    )
    return retrieve_id2creative_id


def _infer_mm_ids_from_state_dict(state_dict):
    """Extract the multimedia embedding feature IDs present in a checkpoint."""

    mm_ids = []
    for key in state_dict.keys():
        if key.startswith("emb_transform."):
            parts = key.split(".")
            if len(parts) >= 3:
                mm_ids.append(parts[1])
    return sorted(set(mm_ids))


def _maybe_override_mm_ids(args, state_dict):
    ckpt_mm_ids = _infer_mm_ids_from_state_dict(state_dict)
    if ckpt_mm_ids and sorted(args.mm_emb_id) != ckpt_mm_ids:
        print(
            "[infer] WARNING: Checkpoint was trained with mm_emb_id="
            f"{ckpt_mm_ids}, overriding requested values {args.mm_emb_id}"
        )
        args.mm_emb_id = ckpt_mm_ids


def infer(args=None):
    args = args or get_args()
    args.device = normalise_device(getattr(args, "device", "cpu"))

    ckpt_path = get_ckpt_path()
    print(f"[infer] Loading checkpoint: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location="cpu")
    _maybe_override_mm_ids(args, state_dict)

    print("[infer] Parsed arguments:")
    for key in sorted(vars(args)):
        if key == "mm_emb_id":
            continue
        value = getattr(args, key)
        print(f"    {key}: {value}")
    print(f"    mm_emb_id: {', '.join(args.mm_emb_id) if args.mm_emb_id else 'none'}")
    data_path = os.environ.get('EVAL_DATA_PATH')
    if not data_path:
        raise ValueError("EVAL_DATA_PATH is not set")
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"EVAL_DATA_PATH does not exist: {data_path}")
    if not data_path.is_dir():
        raise NotADirectoryError(f"EVAL_DATA_PATH must be a directory: {data_path}")
    print(f"[infer] Loading evaluation data from: {data_path}")
    test_dataset = MyTestDataset(data_path, args)
    sequence_source = getattr(test_dataset, "sequence_source", "seq.jsonl")
    num_users = len(test_dataset)
    print(f"[infer] Using sequence file: {sequence_source} | users detected: {num_users}")
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=test_dataset.collate_fn
    )
    num_batches = len(test_loader)
    usernum, itemnum = test_dataset.usernum, test_dataset.itemnum
    feat_statistics, feat_types = test_dataset.feat_statistics, test_dataset.feature_types
    model = BaselineModel(usernum, itemnum, feat_statistics, feat_types, args).to(args.device)
    model.eval()

    load_result = model.load_state_dict(state_dict, strict=False)
    missing_keys = getattr(load_result, "missing_keys", [])
    unexpected_keys = getattr(load_result, "unexpected_keys", [])
    if missing_keys:
        print(f"[infer] WARNING: Missing keys when loading checkpoint: {missing_keys}")
    if unexpected_keys:
        print(f"[infer] WARNING: Unexpected keys when loading checkpoint: {unexpected_keys}")
    all_embs = []
    user_list = []
    print(f"[infer] Starting query embedding export across {num_batches} batches")
    for step, batch in tqdm(enumerate(test_loader), total=num_batches, desc="Query batches"):

        seq, token_type, seq_feat, user_id = batch
        seq = seq.to(args.device)
        logits = model.predict(seq, seq_feat, token_type)
        for i in range(logits.shape[0]):
            emb = logits[i].unsqueeze(0).detach().cpu().numpy().astype(np.float32)
            all_embs.append(emb)
        user_list += user_id

    # 生成候选库的embedding 以及 id文件
    result_root = os.environ.get('EVAL_RESULT_PATH')
    if not result_root:
        raise ValueError("EVAL_RESULT_PATH is not set")
    result_root = Path(result_root)
    result_root.mkdir(parents=True, exist_ok=True)
    print(f"[infer] Writing inference artifacts to: {result_root}")

    retrieve_id2creative_id = get_candidate_emb(
        test_dataset.indexer['i'],
        test_dataset.feature_types,
        test_dataset.feature_default_value,
        test_dataset.mm_emb_dict,
        model,
        data_root=data_path,
        result_root=result_root,
        item_feat_dict=getattr(test_dataset, 'item_feat_dict', None),
    )
    if not all_embs:
        raise ValueError("No query embeddings were generated. Check that the evaluation dataset is not empty.")
    all_embs = np.concatenate(all_embs, axis=0)
    print(f"[infer] Query embedding matrix shape: {all_embs.shape}")
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
    print(f"[infer] Running ANN retrieval command:\n{ann_cmd}")
    exit_code = os.system(ann_cmd)
    if exit_code != 0:
        raise RuntimeError(
            "FAISS demo command failed. Ensure the ANN binary is available at /workspace/faiss-based-ann/faiss_demo"
        )

    # 取出top-k
    print(f"[infer] Reading ANN results from {result_root / 'id100.u64bin'}")
    top10s_retrieved = read_result_ids(result_root / "id100.u64bin")
    top10s_untrimmed = []
    for top10 in tqdm(top10s_retrieved, desc="Mapping ANN IDs"):
        for item in top10:
            top10s_untrimmed.append(retrieve_id2creative_id.get(int(item), 0))

    top10s = [top10s_untrimmed[i : i + 10] for i in range(0, len(top10s_untrimmed), 10)]

    print(f"[infer] Inference complete: produced recommendations for {len(user_list)} users")
    return top10s, user_list


def main():
    try:
        top10s, user_list = infer()
    except Exception as exc:
        print(f"[infer] ERROR: {exc}", file=sys.stderr)
        raise

    if user_list:
        preview = top10s[0] if top10s else []
        print(
            f"[infer] Example user {user_list[0]} top-10 list: {preview if preview else 'no results'}"
        )
    else:
        print("[infer] No users were processed.")


if __name__ == "__main__":
    main()
