import ast
import json
import pickle
import struct
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


class MyDataset(torch.utils.data.Dataset):
    """
    用户序列数据集

    Args:
        data_dir: 数据文件目录
        args: 全局参数

    Attributes:
        data_dir: 数据文件目录
        maxlen: 最大长度
        item_feat_dict: 物品特征字典
        mm_emb_ids: 激活的mm_emb特征ID
        mm_emb_dict: 多模态特征字典
        itemnum: 物品数量
        usernum: 用户数量
        indexer_i_rev: 物品索引字典 (reid -> item_id)
        indexer_u_rev: 用户索引字典 (reid -> user_id)
        indexer: 索引字典
        feature_default_value: 特征缺省值
        feature_types: 特征类型，分为user和item的sparse, array, emb, continual类型
        feat_statistics: 特征统计信息，包括user和item的特征数量
    """

    def __init__(self, data_dir, args):
        """
        初始化数据集
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.maxlen = args.maxlen
        self.dataset_type = self._detect_dataset_type()

        if self.dataset_type == 'kuairec':
            if getattr(args, 'mm_emb_id', None):
                print('KuaiRec dataset detected: ignoring provided mm_emb_id settings.')
            self.mm_emb_ids = []
        else:
            self.mm_emb_ids = args.mm_emb_id

        self._load_data_and_offsets()

        if self.dataset_type == 'tencent':
            self.item_feat_dict = json.load(open(Path(data_dir, "item_feat_dict.json"), 'r'))
            self.mm_emb_dict = load_mm_emb(Path(data_dir, "creative_emb"), self.mm_emb_ids)
            with open(self.data_dir / 'indexer.pkl', 'rb') as ff:
                indexer = pickle.load(ff)
                self.itemnum = len(indexer['i'])
                self.usernum = len(indexer['u'])
            self.indexer_i_rev = {v: k for k, v in indexer['i'].items()}
            self.indexer_u_rev = {v: k for k, v in indexer['u'].items()}
            self.indexer = indexer
            self.feature_default_value, self.feature_types, self.feat_statistics = self._init_feat_info()

    def _detect_dataset_type(self):
        if (self.data_dir / "seq.jsonl").exists():
            return 'tencent'
        if (self.data_dir / "predict_seq.jsonl").exists():
            return 'tencent'
        candidate_dirs = [self.data_dir]
        if (self.data_dir / 'data').exists():
            candidate_dirs.append(self.data_dir / 'data')
        for candidate in candidate_dirs:
            if (candidate / 'small_matrix.csv').exists() or (candidate / 'big_matrix.csv').exists():
                self.kuairec_root = candidate
                return 'kuairec'
        raise FileNotFoundError(
            f"Unable to detect dataset format under {self.data_dir}. Expected Tencent preprocessed files or KuaiRec CSVs."
        )

    def _load_data_and_offsets(self):
        """
        加载用户序列数据和每一行的文件偏移量(预处理好的), 用于快速随机访问数据并I/O
        """
        if self.dataset_type == 'tencent':
            self.data_file = open(self.data_dir / "seq.jsonl", 'rb')
            with open(Path(self.data_dir, 'seq_offsets.pkl'), 'rb') as f:
                self.seq_offsets = pickle.load(f)
        else:
            self._load_kuairec_dataset()

    def _load_user_data(self, uid):
        """
        从数据文件中加载单个用户的数据

        Args:
            uid: 用户ID(reid)

        Returns:
            data: 用户序列数据，格式为[(user_id, item_id, user_feat, item_feat, action_type, timestamp)]
        """
        if self.dataset_type == 'tencent':
            self.data_file.seek(self.seq_offsets[uid])
            line = self.data_file.readline()
            data = json.loads(line)
            return data
        user_reid = self.uid_list[uid]
        return self.user_sequences[user_reid]

    def _load_kuairec_dataset(self):
        data_root = getattr(self, 'kuairec_root', self.data_dir)
        if not data_root.exists():
            data_root = self.data_dir

        interaction_path = data_root / 'small_matrix.csv'
        if not interaction_path.exists():
            interaction_path = data_root / 'big_matrix.csv'
        if not interaction_path.exists():
            raise FileNotFoundError(f"KuaiRec dataset not found under {data_root}.")

        interactions = pd.read_csv(interaction_path)

        user_col = self._find_column(interactions.columns, ['user_id', 'userid', 'uid'])
        item_col = self._find_column(interactions.columns, ['item_id', 'video_id', 'iid', 'cid'])
        if user_col is None or item_col is None:
            raise ValueError('KuaiRec interactions must contain user and item identifier columns.')

        time_col = self._find_column(interactions.columns, ['timestamp', 'time', 'datetime'])
        action_col = self._find_column(
            interactions.columns,
            ['is_click', 'click', 'finish', 'like', 'view', 'watch', 'play', 'interaction'],
        )

        interactions = interactions.dropna(subset=[user_col, item_col])
        interactions[user_col] = interactions[user_col].astype(str)
        interactions[item_col] = interactions[item_col].astype(str)

        user_ids = sorted(interactions[user_col].unique())
        item_ids = sorted(interactions[item_col].unique())
        user2reid = {user_id: idx + 1 for idx, user_id in enumerate(user_ids)}
        item2reid = {item_id: idx + 1 for idx, item_id in enumerate(item_ids)}

        user_feat_path = data_root / 'user_features.csv'
        if user_feat_path.exists():
            user_feat_df = pd.read_csv(user_feat_path)
            user_feat_col = self._find_column(user_feat_df.columns, ['user_id', 'userid', 'uid'])
            if user_feat_col is not None:
                user_feat_df[user_feat_col] = user_feat_df[user_feat_col].astype(str)
                user_feat_df = user_feat_df.drop_duplicates(subset=[user_feat_col]).set_index(user_feat_col)
            else:
                user_feat_df = pd.DataFrame()
        else:
            user_feat_df = pd.DataFrame()

        item_cat_path = data_root / 'item_categories.csv'
        if item_cat_path.exists():
            item_cat_df = pd.read_csv(item_cat_path)
            item_feat_col = self._find_column(item_cat_df.columns, ['item_id', 'video_id', 'iid', 'cid'])
            if item_feat_col is not None:
                item_cat_df[item_feat_col] = item_cat_df[item_feat_col].astype(str)
                item_cat_df = item_cat_df.drop_duplicates(subset=[item_feat_col])
                if 'feat' in item_cat_df.columns:
                    item_cat_df['feat'] = item_cat_df['feat'].apply(lambda x: self._parse_possible_list(x) or [])
                if item_feat_col != item_col:
                    item_cat_df = item_cat_df.rename(columns={item_feat_col: item_col})
            else:
                item_cat_df = pd.DataFrame({item_col: []})
        else:
            item_cat_df = pd.DataFrame({item_col: []})

        exclude_cols = {user_col, item_col}
        if time_col:
            exclude_cols.add(time_col)
        if action_col:
            exclude_cols.add(action_col)
        extra_cols = [col for col in interactions.columns if col not in exclude_cols]
        if extra_cols:
            item_extra_df = interactions.groupby(item_col)[extra_cols].first().reset_index()
        else:
            item_extra_df = pd.DataFrame(columns=[item_col])

        if not item_cat_df.empty and not item_extra_df.empty:
            item_feature_df = pd.merge(item_cat_df, item_extra_df, on=item_col, how='outer')
        elif not item_cat_df.empty:
            item_feature_df = item_cat_df.copy()
        else:
            item_feature_df = item_extra_df.copy()

        if not item_feature_df.empty:
            item_feature_df[item_col] = item_feature_df[item_col].astype(str)
            item_feature_df = item_feature_df.set_index(item_col)

        self.mm_emb_dict = {}
        self.itemnum = len(item2reid)
        self.usernum = len(user2reid)
        self.indexer_i_rev = {v: k for k, v in item2reid.items()}
        self.indexer_u_rev = {v: k for k, v in user2reid.items()}
        self.indexer = {'i': item2reid, 'u': user2reid, 'f': {}}

        self.feature_types = {
            'user_sparse': [],
            'item_sparse': [],
            'user_array': [],
            'item_array': [],
            'user_continual': [],
            'item_continual': [],
            'item_emb': [],
        }
        self.feature_default_value = {}
        self.feat_statistics = {}
        self.kuairec_user_feature_specs = {}
        self.kuairec_item_feature_specs = {}
        self.kuairec_user_column2feat = {}
        self.kuairec_item_column2feat = {}

        # Ensure bias features exist so each record retains user/item entries
        self.feature_types['user_continual'].append('u_bias')
        self.feature_default_value['u_bias'] = 0.0
        self.indexer['f']['u_bias'] = {}
        self.feature_types['item_continual'].append('i_bias')
        self.feature_default_value['i_bias'] = 0.0
        self.indexer['f']['i_bias'] = {}

        if not user_feat_df.empty:
            for column in user_feat_df.columns:
                series = user_feat_df[column]
                if series.dropna().empty:
                    continue
                feat_id = f'u_{column}'
                spec = self._analyze_feature_series(series, 'user', column)
                if spec is None:
                    continue
                self.kuairec_user_column2feat[column] = feat_id
                self._register_feature_spec(feat_id, 'user', spec)

        if not item_feature_df.empty:
            for column in item_feature_df.columns:
                series = item_feature_df[column]
                if isinstance(series, pd.Series) and series.dropna().empty:
                    continue
                feat_id = f'i_{column}'
                spec = self._analyze_feature_series(series, 'item', column)
                if spec is None:
                    continue
                self.kuairec_item_column2feat[column] = feat_id
                self._register_feature_spec(feat_id, 'item', spec)

        self.user_feature_lookup = {}
        for user_id, user_reid in user2reid.items():
            feat_dict = {'u_bias': 1.0}
            if not user_feat_df.empty and user_id in user_feat_df.index:
                row = user_feat_df.loc[user_id]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                for column, feat_id in self.kuairec_user_column2feat.items():
                    value = row[column]
                    encoded = self._encode_feature_value(value, feat_id, self.kuairec_user_feature_specs)
                    if encoded is not None:
                        feat_dict[feat_id] = encoded
            self.user_feature_lookup[user_reid] = feat_dict

        self.item_feat_dict = {}
        if not item_feature_df.empty:
            for item_id, item_reid in item2reid.items():
                feat_dict = {'i_bias': 1.0}
                if item_id in item_feature_df.index:
                    row = item_feature_df.loc[item_id]
                    if isinstance(row, pd.DataFrame):
                        row = row.iloc[0]
                    for column, feat_id in self.kuairec_item_column2feat.items():
                        value = row[column]
                        encoded = self._encode_feature_value(value, feat_id, self.kuairec_item_feature_specs)
                        if encoded is not None:
                            feat_dict[feat_id] = encoded
                self.item_feat_dict[str(item_reid)] = feat_dict
        else:
            for item_id, item_reid in item2reid.items():
                self.item_feat_dict[str(item_reid)] = {'i_bias': 1.0}

        self.user_sequences = {}
        sorted_interactions = interactions if not time_col else interactions.sort_values(time_col)
        for user_id, group in sorted_interactions.groupby(user_col):
            user_reid = user2reid[user_id]
            user_feat = self.user_feature_lookup.get(user_reid, {'u_bias': 1.0})
            seq_records = []
            ordered_group = group if time_col is None else group.sort_values(time_col)
            for order, (_, row) in enumerate(ordered_group.iterrows()):
                item_id = row[item_col]
                if item_id not in item2reid:
                    continue
                item_reid = item2reid[item_id]
                item_feat = self.item_feat_dict.get(str(item_reid), {'i_bias': 1.0})
                action_value = self._extract_action_value(row, action_col)
                timestamp = self._extract_timestamp(row, time_col, order)
                seq_records.append((user_reid, item_reid, dict(user_feat), dict(item_feat), action_value, timestamp))
            if seq_records:
                self.user_sequences[user_reid] = seq_records

        self.uid_list = sorted(self.user_sequences.keys())
        self.seq_offsets = list(range(len(self.uid_list)))

        print(
            f"Loaded KuaiRec interactions from {interaction_path} with {self.usernum} users and {self.itemnum} items."
        )

    @staticmethod
    def _find_column(columns, candidates):
        lowered = {col.lower(): col for col in columns}
        for candidate in candidates:
            for col in columns:
                if candidate in col.lower():
                    return col
        return None

    def _parse_possible_list(self, value):
        if isinstance(value, str):
            trimmed = value.strip()
            if not trimmed:
                return []
            if trimmed[0] in {'[', '{', '('} and trimmed[-1] in {']', '}', ')'}:
                try:
                    parsed = ast.literal_eval(trimmed)
                except (ValueError, SyntaxError):
                    return []
            else:
                return None
        elif isinstance(value, dict):
            parsed = list(value.values())
        elif isinstance(value, (list, tuple, set)):
            parsed = list(value)
        else:
            return None

        if isinstance(parsed, dict):
            parsed = list(parsed.values())
        if isinstance(parsed, (list, tuple, set)):
            cleaned = []
            for item in parsed:
                if item is None:
                    continue
                if isinstance(item, float) and np.isnan(item):
                    continue
                cleaned.append(item)
            return cleaned
        return None

    def _analyze_feature_series(self, series, scope, column_name):
        non_null = series.dropna()
        if non_null.empty:
            return None

        sample_values = non_null.iloc[: min(len(non_null), 50)]
        is_array = False
        for value in sample_values:
            parsed = self._parse_possible_list(value)
            if parsed is not None:
                is_array = True
                break
            if isinstance(value, (list, tuple, set, dict)):
                is_array = True
                break

        if is_array:
            tokens = set()
            for value in non_null:
                parsed = self._parse_possible_list(value)
                if parsed is None:
                    if isinstance(value, dict):
                        parsed = list(value.values())
                    elif isinstance(value, (list, tuple, set)):
                        parsed = list(value)
                    else:
                        continue
                for token in parsed:
                    if token is None:
                        continue
                    tokens.add(str(token))
            if not tokens:
                return None
            mapping = {token: idx + 1 for idx, token in enumerate(sorted(tokens))}
            return {'type': 'array', 'mapping': mapping, 'default': [0]}

        unique_count = non_null.nunique(dropna=True)
        max_sparse_threshold = 200
        if non_null.dtype == object or unique_count <= max_sparse_threshold:
            unique_tokens = sorted(set(non_null.astype(str)))
            mapping = {token: idx + 1 for idx, token in enumerate(unique_tokens)}
            return {'type': 'sparse', 'mapping': mapping, 'default': 0}

        return {'type': 'continual', 'mapping': None, 'default': 0.0}

    def _register_feature_spec(self, feat_id, scope, spec):
        feature_type = spec['type']
        key = f"{scope}_{feature_type}"
        if key not in self.feature_types:
            self.feature_types[key] = []
        if feat_id not in self.feature_types[key]:
            self.feature_types[key].append(feat_id)
        self.feature_default_value[feat_id] = spec['default']
        if spec['mapping'] is not None:
            self.indexer['f'][feat_id] = spec['mapping']
            self.feat_statistics[feat_id] = len(spec['mapping'])
        else:
            self.indexer['f'][feat_id] = {}

        if scope == 'user':
            self.kuairec_user_feature_specs[feat_id] = spec
        else:
            self.kuairec_item_feature_specs[feat_id] = spec

    def _encode_feature_value(self, value, feat_id, spec_dict):
        spec = spec_dict.get(feat_id)
        if spec is None:
            return None
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None

        feature_type = spec['type']
        if feature_type == 'continual':
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        if feature_type == 'sparse':
            key = str(value)
            return spec['mapping'].get(key, 0)

        if feature_type == 'array':
            parsed = self._parse_possible_list(value)
            if parsed is None:
                if isinstance(value, dict):
                    parsed = list(value.values())
                elif isinstance(value, (list, tuple, set)):
                    parsed = list(value)
                else:
                    parsed = [value]
            encoded = [spec['mapping'].get(str(token), 0) for token in parsed if token is not None]
            return encoded if encoded else [0]

        return None

    def _extract_action_value(self, row, action_col):
        if action_col is None:
            return 1
        value = row[action_col]
        if isinstance(value, (list, dict)):
            return 1
        if pd.isna(value):
            return 0
        if isinstance(value, str):
            stripped = value.strip().lower()
            if not stripped:
                return 0
            if stripped in {'true', 'yes', 'y'}:
                return 1
            if stripped in {'false', 'no', 'n'}:
                return 0
            try:
                value = float(stripped)
            except ValueError:
                return 0
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return 0

    def _extract_timestamp(self, row, time_col, fallback):
        if time_col is None:
            return fallback
        value = row[time_col]
        if pd.isna(value):
            return fallback
        if isinstance(value, (np.integer, int)):
            return int(value)
        if isinstance(value, (np.floating, float)):
            if np.isnan(value):
                return fallback
            return float(value)
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return fallback
            try:
                parsed = pd.to_datetime(stripped, errors='coerce')
            except Exception:
                parsed = None
            if parsed is not None and not pd.isna(parsed):
                return parsed.value // 10**9
            try:
                return float(stripped)
            except ValueError:
                return fallback
        return fallback

    def _random_neq(self, l, r, s):
        """
        生成一个不在序列s中的随机整数, 用于训练时的负采样

        Args:
            l: 随机整数的最小值
            r: 随机整数的最大值
            s: 序列

        Returns:
            t: 不在序列s中的随机整数
        """
        t = np.random.randint(l, r)
        while t in s or str(t) not in self.item_feat_dict:
            t = np.random.randint(l, r)
        return t

    def __getitem__(self, uid):
        """
        获取单个用户的数据，并进行padding处理，生成模型需要的数据格式

        Args:
            uid: 用户ID(reid)

        Returns:
            seq: 用户序列ID
            pos: 正样本ID（即下一个真实访问的item）
            neg: 负样本ID
            token_type: 用户序列类型，1表示item，2表示user
            next_token_type: 下一个token类型，1表示item，2表示user
            seq_feat: 用户序列特征，每个元素为字典，key为特征ID，value为特征值
            pos_feat: 正样本特征，每个元素为字典，key为特征ID，value为特征值
            neg_feat: 负样本特征，每个元素为字典，key为特征ID，value为特征值
        """
        user_sequence = self._load_user_data(uid)  # 动态加载用户数据

        ext_user_sequence = []
        for record_tuple in user_sequence:
            u, i, user_feat, item_feat, action_type, _ = record_tuple
            if u and user_feat:
                ext_user_sequence.insert(0, (u, user_feat, 2, action_type))
            if i and item_feat:
                ext_user_sequence.append((i, item_feat, 1, action_type))

        seq = np.zeros([self.maxlen + 1], dtype=np.int32)
        pos = np.zeros([self.maxlen + 1], dtype=np.int32)
        neg = np.zeros([self.maxlen + 1], dtype=np.int32)
        token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        next_token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        next_action_type = np.zeros([self.maxlen + 1], dtype=np.int32)

        seq_feat = np.empty([self.maxlen + 1], dtype=object)
        pos_feat = np.empty([self.maxlen + 1], dtype=object)
        neg_feat = np.empty([self.maxlen + 1], dtype=object)

        nxt = ext_user_sequence[-1]
        idx = self.maxlen

        ts = set()
        for record_tuple in ext_user_sequence:
            if record_tuple[2] == 1 and record_tuple[0]:
                ts.add(record_tuple[0])

        # left-padding, 从后往前遍历，将用户序列填充到maxlen+1的长度
        for record_tuple in reversed(ext_user_sequence[:-1]):
            i, feat, type_, act_type = record_tuple
            next_i, next_feat, next_type, next_act_type = nxt
            feat = self.fill_missing_feat(feat, i)
            next_feat = self.fill_missing_feat(next_feat, next_i)
            seq[idx] = i
            token_type[idx] = type_
            next_token_type[idx] = next_type
            if next_act_type is not None:
                next_action_type[idx] = next_act_type
            seq_feat[idx] = feat
            if next_type == 1 and next_i != 0:
                pos[idx] = next_i
                pos_feat[idx] = next_feat
                neg_id = self._random_neq(1, self.itemnum + 1, ts)
                neg[idx] = neg_id
                neg_feat[idx] = self.fill_missing_feat(self.item_feat_dict[str(neg_id)], neg_id)
            nxt = record_tuple
            idx -= 1
            if idx == -1:
                break

        seq_feat = np.where(seq_feat == None, self.feature_default_value, seq_feat)
        pos_feat = np.where(pos_feat == None, self.feature_default_value, pos_feat)
        neg_feat = np.where(neg_feat == None, self.feature_default_value, neg_feat)

        return seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat

    def __len__(self):
        """
        返回数据集长度，即用户数量

        Returns:
            usernum: 用户数量
        """
        return len(self.seq_offsets)

    def _init_feat_info(self):
        """
        初始化特征信息, 包括特征缺省值和特征类型

        Returns:
            feat_default_value: 特征缺省值，每个元素为字典，key为特征ID，value为特征缺省值
            feat_types: 特征类型，key为特征类型名称，value为包含的特征ID列表
        """
        feat_default_value = {}
        feat_statistics = {}
        feat_types = {}
        feat_types['user_sparse'] = ['103', '104', '105', '109']
        feat_types['item_sparse'] = [
            '100',
            '117',
            '111',
            '118',
            '101',
            '102',
            '119',
            '120',
            '114',
            '112',
            '121',
            '115',
            '122',
            '116',
        ]
        feat_types['item_array'] = []
        feat_types['user_array'] = ['106', '107', '108', '110']
        feat_types['item_emb'] = self.mm_emb_ids
        feat_types['user_continual'] = []
        feat_types['item_continual'] = []

        for feat_id in feat_types['user_sparse']:
            feat_default_value[feat_id] = 0
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['item_sparse']:
            feat_default_value[feat_id] = 0
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['item_array']:
            feat_default_value[feat_id] = [0]
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['user_array']:
            feat_default_value[feat_id] = [0]
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['user_continual']:
            feat_default_value[feat_id] = 0
        for feat_id in feat_types['item_continual']:
            feat_default_value[feat_id] = 0
        for feat_id in feat_types['item_emb']:
            feat_default_value[feat_id] = np.zeros(
                list(self.mm_emb_dict[feat_id].values())[0].shape[0], dtype=np.float32
            )

        return feat_default_value, feat_types, feat_statistics

    def fill_missing_feat(self, feat, item_id):
        """
        对于原始数据中缺失的特征进行填充缺省值

        Args:
            feat: 特征字典
            item_id: 物品ID

        Returns:
            filled_feat: 填充后的特征字典
        """
        if feat == None:
            feat = {}
        filled_feat = {}
        for k in feat.keys():
            filled_feat[k] = feat[k]

        all_feat_ids = []
        for feat_type in self.feature_types.values():
            all_feat_ids.extend(feat_type)
        missing_fields = set(all_feat_ids) - set(feat.keys())
        for feat_id in missing_fields:
            filled_feat[feat_id] = self.feature_default_value[feat_id]
        for feat_id in self.feature_types['item_emb']:
            if item_id != 0 and self.indexer_i_rev[item_id] in self.mm_emb_dict[feat_id]:
                if type(self.mm_emb_dict[feat_id][self.indexer_i_rev[item_id]]) == np.ndarray:
                    filled_feat[feat_id] = self.mm_emb_dict[feat_id][self.indexer_i_rev[item_id]]

        return filled_feat

    @staticmethod
    def collate_fn(batch):
        """
        Args:
            batch: 多个__getitem__返回的数据

        Returns:
            seq: 用户序列ID, torch.Tensor形式
            pos: 正样本ID, torch.Tensor形式
            neg: 负样本ID, torch.Tensor形式
            token_type: 用户序列类型, torch.Tensor形式
            next_token_type: 下一个token类型, torch.Tensor形式
            seq_feat: 用户序列特征, list形式
            pos_feat: 正样本特征, list形式
            neg_feat: 负样本特征, list形式
        """
        seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = zip(*batch)
        seq = torch.from_numpy(np.array(seq))
        pos = torch.from_numpy(np.array(pos))
        neg = torch.from_numpy(np.array(neg))
        token_type = torch.from_numpy(np.array(token_type))
        next_token_type = torch.from_numpy(np.array(next_token_type))
        next_action_type = torch.from_numpy(np.array(next_action_type))
        seq_feat = list(seq_feat)
        pos_feat = list(pos_feat)
        neg_feat = list(neg_feat)
        return seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat


class MyTestDataset(MyDataset):
    """
    测试数据集
    """

    def __init__(self, data_dir, args):
        super().__init__(data_dir, args)

    def _load_data_and_offsets(self):
        if self.dataset_type == 'tencent':
            self.data_file = open(self.data_dir / "predict_seq.jsonl", 'rb')
            with open(Path(self.data_dir, 'predict_seq_offsets.pkl'), 'rb') as f:
                self.seq_offsets = pickle.load(f)
        else:
            # Reuse KuaiRec loader for inference scenarios
            self._load_kuairec_dataset()

    def _process_cold_start_feat(self, feat):
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

    def __getitem__(self, uid):
        """
        获取单个用户的数据，并进行padding处理，生成模型需要的数据格式

        Args:
            uid: 用户在self.data_file中储存的行号
        Returns:
            seq: 用户序列ID
            token_type: 用户序列类型，1表示item，2表示user
            seq_feat: 用户序列特征，每个元素为字典，key为特征ID，value为特征值
            user_id: user_id eg. user_xxxxxx ,便于后面对照答案
        """
        user_sequence = self._load_user_data(uid)  # 动态加载用户数据

        ext_user_sequence = []
        for record_tuple in user_sequence:
            u, i, user_feat, item_feat, _, _ = record_tuple
            if u:
                if type(u) == str:  # 如果是字符串，说明是user_id
                    user_id = u
                else:  # 如果是int，说明是re_id
                    user_id = self.indexer_u_rev[u]
            if u and user_feat:
                if type(u) == str:
                    u = 0
                if user_feat:
                    user_feat = self._process_cold_start_feat(user_feat)
                ext_user_sequence.insert(0, (u, user_feat, 2))

            if i and item_feat:
                # 序列对于训练时没见过的item，不会直接赋0，而是保留creative_id，creative_id远大于训练时的itemnum
                if i > self.itemnum:
                    i = 0
                if item_feat:
                    item_feat = self._process_cold_start_feat(item_feat)
                ext_user_sequence.append((i, item_feat, 1))

        seq = np.zeros([self.maxlen + 1], dtype=np.int32)
        token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        seq_feat = np.empty([self.maxlen + 1], dtype=object)

        idx = self.maxlen

        ts = set()
        for record_tuple in ext_user_sequence:
            if record_tuple[2] == 1 and record_tuple[0]:
                ts.add(record_tuple[0])

        for record_tuple in reversed(ext_user_sequence[:-1]):
            i, feat, type_ = record_tuple
            feat = self.fill_missing_feat(feat, i)
            seq[idx] = i
            token_type[idx] = type_
            seq_feat[idx] = feat
            idx -= 1
            if idx == -1:
                break

        seq_feat = np.where(seq_feat == None, self.feature_default_value, seq_feat)

        return seq, token_type, seq_feat, user_id

    def __len__(self):
        """
        Returns:
            len(self.seq_offsets): 用户数量
        """
        if self.dataset_type == 'tencent':
            with open(Path(self.data_dir, 'predict_seq_offsets.pkl'), 'rb') as f:
                temp = pickle.load(f)
            return len(temp)
        return len(self.seq_offsets)

    @staticmethod
    def collate_fn(batch):
        """
        将多个__getitem__返回的数据拼接成一个batch

        Args:
            batch: 多个__getitem__返回的数据

        Returns:
            seq: 用户序列ID, torch.Tensor形式
            token_type: 用户序列类型, torch.Tensor形式
            seq_feat: 用户序列特征, list形式
            user_id: user_id, str
        """
        seq, token_type, seq_feat, user_id = zip(*batch)
        seq = torch.from_numpy(np.array(seq))
        token_type = torch.from_numpy(np.array(token_type))
        seq_feat = list(seq_feat)

        return seq, token_type, seq_feat, user_id


def save_emb(emb, save_path):
    """
    将Embedding保存为二进制文件

    Args:
        emb: 要保存的Embedding，形状为 [num_points, num_dimensions]
        save_path: 保存路径
    """
    num_points = emb.shape[0]  # 数据点数量
    num_dimensions = emb.shape[1]  # 向量的维度
    print(f'saving {save_path}')
    with open(Path(save_path), 'wb') as f:
        f.write(struct.pack('II', num_points, num_dimensions))
        emb.tofile(f)


def load_mm_emb(mm_path, feat_ids):
    """
    加载多模态特征Embedding

    Args:
        mm_path: 多模态特征Embedding路径
        feat_ids: 要加载的多模态特征ID列表

    Returns:
        mm_emb_dict: 多模态特征Embedding字典，key为特征ID，value为特征Embedding字典（key为item ID，value为Embedding）
    """
    #SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
    SHAPE_DICT = {"82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}

    mm_emb_dict = {}
    for feat_id in tqdm(feat_ids, desc='Loading mm_emb'):
        shape = SHAPE_DICT[feat_id]
        emb_dict = {}
        if feat_id != '81':
            try:
                base_path = Path(mm_path, f'emb_{feat_id}_{shape}')
                for json_file in base_path.glob('part-*'):
                    with open(json_file, 'r', encoding='utf-8') as file:
                        for line in file:
                            data_dict_origin = json.loads(line.strip())
                            insert_emb = data_dict_origin['emb']
                            if isinstance(insert_emb, list):
                                insert_emb = np.array(insert_emb, dtype=np.float32)
                            data_dict = {data_dict_origin['anonymous_cid']: insert_emb}
                            emb_dict.update(data_dict)
            except Exception as e:
                print(f"transfer error: {e}")
                
        '''
        if feat_id == '81':
            with open(Path(mm_path, f'emb_{feat_id}_{shape}.pkl'), 'rb') as f:
                emb_dict = pickle.load(f)
        '''
        mm_emb_dict[feat_id] = emb_dict
        print(f'Loaded #{feat_id} mm_emb')
    return mm_emb_dict


def _resolve_data_root(cli_args):
    script_dir = Path(__file__).resolve().parent
    if cli_args.data_path is not None:
        return Path(cli_args.data_path)

    from os import environ

    env_path = None if cli_args.no_env else environ.get("TRAIN_DATA_PATH")
    if env_path:
        return Path(env_path)

    candidate_roots = [
        Path("./data"),
        script_dir / "data",
        script_dir.parent / "data",
        Path("./dataset"),
        script_dir / "dataset",
        script_dir.parent / "dataset",
    ]
    for candidate in candidate_roots:
        if candidate.exists():
            return candidate
        if (candidate / "KuaiRec").exists():
            return candidate / "KuaiRec"
    return candidate_roots[0]


def main():
    import argparse
    import traceback
    from types import SimpleNamespace

    import sys

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    parser = argparse.ArgumentParser(description="Inspect dataset loading and derived features.")
    parser.add_argument(
        "--data-path",
        default=None,
        help="Root directory that contains the Tencent preprocessed files or KuaiRec CSVs.",
    )
    parser.add_argument("--maxlen", type=int, default=100, help="Maximum sequence length to keep per user.")
    parser.add_argument(
        "--mm-emb-id",
        nargs="*",
        default=[],
        help="Tencent only: multimedia embedding feature IDs to load (e.g. 82 83).",
    )
    parser.add_argument(
        "--sample-users",
        type=int,
        default=3,
        help="How many sample sequences to summarise after loading the dataset.",
    )
    parser.add_argument(
        "--show-features",
        action="store_true",
        help="Print the derived feature groups (sparse/array/continual) detected for the dataset.",
    )
    parser.add_argument(
        "--no-env",
        action="store_true",
        help="Ignore TRAIN_DATA_PATH from the environment and rely on --data-path only.",
    )

    cli_args = parser.parse_args()
    data_root = Path(_resolve_data_root(cli_args))
    resolved_path = data_root.resolve() if data_root.exists() else data_root
    print(f"Inspecting data root: {resolved_path}", flush=True)

    runtime_args = SimpleNamespace(maxlen=cli_args.maxlen, mm_emb_id=cli_args.mm_emb_id)

    try:
        dataset = MyDataset(str(data_root), runtime_args)
    except FileNotFoundError as err:
        print(f"\n❌ {err}")
        print("Provide the KuaiRec/Tencent files via --data-path or set TRAIN_DATA_PATH before running this helper.")
        return 1
    except Exception:
        print("\n❌ Unexpected error while loading the dataset:")
        traceback.print_exc()
        return 1

    print("Dataset initialised successfully.\n", flush=True)

    print(f"Detected dataset type: {dataset.dataset_type}")
    user_count = getattr(dataset, "usernum", len(dataset))
    item_count = getattr(dataset, "itemnum", "unknown")
    print(f"Users: {user_count}  Items: {item_count}  User sequences: {len(dataset)}")

    if cli_args.show_features and getattr(dataset, "feature_types", None):
        print("\nDerived feature groups:")
        for group, feature_ids in dataset.feature_types.items():
            print(f"  - {group}: {len(feature_ids)} feature(s)")

    if len(dataset) == 0:
        print("\nThe dataset is empty—no user sequences were loaded.")
        return 0

    print("\nSample sequences:")
    sample_total = min(cli_args.sample_users, len(dataset))
    for idx in range(sample_total):
        sample = dataset[idx]
        (
            seq,
            pos,
            neg,
            token_type,
            next_token_type,
            next_action_type,
            seq_feat,
            pos_feat,
            neg_feat,
        ) = sample

        seq = np.asarray(seq)
        pos = np.asarray(pos)

        history_len = int(np.count_nonzero(seq))
        positives = pos[pos > 0]
        next_item_id = int(positives[-1]) if positives.size else 0
        original_item = dataset.indexer_i_rev.get(next_item_id, "n/a") if next_item_id else "n/a"
        print(
            f"  • User #{idx}: history length={history_len}, next positive item id={next_item_id} (original={original_item})"
        )

    print("\nUse python train/main.py to launch full training once the dataset looks correct.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
