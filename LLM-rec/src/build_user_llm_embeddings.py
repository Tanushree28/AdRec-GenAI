# LLM-rec/src/build_user_llm_embeddings.py
"""
Build per-user LLM embeddings from KuaiRec interaction history.

For each user we collect their top-K watched videos (by watch_ratio),
build a natural-language interest profile from the item metadata, and
encode it with the same sentence-transformer used for item embeddings.

Output: kuairec/embeddings/user_llm_embeddings.npy  [n_users, emb_dim]
         indexed directly by user_id.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

TOP_K = 20          # videos per user to include in the profile
BATCH_SIZE = 64
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def _load_metadata(data_dir: Path) -> dict[int, str]:
    """Return {video_id: text_description} from the caption/category CSV."""
    meta_path = data_dir / "kuairec_caption_category.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    try:
        df = pd.read_csv(meta_path, sep=None, engine="python", on_bad_lines="skip")
    except TypeError:
        df = pd.read_csv(meta_path, sep=None, engine="python")

    df.columns = [c.strip() for c in df.columns]

    vid_col = next(
        (c for c in ["video_id", "item_id", "id"] if c in df.columns), None
    )
    if vid_col is None:
        raise KeyError(f"No video/item id column found in: {df.columns.tolist()}")

    df[vid_col] = pd.to_numeric(df[vid_col], errors="coerce")
    df = df.dropna(subset=[vid_col])
    df[vid_col] = df[vid_col].astype(int)

    id2text: dict[int, str] = {}
    for _, row in df.iterrows():
        parts = []
        for col, prefix in [
            ("manual_cover_text", "cover"),
            ("caption", "caption"),
            ("topic_tag", "topics"),
            ("first_level_category_name", "category1"),
            ("second_level_category_name", "category2"),
        ]:
            if col in row and isinstance(row[col], str) and row[col].strip():
                parts.append(f"{prefix}: {row[col]}")
        id2text[int(row[vid_col])] = " | ".join(parts) if parts else "no description"

    return id2text


def _build_user_profiles(
    big_matrix: pd.DataFrame, id2text: dict[int, str], top_k: int
) -> tuple[list[int], list[str]]:
    """Return parallel (user_ids, profile_texts) lists."""
    sorted_df = big_matrix.sort_values("watch_ratio", ascending=False)
    groups = sorted_df.groupby("user_id")

    user_ids: list[int] = []
    texts: list[str] = []

    for uid, grp in tqdm(groups, desc="Building user profiles"):
        top_items = grp["video_id"].head(top_k).tolist()
        descriptions = [id2text.get(int(vid), "no description") for vid in top_items]
        # Deduplicate while preserving order
        seen: set[str] = set()
        unique_desc: list[str] = []
        for d in descriptions:
            if d not in seen:
                seen.add(d)
                unique_desc.append(d)

        profile = "This user frequently watches: " + "; ".join(unique_desc[:top_k])
        user_ids.append(int(uid))
        texts.append(profile)

    return user_ids, texts


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    data_dir = root / "kuairec" / "data"
    big_matrix_path = data_dir / "big_matrix.csv"
    out_dir = root / "kuairec" / "embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "user_llm_embeddings.npy"

    print(f"Loading big_matrix from: {big_matrix_path}")
    big_matrix = pd.read_csv(
        big_matrix_path,
        dtype={"user_id": "int32", "video_id": "int32"},
        usecols=["user_id", "video_id", "watch_ratio"],
    )
    print(f"  {len(big_matrix):,} interactions, {big_matrix['user_id'].nunique():,} users")

    print("Loading item metadata...")
    id2text = _load_metadata(data_dir)
    print(f"  {len(id2text):,} items with metadata")

    user_ids, texts = _build_user_profiles(big_matrix, id2text, TOP_K)

    print(f"\nEncoding {len(texts):,} user profiles with {MODEL_NAME}...")
    encoder = SentenceTransformer(MODEL_NAME)

    all_embs: list[np.ndarray] = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Encoding"):
        batch = texts[i : i + BATCH_SIZE]
        emb = encoder.encode(batch, batch_size=BATCH_SIZE, show_progress_bar=False, convert_to_numpy=True)
        all_embs.append(emb)

    emb_matrix = np.vstack(all_embs)  # [n_users_seen, emb_dim]
    emb_dim = emb_matrix.shape[1]

    # Build dense array indexed by user_id
    max_uid = max(user_ids)
    dense = np.zeros((max_uid + 1, emb_dim), dtype=np.float32)
    for idx, uid in enumerate(user_ids):
        dense[uid] = emb_matrix[idx]

    print(f"\nFinal user_llm_embeddings shape: {dense.shape}")
    np.save(out_path, dense)
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
