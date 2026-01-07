import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


def build_text(row) -> str:
    """
    Turn one row of the caption/category file into a single text string.

    We try to use columns if they exist; if not, we just skip them.
    """
    parts = []

    def add_if_exists(col_name: str, prefix: str):
        if col_name in row and isinstance(row[col_name], str) and row[col_name].strip():
            parts.append(f"{prefix}: {row[col_name]}")

    add_if_exists("manual_cover_text", "cover")
    add_if_exists("caption", "caption")
    add_if_exists("topic_tag", "topics")
    add_if_exists("first_level_category_name", "category1")
    add_if_exists("second_level_category_name", "category2")

    # Fallback
    if not parts:
        return "no description"

    return " | ".join(parts)


def main():
    # Root: .../AdRec-GenAI
    root = Path(__file__).resolve().parents[2]
    data_path = root / "kuairec" / "data" / "kuairec_caption_category.csv"
    out_dir = root / "kuairec" / "embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "item_llm_embeddings.npy"

    print(f"Loading caption/category data from: {data_path}")

    # Robust CSV loading
    try:
        df = pd.read_csv(
            data_path,
            sep=None,            # let pandas guess delimiter
            engine="python",     # more flexible parser
            on_bad_lines="skip", # skip broken lines
        )
    except TypeError:
        # older pandas may not support on_bad_lines
        df = pd.read_csv(
            data_path,
            sep=None,
            engine="python",
        )

    # Strip whitespace from column names (your file had spaces)
    df.columns = [c.strip() for c in df.columns]
    print("Columns detected:", df.columns.tolist())
    print("Loaded rows (raw):", len(df))

    # Figure out which column is the item/video id
    if "video_id" in df.columns:
        vid_col = "video_id"
    elif "item_id" in df.columns:
        vid_col = "item_id"
    elif "id" in df.columns:
        vid_col = "id"
    else:
        raise KeyError(
            f"Could not find a video/item id column in: {df.columns.tolist()}"
        )

    # Coerce video_id to numeric and drop non-numeric rows
    df[vid_col] = pd.to_numeric(df[vid_col], errors="coerce")
    df = df.dropna(subset=[vid_col])
    df[vid_col] = df[vid_col].astype(int)

    print("Loaded rows (after cleaning ids):", len(df))

    video_ids = df[vid_col].values
    max_id = int(df[vid_col].max())
    print(f"Using id column: {vid_col}, max id: {max_id}")

    # Build texts
    print("Building text descriptions...")
    texts = [build_text(row) for _, row in df.iterrows()]

    # Load local embedding model
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    print(f"Loading SentenceTransformer model: {model_name}")
    model = SentenceTransformer(model_name)

    batch_size = 64
    all_embeddings = []

    print("Encoding texts into embeddings...")
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i : i + batch_size]
        emb = model.encode(
            batch_texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        all_embeddings.append(emb)

    all_embeddings = np.vstack(all_embeddings)  # [num_rows, emb_dim]
    emb_dim = all_embeddings.shape[1]

    # Build [n_items, emb_dim] array, index = item_id
    n_items = max_id + 1
    item_emb = np.zeros((n_items, emb_dim), dtype=np.float32)

    for idx, vid in enumerate(video_ids):
        vid = int(vid)
        if 0 <= vid < n_items:
            item_emb[vid] = all_embeddings[idx]

    print(f"Final item_emb shape: {item_emb.shape}")
    np.save(out_path, item_emb)
    print(f"Saved item LLM embeddings to: {out_path}")


if __name__ == "__main__":
    main()
