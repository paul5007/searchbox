"""FSV for #19: recall@8 + throughput, int8 ONNX vs fp32, on an onnx-capable model.

v5 cannot onnx (discovery #41), so the int8 PATH is measured on all-MiniLM-L6-v2 (which ships the
prebuilt qint8 avx512_vnni graph). fp32 (treated as ground truth) = the same model in torch.
Self-consistency recall@8: does int8 reproduce fp32's own top-8 per query?
"""
import os, sys, time, zipfile, re
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""
MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# --- corpus: chunk each doc into ~700-char windows (more docs -> meaningful top-8) ---
z = zipfile.ZipFile("data/default-dataroom.zip")
chunks = []
for n in sorted(z.namelist()):
    if n.endswith("/"):
        continue
    txt = z.read(n).decode("utf-8", "ignore")
    parts = re.split(r"\n\s*\n", txt)  # paragraph chunks
    for p in parts:
        p = p.strip()
        if len(p) >= 20:
            chunks.append(p)
print(f"corpus chunks: {len(chunks)}")

import json
queries = [json.loads(l)["query"] for l in open("data/prd/benchmarks/eval/queryset.jsonl")]
print(f"queries: {len(queries)}")


def embed_all(model, texts):
    return np.asarray(model.encode(texts, normalize_embeddings=True))


def topk(qv, dv, k=8):
    sims = dv @ qv.T  # (docs, queries)
    return np.argsort(-sims, axis=0)[:k, :].T  # (queries, k)


from sentence_transformers import SentenceTransformer

# fp32 (ground truth)
print("\n=== fp32 (torch) ===")
m32 = SentenceTransformer(MODEL, device="cpu")
t0 = time.time(); D32 = embed_all(m32, chunks); t_embed_32 = time.time() - t0
Q32 = embed_all(m32, queries)
top32 = topk(Q32, D32, 8)
print(f"fp32 embed {len(chunks)} chunks in {t_embed_32*1000:.0f} ms")

# int8 onnx
print("\n=== int8 (onnx qint8_avx512_vnni) ===")
m8 = SentenceTransformer(MODEL, backend="onnx",
                         model_kwargs={"provider": "CPUExecutionProvider",
                                       "file_name": "model_qint8_avx512_vnni.onnx"})
t0 = time.time(); D8 = embed_all(m8, chunks); t_embed_8 = time.time() - t0
Q8 = embed_all(m8, queries)
top8 = topk(Q8, D8, 8)
print(f"int8 embed {len(chunks)} chunks in {t_embed_8*1000:.0f} ms")

# recall@8: overlap of int8 top-8 with fp32 top-8, per query, averaged
recalls = []
for i in range(len(queries)):
    a, b = set(top32[i].tolist()), set(top8[i].tolist())
    recalls.append(len(a & b) / 8.0)
recall8 = float(np.mean(recalls))

print("\n=== RESULT (SoT) ===")
print(f"recall@8 int8-vs-fp32: {recall8:.4f}  (gate: >= 0.98 -> {'PASS' if recall8>=0.98 else 'FAIL'})")
print(f"throughput: fp32 {t_embed_32*1000:.0f}ms  int8 {t_embed_8*1000:.0f}ms  speedup {t_embed_32/t_embed_8:.2f}x")
print(f"recall@8==1.0 on {sum(1 for r in recalls if r==1.0)}/{len(queries)} queries")
