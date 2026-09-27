set -e
# doi train xong (train.py ghi dong "Da luu model" o cuoi)
until grep -q "Da luu model" "d:/DSC2026/Finetune-LegalIR/logs/train.log" 2>/dev/null; do sleep 15; done
echo "=== TRAIN XONG, bat dau encode lai 487k chunk ==="
cd "d:/DSC2026/Retrieval-LegalIR"
python -u encode.py --chunks ../Chunking-LegalIR/chunks.jsonl --out emb_ft     --model ../Finetune-LegalIR/models/halong-ft
echo "=== ENCODE XONG, do ket qua ==="
cd "d:/DSC2026/Finetune-LegalIR"
python -u eval_ft.py --emb ../Retrieval-LegalIR/emb_ft --base ../Retrieval-LegalIR/emb     --train "../LegalIR - Public Test-20260806T081424Z-1-001/LegalIR - Public Test/train.json"
