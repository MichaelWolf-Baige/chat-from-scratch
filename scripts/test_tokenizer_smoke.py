"""Tokenizer encode-decode roundtrip tests — Phase 1 smoke test."""
from tokenizers import Tokenizer

tok = Tokenizer.from_file("tokenizers/phase1_synthetic/tokenizer.json")
print(f"Vocab size: {tok.get_vocab_size()}")

# Test 1: English roundtrip
en = "The model is efficient."
decoded = tok.decode(tok.encode(en).ids)
ok = "OK" if decoded == en else "FAIL"
print(f"EN: {en!r} -> {decoded!r}  {ok}")

# Test 2: Chinese roundtrip (will fail with small vocab, expected)
zh = "test"
enc = tok.encode(zh)
decoded_zh = tok.decode(enc.ids)
print(f"ZH: {zh!r} -> ids={enc.ids} -> {decoded_zh!r}")

# Test 3: Mixed
mixed = "HelloWorld"
enc_m = tok.encode(mixed)
decoded_m = tok.decode(enc_m.ids)
print(f"Mixed: {mixed!r} -> ids={enc_m.ids} -> {decoded_m!r}")

# Test 4: Special token isolation
ids = tok.encode("hello").ids
s_id = tok.token_to_id("<s>") or 1
print(f"BOS id={s_id} in normal encode: {s_id in ids}")

# Test 5: UNK rate
text = "自然语言处理是人工智能的一个重要分支"
ids_zh = tok.encode(text).ids
unk_id = tok.token_to_id("<unk>") or 3
unk_count = ids_zh.count(unk_id)
rate = unk_count / max(len(ids_zh), 1)
print(f"UNK rate on Chinese: {unk_count}/{len(ids_zh)} = {rate:.1%}")
print(f"Note: High UNK rate expected with only {tok.get_vocab_size()} vocab on tiny synthetic data")
print()
print("Tokenizer smoke test complete")
