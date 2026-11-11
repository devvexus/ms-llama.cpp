# MS (Mad Shift) quantization

MS (Mad Shift) is a GGUF weight format for llama.cpp created by **Matthew Priester**
([GitHub](https://github.com/devvexus), [ORCID 0009-0000-0032-5292](https://orcid.org/0009-0000-0032-5292)). At the same file
size, MS files stay much closer to the original model than standard K-/I-quants and unsloth's dynamic quants: on Qwen3.5-2B,
4-9x lower KL divergence to the BF16 model than the unsloth file of the same or larger size.

Models: [huggingface.co/devvexus](https://huggingface.co/devvexus) ("MS Quants" collection).

## Format

- **Types:** `MS_C1` ... `MS_C6` (k = 1..6 bits per weight; type ids 43..48 on this fork's master), file type 42.
- **Block:** 512 weights along a row: fp16 scale `d`, 16-bit field holding the 14-bit start state `s0`, then 511 k-bit symbols
  (MSB first, k zero pad bits): k + 0.0625 bits per weight.
- **Decode** (QTIP's bitshift trellis with the computed "1MAD" code): for each weight, `s = ((s << k) | symbol) & 0x3FFF`
  (the first weight uses `s0`); `x = 34038481 * s + 76625530 (mod 2^32)`; `w = (byte0(x) + byte1(x) + byte2(x) + byte3(x) - 510) * d`.
  Every weight's state is the 14-bit window of the code stream that ends at its symbol, so weights decode independently.
- **Rotation:** each MS tensor stores the codes of `W R`, where `R` is a random Hadamard rotation given by `prism.hadamard.*`
  metadata (block sizes, sign vectors); the graph rotates the input instead (`(W R)(R^T x) = W x`).
- **Split tables / heads:** an embedding table or output head may be stored as two tensors (`token_embd` + `token_embd_lo`,
  `output` + `output_lo`): the 65,536 most frequent tokens at a higher k, the rest lower, with the stored-row order in
  `msffn.emb_split_order`; logits come out in vocabulary order.
- **Attribution keys** in released files: `general.author`, `general.url`, `general.description`.

## Run

```bash
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release   # or without -DGGML_CUDA=ON for CPU only
cmake --build build --config Release -j
./build/bin/llama-cli -m <MS file>.gguf -ngl 99
```

## Credit

MS's trellis code builds on QTIP: A. Tseng, Q. Sun, D. Hou, C. De Sa, *QTIP: Quantization with Trellises and Incoherence
Processing*, NeurIPS 2024, [arXiv:2406.11235](https://arxiv.org/abs/2406.11235). The MS format, its llama.cpp integration and
the MS files are by Matthew Priester; see `NOTICE`.
