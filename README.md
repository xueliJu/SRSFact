# SRSFact

The official implementation of the under-review work "Fact2Fiction: Targeted Poisoning Attack to Agentic Fact-checking System".

## Dataset

Please place the required datasets in the `src/data/` folder before running the code.

For AVeriTeC experiments, prepare the AVeriTeC files under `src/data/AVeriTeC/` and build the knowledge bases:

```bash
cd src
python scripts/averitec/build.py
```

Due to double-blind review requirements, large processed datasets, cached knowledge bases, generated attack evidence, and experiment outputs are not included in this repository. The full code and datasets will be made available after the paper is accepted.

## Installation

```bash
pip install -r requirements.txt
```

Please configure API keys before running experiments:

```bash
cd src
python scripts/setup.py
```

## Usage

All commands are run from `src/`:

```bash
cd src
```

### Base Fact-Checking

```bash
# InFact/DEFAME baseline 
python -m scripts.averitec.evaluate --procedure_variant infact --llm gemini_25_flash --variant dev --n-workers 4

# SRSFact
python -m scripts.averitec.evaluate --procedure_variant srsfact --srs true --llm gemini_25_flash --variant dev --n-workers 4
```

Common `--procedure_variant` choices: `infact`, `srsfact`, `summary`, `no_qa`, `naive`, `no_interpretation`, `no_evidence`, `no_query_generation`. (`infact` for InFact and `summary` for DEFAME)

### Attack Experiments

```bash
# FPlus
python -m attack.main --attack-type fplus --victim infact --poison-rate 0.01 --gpu-ids 0 --n-processes 4 --fact-checker-model gemini_25_flash --attacker-model gemini_25_flash 

```

Common `--attack-type` choices: `naive`, `disinformation`, `poisoned_rag`, `prompt_injection`, `fact2fiction`, `fplus`.
Common `--victim` choices: `infact`, `SRSFact`, `defame`, `no_qa`.

### Evaluation

```bash
# Semantic drift evaluation
python -m eval.stealth_eval --attack-dir {attack_result_path} --device cuda:0

# Hallucination comparison between SRSFact and InFact
python -m hallucination_eval.compare_hallucination \
  --srs-dir {srs_attack_result_dir} \
  --infact-dir {infact_attack_result_dir} \
  --output {hallucination_output_dir}
```

Key parameters:


- `--llm`: model shorthand in `config/available_models.csv`.
- `--poison-rate`: fake-evidence ratio, e.g., `0.01` or `0.08`.
- `--gpu-ids`: GPU IDs used by attack workers, e.g., `0` or `0 1`.
- `--n-processes` / `--n-workers`: parallel worker count.
- `--srs-dir`: SRSFact attack result directory.
- `--infact-dir`: InFact attack result directory.
- `--output`: output JSON path for hallucination comparison.

Fact-checking outputs are saved under `src/out/averitec/` by default.

Attack outputs are saved under `src/attack/attack_results/` by default.

Evaluation outputs are saved under `src/eval/out/` by default.

Hallucination comparison outputs are saved to the path specified by `--output`, for example `src/hallucination/out/result.json`.

## Notice

This work is currently under review. The code is not licensed for any use, reproduction, or distribution at this time. All rights are reserved by the authors and their affiliated institutions.

Full code and datasets will be made available upon acceptance. For any inquiries, please contact the authors.


## Reference

Our implementation is based on the following open-source works:

[1] He H, Li Y, Zhu B B, et al. Fact2Fiction: Targeted poisoning attack to agentic fact-checking system[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2026, 40(37): 30943-30950.

[2] Rothermel M, Braun T, Rohrbach M, et al. InFact: A strong baseline for automated fact-checking[C]//Proceedings of the Seventh Fact Extraction and VERification Workshop (FEVER). 2024: 108-112.

[3] Braun T, Rothermel M, Rohrbach M, et al. Defame: Dynamic evidence-based fact-checking with multimodal experts[J]. arXiv preprint arXiv:2412.10510, 2024.

[4] Liu F, Abuadbba S, Moore K, et al. Adversarial attacks against automated fact-checking: A survey[C]//Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing. 2025: 22979-23001.

[5] Zou W, Geng R, Wang B, et al. {PoisonedRAG}: Knowledge corruption attacks to {Retrieval-Augmented} generation of large language models[C]//34th USENIX Security Symposium (USENIX Security 25). 2025: 3827-3844.

[6] Du Y, Bosselut A, Manning C D. Synthetic disinformation attacks on automated fact verification systems[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2022, 36(10): 10581-10589.

[7] Liu Y, Jia Y, Geng R, et al. Formalizing and benchmarking prompt injection attacks and defenses[C]//33rd USENIX Security Symposium (USENIX Security 24). 2024: 1831-1847.

[8] Reimers N, Gurevych I. Sentence-bert: Sentence embeddings using siamese bert-networks[C]//Proceedings of the 2019 conference on empirical methods in natural language processing and the 9th international joint conference on natural language processing (EMNLP-IJCNLP). 2019: 3982-3992.