# Raw inputs of the literature search

- `s2_cites_*.json`: the Semantic Scholar API responses of the citation crawl (`../citation_crawl.md`, fetched by
  `../scripts/crawl.sh`).
- `openalex_*.json`: the OpenAlex responses that could not serve as a cross-check, since OpenAlex lists no citations of these
  papers (`../citation_crawl.md`).
- `DeepSeek-*`: DeepSeek's reference inference code and model configurations, downloaded on 2026-09-26 from
  `huggingface.co/deepseek-ai/{DeepSeek-V4.1-Flash, DeepSeek-V4-Flash, DeepSeek-V4-Pro}` (`inference/` and `config.json`) to check
  how mHC is implemented (`../core_papers_spec.md`). They are copyright DeepSeek and distributed under its MIT license,
  reproduced in `LICENSE-DeepSeek`.
