"""Single source of truth for package metadata.

``pyproject.toml`` reads ``__version__`` from this file through
``[tool.hatch.version]``, so bump the version here and nowhere else.

The release identifiers live here too, because they are used by more than one
module (the CLI, the Hugging Face publisher, the model/dataset card templates)
and a typo in a Hub repo id is only discovered after a failed upload.
"""

__version__ = "0.1.0"

PROJECT_NAME = "shingan"
PROJECT_DISPLAY_NAME = "Shingan"
PROJECT_TAGLINE = "Evidence-grounded financial risk modelling"
PROJECT_DESCRIPTION_ZH = "心眼：双轨（QLoRA 文本 + 梯度提升结构化信号）金融风险建模 POC"

GITHUB_ORG = "shuurai"
# The repository slug is shingan-finrisk (renamed from shingan); every GitHub link
# rendered into the HF cards comes from this constant, so it must match the remote.
GITHUB_REPO = "shingan-finrisk"
GITHUB_URL = f"https://github.com/{GITHUB_ORG}/{GITHUB_REPO}"

HF_MODEL_ID = f"{GITHUB_ORG}/shingan-qwen3-14b-finrisk"
HF_DATASET_ID = f"{GITHUB_ORG}/shingan-finrisk-labels"
HF_BENCHMARK_ID = f"{GITHUB_ORG}/shingan-bench"

#: Model id of the base LLM the text track is fine-tuned from. Kept in sync with
#: ``configs/train/qlora_qwen3_14b.yaml`` and ``templates/model_card.md``.
DEFAULT_BASE_MODEL = "Qwen/Qwen3-14B"

#: Schema version of the processed panel dataset. Written into every dataset as
#: the ``data_version`` column and quoted in the dataset card. Bump it whenever a
#: column is added, removed, renamed or its meaning changes.
DATA_SCHEMA_VERSION = "0.1.0"

LICENSE_ID = "Apache-2.0"
LICENSE_URL = "https://www.apache.org/licenses/LICENSE-2.0"
