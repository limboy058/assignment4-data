"""Train a fastText classifier to distinguish Wiki vs CC text.

Positive samples (label ``wiki``) come from the content of URLs listed in
``ENWIKI_URL_LIST``; negative samples (label ``cc``) come from CommonCrawl WET
data under ``CC_PATH`` (see ``cs336_data.utils.CC_PATH``).

Run this file directly to train a model and save it to ``MODEL_OUTPUT_PATH``.

Example:

	python -m cs336_data.classifier

You may want to adjust the paths and sample sizes below to match your
environment and available compute.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable
import tqdm

import fasttext
import requests
from fastwarc.warc import ArchiveIterator, WarcRecordType
from resiliparse.parse.encoding import bytes_to_str

from cs336_data.utils import (
	CC_PATH,
	extract_text_from_html_bytes,
	gopher_quality_filter,
	identify_language,
)

# wget --timeout=5 --dns-timeout=5 --connect-timeout=5 --read-timeout=5 --tries=1 --warc-file=subsampled_positive_urls -i subsampled_positive_urls.txt -O /dev/null -o wget.log

# ---- Paths & basic config -------------------------------------------------

# Path to the file containing one English Wikipedia URL per line.
# （仅用于你用 shuf/head 抽样，然后用 wget 生成 WARC。）
ENWIKI_URL_LIST = Path("/mnt/mnt/zjdx/tyhcyq/classifier/enwiki-20240420-extracted_urls.txt")

# WARC file produced by wget from the subsampled Wiki URLs.
WIKI_WARC_PATH = Path("/mnt/mnt/zjdx/tyhcyq/classifier/subsampled_positive_urls_1.warc.gz")

# Where to write the fastText training file and final model.
TRAIN_OUTPUT_PATH = Path("/mnt/mnt/zjdx/tyhcyq/classifier/wiki_vs_cc_fasttext_train_1.txt")
MODEL_OUTPUT_PATH = Path("/mnt/mnt/zjdx/tyhcyq/classifier/wiki_vs_cc_fasttext_1.bin")
# Sampling limits to avoid overloading the machine.
# MAX_WIKI_SAMPLES = 50_000
# MAX_CC_SAMPLES = 50_000
MAX_WIKI_SAMPLES = 50000
MAX_CC_SAMPLES = 30000

# ---- Helper functions -----------------------------------------------------

def _is_english(text: str, threshold: float = 0.6) -> bool:
	"""Return True iff text is confidently English according to our lang ID."""

	lang, score = identify_language(text)
	return lang == "en" and score >= threshold


def _clean_and_filter(text: str) -> str | None:
	"""Apply basic cleaning + language & quality filters.

	Returns cleaned text if it passes all filters, else ``None``.
	"""

	if not text:
		return None
	text = text.strip()
	if not text:
		return None

	if not _is_english(text):
		return None

	# if not gopher_quality_filter(text):
	# 	return None

	return text


def _iter_wiki_texts(max_samples: int) -> Iterable[str]:
	"""Yield cleaned Wiki texts from the subsampled Wiki WARC file.

	Assumes you've already created ``subsampled_positive_urls.warc.gz`` via
	wget from a subsampled URL list.
	"""

	if not WIKI_WARC_PATH.is_file():
		raise FileNotFoundError(f"Wiki WARC file not found: {WIKI_WARC_PATH}")

	num = 0
	with WIKI_WARC_PATH.open("rb") as stream:
		print(len(list(ArchiveIterator(stream, record_types=WarcRecordType.response))))
	with WIKI_WARC_PATH.open("rb") as stream:
		for record in ArchiveIterator(stream, record_types=WarcRecordType.response):
			if num >= max_samples:
				break
			try:
				payload = record.reader.read()
				text = extract_text_from_html_bytes(payload)
				text = _clean_and_filter(text)
				if text is None:
					continue
				num += 1
				yield text
			except Exception as e:
				print(f"Error processing record: {e}")
				continue


def _iter_cc_texts(max_samples: int) -> Iterable[str]:
	"""Yield cleaned CC texts from WET/WARC files under CC_PATH."""

	root = Path(CC_PATH)
	if not root.is_dir():
		raise FileNotFoundError(f"CC_PATH does not exist or is not a directory: {root}")
	import random
	num = 0
	entries = list(sorted(root.iterdir()))
	random.shuffle(entries)
	while num < max_samples:
		for entry in entries:
			if num >= max_samples:
				break
			if not entry.name.endswith((".warc.gz", ".wet.gz")):
				continue

			try:
				with open(entry, 'rb') as stream:
					count = sum(1 for _ in ArchiveIterator(stream))
					selected = set(random.sample(range(count), 100))
				with entry.open("rb") as stream:
					for i, record in enumerate(ArchiveIterator(stream)):
						if num >= max_samples:
							break
						try:
							if i not in selected:
								continue
							payload = record.reader.read()
							# text = extract_text_from_html_bytes(payload)
							if payload is None:
								continue
							text = bytes_to_str(payload)
							text = _clean_and_filter(text)
							if text is None:
								continue
							num += 1
							yield text
						except Exception as e:
							print(f"Error processing record: {e}")
							continue
			except Exception as e:
				print(f"Error opening CC file {entry}: {e}")
				continue


def build_fasttext_dataset(
	train_path: Path,
	wiki_limit: int = MAX_WIKI_SAMPLES,
	cc_limit: int = MAX_CC_SAMPLES,
) -> None:
	"""Construct a fastText supervised training file.

	Each line has the form ``__label__wiki <text>`` or ``__label__cc <text>``.
	"""

	train_path.parent.mkdir(parents=True, exist_ok=True)

	print(wiki_limit, cc_limit)

	with train_path.open("w") as f:
		# Positive (Wiki) samples.
		for text in _iter_wiki_texts(wiki_limit):
			line = text.replace("\n", " ")
			f.write("__label__wiki " + line + "\n")

		# Negative (CC) samples.
		for text in _iter_cc_texts(cc_limit):
			line = text.replace("\n", " ")
			f.write("__label__cc " + line + "\n")


def train_fasttext_model(train_path: Path, model_path: Path) -> fasttext.FastText:
	"""Train a fastText supervised model and save it to ``model_path``."""

	model_path.parent.mkdir(parents=True, exist_ok=True)

	model = fasttext.train_supervised(
		input=str(train_path),
		lr=0.1,
		epoch=5,
		wordNgrams=2,
		dim=256,
		loss="softmax",
	)
	model.save_model(str(model_path))
	return model


def main() -> None:
	print(f"Building fastText dataset at {TRAIN_OUTPUT_PATH} ...")
	build_fasttext_dataset(TRAIN_OUTPUT_PATH, MAX_WIKI_SAMPLES, MAX_CC_SAMPLES)

	print(f"Training fastText model; output = {MODEL_OUTPUT_PATH} ...")
	model = train_fasttext_model(TRAIN_OUTPUT_PATH, MODEL_OUTPUT_PATH)

	# 简单打印一下在训练集上的性能（主要用于 sanity check）。
	n_examples, precision, recall = model.test(str(TRAIN_OUTPUT_PATH))
	print(f"Training data size: {n_examples}")
	print(f"P@1 = {precision:.4f}, R@1 = {recall:.4f}")


if __name__ == "__main__":
	main()

