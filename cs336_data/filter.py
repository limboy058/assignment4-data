"""CommonCrawl 清洗与去重流水线。

使用 `cs336_data.utils` 中已经实现的工具，对
`/mnt/mnt/zjdx/commoncrawl/commoncrawl_wet_100g/` 目录下的
WARC/WET 文件进行：

- HTML → 纯文本抽取
- 语言过滤（仅保留英文）
- Gopher 质量过滤
- Wiki-vs-CC 质量分类（只保留高质量 / wiki-like 文本）
- NSFW / toxic 文本过滤
- 邮箱 / 电话 / IP 等 PII 屏蔽
- 简单重复段落去重（基于内置 `hash` 的精确去重）

每个输入 CC 文件生成一个对应的 ``.valid`` 文件，写入到
``/mnt/mnt/zjdx/tyhcyq/valid_corpus_raw`` 目录下，文件名为
``<原文件名>.valid``，同一文件中的多个有效语段使用
``<|endoftext|>`` 分隔。

此外，会将每个通过筛选的语段单独写成一个小文件，存放在
``/mnt/mnt/zjdx/tyhcyq/valid_segments_raw`` 下，然后使用
MinHash+LSH 在磁盘级对这些 segment 文档做全局去重，去重后
的 segment 文件保存在 ``/mnt/mnt/zjdx/tyhcyq/valid_segments_dedup``，
最后将它们合并为一个 ``corpus.txt`` 输出到
``/mnt/mnt/zjdx/tyhcyq/valid_corpus`` 中。

可以直接运行本模块：

    python -m cs336_data.filter
"""

from __future__ import annotations

import os
import gc
import tqdm
import time
import hashlib
import random
import re
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Set
from concurrent.futures import ProcessPoolExecutor

from fastwarc.warc import ArchiveIterator
from resiliparse.parse.encoding import bytes_to_str
from resiliparse.extract.html2text import extract_plain_text

from cs336_data.utils import (
	CC_PATH,
	classify_nsfw,
	classify_quality,
	classify_toxic_speech,
	extract_text_from_html_bytes,
	gopher_quality_filter,
	identify_language,
	mask_emails,
	mask_ips,
	mask_phone_numbers,
)


# 输出目录：先写入 RAW 目录，再对所有 segment 做全局 MinHash+LSH 去重后合并
RAW_VALID_CORPUS_DIR = Path("/mnt/mnt/zjdx/tyhcyq/valid_corpus_raw_3")
VALID_CORPUS_DIR = Path("/mnt/mnt/zjdx/tyhcyq/valid_corpus_3")


# 最终合并后的大语料文件
FINAL_CORPUS_PATH = VALID_CORPUS_DIR / "corpus.txt"

# MinHash+LSH参数
MINHASH_NUM_HASHES = 64
MINHASH_NUM_BANDS = 8
MINHASH_NGRAMS = 5
MINHASH_JACCARD_THRESHOLD = 0.8
import hashlib
import random

def _get_ngrams(text: str, n: int) -> Set[str]:
	tokens = text.split()
	if len(tokens) < n:
		return set([' '.join(tokens)])
	return set(' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1))

def _minhash_signature(ngrams: Set[str], num_hashes: int, hash_seeds: List[int]) -> List[int]:
	sig = []
	for seed in hash_seeds:
		min_hash = None
		for ng in ngrams:
			# h = int(hashlib.md5((ng+str(seed)).encode()).hexdigest(), 16)
			h = hash((ng, seed))
			if min_hash is None or h < min_hash:
				min_hash = h
		sig.append(min_hash)
	return sig

def _lsh_bands(signature: List[int], num_bands: int) -> List[Tuple[int]]:
	rows_per_band = len(signature) // num_bands
	bands = []
	for i in range(num_bands):
		start = i * rows_per_band
		end = start + rows_per_band
		bands.append(tuple(signature[start:end]))
	return bands

def _jaccard(a: Set[str], b: Set[str]) -> float:
	if not a or not b:
		return 0.0
	return len(a & b) / len(a | b)

# 多进程 worker 数（可按需调整）
NUM_WORKERS = max(1, (os.cpu_count() // 16 or 1))

# 各种过滤阈值（可按需要调整）
EN_THRESHOLD = 0.7        # 语言识别置信度
QUALITY_THRESHOLD = 0.5   # 质量分类置信度
NSFW_THRESHOLD = 0.5      # NSFW 置信度
TOXIC_THRESHOLD = 0.5     # toxic 置信度


def _is_english(text: str, threshold: float = EN_THRESHOLD) -> bool:
	"""使用 fastText 语言模型判断是否为高置信度英文。"""
	lang, score = identify_language(text)
	return lang == "en" and score >= threshold


def _passes_quality(text: str, threshold: float = QUALITY_THRESHOLD) -> bool:
	"""使用 Wiki-vs-CC 模型做质量筛选，仅保留 wiki-like 文本。"""
	# return True
	label, score = classify_quality(text)
	return label == "wiki" or label == "cc" and score < 0.75


def _is_safe_text(text: str) -> bool:
	"""用 NSFW / toxic 模型过滤不安全内容。"""
	nsfw_label, nsfw_score = classify_nsfw(text)
	if nsfw_label == "nsfw" and nsfw_score >= NSFW_THRESHOLD:
		return False

	toxic_label, toxic_score = classify_toxic_speech(text)
	if toxic_label == "toxic" and toxic_score >= TOXIC_THRESHOLD:
		return False

	return True

# 识别“独占一行的日期”用的正则
_MONTHS_FULL = r"(January|February|March|April|May|June|July|August|September|October|November|December)"
_MONTHS_ABBR = r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
_DOW = r"(Mon|Tue|Wed|Thu|Fri|Sat|Sun|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)"

_DATE_PATTERNS = [
    re.compile(r"^\s*\d{4}[-/.](0?[1-9]|1[0-2])[-/.](0?[1-9]|[12]\d|3[01])\s*$", re.IGNORECASE),  # YYYY-MM-DD
    re.compile(r"^\s*(0?[1-9]|1[0-2])[-/.](0?[1-9]|[12]\d|3[01])[-/.]\d{4}\s*$", re.IGNORECASE),  # MM/DD/YYYY
    re.compile(rf"^\s*{_MONTHS_FULL}\s+\d{{1,2}},\s*\d{{4}}\s*$", re.IGNORECASE),                 # Month DD, YYYY
    re.compile(rf"^\s*\d{{1,2}}\s+{_MONTHS_FULL}\s+\d{{4}}\s*$", re.IGNORECASE),                  # DD Month YYYY
    re.compile(rf"^\s*{_DOW},?\s+\d{{1,2}}\s+{_MONTHS_ABBR}\s+\d{{4}}\s*$", re.IGNORECASE),       # DOW DD Mon YYYY
    # 可选：中文日期（通常会被英文过滤挡住，这里补充以防万一）
    re.compile(r"^\s*\d{4}年(0?[1-9]|1[0-2])月(0?[1-9]|[12]\d|3[01])日?\s*$"),
]

def _is_date_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    return any(p.fullmatch(s) for p in _DATE_PATTERNS)

def _is_short_noisy_line(line: str, min_words: int = 4) -> bool:
    """若一行英文单词数少于 min_words，则视为噪声行。
    豁免：Markdown 标题/列表项/引用/表格行/代码围栏起止。
    """
    s = line.strip()
    if not s:
        return False
    # 豁免常见结构
    if s.startswith(("```", "#", ">", "|")):
        return False
    if re.match(r"^(\s*[-*•])\s+", s):  # 无序列表
        return False
    if re.match(r"^\s*\d+\.\s+", s):    # 有序列表
        return False
    # 统计英文单词数（仅字母与可选撇号）
    words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", s)
    return len(words) < min_words

# 常见样板/噪声行（Cookie、隐私、条款、订阅、版权等）
_BOILERPLATE_LINE_PATTERNS = [
    re.compile(r"\bwe use cookies\b", re.I),
    re.compile(r"\baccept( all)? cookies\b", re.I),
    re.compile(r"\bprivacy policy\b", re.I),
    re.compile(r"\bterms (of service|and conditions)\b", re.I),
    re.compile(r"\bsubscribe (now|to our newsletter)\b", re.I),
    re.compile(r"©\s?\d{4}", re.I),
    re.compile(r"\ball rights reserved\b", re.I),
    re.compile(r"\bclick here\b", re.I),
    re.compile(r"\bcookie (policy|settings)\b", re.I),
]

def _is_boilerplate_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    return any(p.search(s) for p in _BOILERPLATE_LINE_PATTERNS)

def _is_overlong_line(line: str, max_chars: int = 30000) -> bool:
	"""若一行字符数超过 max_chars，则视为噪声行。"""
	return len(line.split()) > max_chars

def _clean_segment(raw_text: str) -> tuple[str | None, int]:
	"""对单个文档/段落做完整清洗与过滤。

	流程：strip → 语言过滤 → Gopher 质量过滤 → 质量模型过滤 →
	NSFW/toxic 过滤 → PII 屏蔽。如果任一步不通过，返回 ``None``。
	"""

	if not raw_text:
		return None, 6

	text = raw_text.strip()
	if not text:
		return None, 5

	# 语言过滤
	if not _is_english(text):
		# print("[filter] Non-English text filtered out.")
		return None,0

	# Gopher 质量过滤
	if not gopher_quality_filter(text):
		# print("[filter] Low-quality text filtered out by Gopher filter.")
		return None,1

	# Wiki-vs-CC 质量过滤
	if not _passes_quality(text):
		# print("[filter] Low-quality text filtered out by quality classifier.")
		return None,2

	# NSFW / toxic 过滤
	if not _is_safe_text(text):
		# print("[filter] Unsafe text filtered out by NSFW/toxic classifier.")
		return None,3

	# PII 屏蔽
	text, _ = mask_emails(text)
	text, _ = mask_phone_numbers(text)
	text, _ = mask_ips(text)



	# 文档内行级精确去重：同一段落内如果有重复行，只保留第一次出现
	# 以及去除独占一行的日期行
	# 以及去除过短的噪声行
	# 以及去除常见样板/噪声行
	lines = text.splitlines()
	text = None
	seen_line_hashes: set[int] = set()
	unique_lines: list[str] = []
	for line in lines:
		if _is_date_line(line):
			continue
		if _is_short_noisy_line(line):
			continue
		if _is_boilerplate_line(line):
			continue
		if _is_overlong_line(line):
			continue
		line_hash = hash(line)
		if line_hash in seen_line_hashes:
			continue
		seen_line_hashes.add(line_hash)
		unique_lines.append(line)

	text_dedup = "\n".join(unique_lines)
	del lines
	del seen_line_hashes
	del unique_lines
	return text_dedup.strip() or None,4


def _iter_cc_records(cc_file: Path) -> Iterable[str]:
	"""从单个 CC WARC/WET 文件中迭代抽取的纯文本。"""
	with cc_file.open("rb") as stream:
		# print(f"[filter] Reading CC file: {cc_file.name}")
		# 对 WET 文件，记录通常已经是纯文本；我们直接读取 payload 并按编码解码。
		for record in ArchiveIterator(stream):
			try:
				# 不同 fastwarc 版本的 WarcRecord 提供 reader 接口读取内容
				payload = record.reader.read()
				if not payload:
					continue
				text = bytes_to_str(payload)
				del payload
				if not text:
					continue
				yield text
			except Exception as e:
				# 单条记录出错则跳过
				print(f"[filter] Error extracting text from record; skipping. Error: {e}")
				continue


def filter_cc_file(idx:int, cc_file: Path, output_dir: Path = RAW_VALID_CORPUS_DIR) -> None:
	"""对单个 CC 文件进行清洗和去重，并写出 valid 文件。

	- 输入：``cc_file`` 是一个 .warc.gz 或 .wet.gz 文件。
	- 输出：在 ``output_dir`` 下生成 ``<cc_file.name>.valid``，
	  其中包含多个通过筛选的段落，彼此之间由 ``<|endoftext|>`` 分隔。
	- 去重：在同一个 CC 文件内部做基于内置 ``hash`` 的精确去重，
	  避免重复段落重复写入。
	"""

	output_dir.mkdir(parents=True, exist_ok=True)
	out_path = output_dir / f"{cc_file.name}.valid"
	out_path.unlink(missing_ok=True)

	# print(f"[filter] Processing {cc_file.name} → {out_path.name}")

	seen_hashes: set[int] = set()
	skip_typ = {i:0 for i in range(10)}  # 记录每种过滤类型跳过的数量
	segment_index = 0
	try:
		start = time.time()
		for raw_text in _iter_cc_records(cc_file):
			# print(f"[filter] Extracted raw text segment. {raw_text[:60]!r}...")
			clean, typ = _clean_segment(raw_text)
			if clean is None:
				skip_typ[typ] += 1
				continue

			# 文件内部精确去重
			h = hash(clean)
			if h in seen_hashes:
				continue
			seen_hashes.add(h)

			# 写入当前 CC 对应的 .valid 文件，方便按源文件检查
			mode = "a" if out_path.exists() else "w"
			with out_path.open(mode) as f:
				if mode == "a":
					f.write("\n<|endoftext|>\n")
				f.write(clean)

			segment_index += 1

		if segment_index == 0:
			# 没有保留任何 segment，则删除空的 .valid
			if out_path.exists():
				out_path.unlink()
		end = time.time()
		print(f"[filter] {idx} Kept {segment_index} segments from {cc_file.name} in {end - start:.2f} seconds.")
		print(f"[filter] Skipped segments by type: {skip_typ}")
	except Exception as e:
		print(f"[filter] Error processing {cc_file.name}: {e}")


def _filter_cc_file_worker(args: tuple[str, str]) -> None:
	"""多进程 worker：包装 filter_cc_file，参数简单可 pickling。"""
	idx,cc_path_str, out_dir_str = args
	filter_cc_file(idx,Path(cc_path_str), Path(out_dir_str))

import functools
from concurrent.futures import ProcessPoolExecutor, as_completed

def process_valid_file(path_str, hash_seeds, ngram_n, minhash_num_hashes):
	from pathlib import Path
	path = Path(path_str)
	# 以二进制模式读取，所有offset/size均基于bytes
	with path.open("rb") as valid_f:
		content_bytes = valid_f.read()
	sep = b"<|endoftext|>"
	segments = []
	offsets = []
	cur = 0
	while cur < len(content_bytes):
		# 找下一个分隔符
		next_sep = content_bytes.find(sep, cur)
		if next_sep == -1:
			seg_bytes = content_bytes[cur:]
			seg_bytes = seg_bytes.strip()
			if seg_bytes:
				start = cur
				end = len(content_bytes)
				offsets.append((start, end))
				segments.append(seg_bytes)
			break
		seg_bytes = content_bytes[cur:next_sep]
		seg_bytes = seg_bytes.strip()
		if seg_bytes:
			start = cur + (0 if content_bytes[cur:cur+1] != b'\n' else 1)
			end = next_sep
			offsets.append((start, end))
			segments.append(seg_bytes)
		cur = next_sep + len(sep)
	result = []
	for (start, end), seg_bytes in zip(offsets, segments):
		# 尝试用utf-8解码，遇到问题忽略
		try:
			seg = seg_bytes.decode("utf-8", errors="ignore")
		except Exception:
			seg = ""
		ngrams = _get_ngrams(seg, ngram_n)
		sig = _minhash_signature(ngrams, minhash_num_hashes, hash_seeds)
		result.append((str(path), start, end-start, sig))
	gc.collect()
	return result

def main() -> None:
	"""遍历 CC_PATH 下所有 CC 文件并生成 valid_corpus。"""

	root = Path(CC_PATH)
	if not root.is_dir():
		raise FileNotFoundError(f"CC_PATH 不存在或不是目录: {root}")

	print(f"[filter] Reading CC files from {root}")

	print(f"[filter] Raw valid corpus dir: {RAW_VALID_CORPUS_DIR}")
	print(f"[filter] Final valid corpus dir: {VALID_CORPUS_DIR}")

	# 第一步：逐文件清洗，写入 RAW 目录（逐文件内部做精确去重）
	cc_files = [
		entry
		for entry in sorted(root.iterdir())
		if entry.name.endswith((".warc.gz", ".wet.gz"))
	]
	print(f"[filter] Found {len(cc_files)} CC files to filter; using {NUM_WORKERS} workers")
	assert None not in [str(p) for p in cc_files]
	if cc_files:
		with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
			list(
				executor.map(
					_filter_cc_file_worker,
					[(idx, str(p), str(RAW_VALID_CORPUS_DIR)) for idx, p in enumerate(cc_files)],
				)
			)

	print(f"[filter] All CC files filtered, now merging valid files with MinHash+LSH deduplication ...")

	# 第二步：以段落为单位进行MinHash+LSH去重，合并写入最终语料文件（仅存索引和签名，精确比对时按需读取磁盘）
	VALID_CORPUS_DIR.mkdir(parents=True, exist_ok=True)
	valid_files = sorted(p for p in RAW_VALID_CORPUS_DIR.iterdir() if p.is_file() and p.name.endswith('.valid'))

	random.seed(42)
	hash_seeds = [random.randint(0, 2**32-1) for _ in range(MINHASH_NUM_HASHES)]
	lsh_tables: List[Dict[Tuple[int], List[int]]] = [{} for _ in range(MINHASH_NUM_BANDS)]
	segment_index: List[Tuple[Path, int, int, List[int]]] = []  # (file_path, offset, size, minhash_sig)

	# 先扫描所有段落，记录offset/size和签名
	print("[filter] Scanning all segments and building index...")


	print("[filter] Scanning all segments and building index (multiprocess)...")
	all_segments: List[Tuple[str, int, int, List[int]]] = []
	with ProcessPoolExecutor(max_workers=32) as executor:
		futures = [executor.submit(process_valid_file, str(path), hash_seeds, MINHASH_NGRAMS, MINHASH_NUM_HASHES) for path in valid_files]
		for idx, future in enumerate(tqdm.tqdm(as_completed(futures), total=len(futures))):
			res = future.result()
			all_segments.extend(res)
	# for idx, path in enumerate(tqdm.tqdm(valid_files)):
	# 	res = process_valid_file(str(path), hash_seeds, MINHASH_NGRAMS, MINHASH_NUM_HASHES)
	# 	all_segments.extend(res)

	print(f"[filter] Total segments scanned: {len(all_segments)}")

	# 逐个做LSH去重，内存只存索引和签名，精确比对时再读取磁盘计算ngrams
	print("[filter] Deduplicating and writing final corpus ...")

	with FINAL_CORPUS_PATH.open("w", encoding="utf-8") as f:
		first = True
		kept_segments: List[Tuple[str, int, int, List[int]]] = []
		for idx, (file_path, offset, size, sig) in enumerate(tqdm.tqdm(all_segments)):
			bands = _lsh_bands(sig, MINHASH_NUM_BANDS)
			is_dup = False
			# print(file_path, offset, size)
			with open(file_path, "rb") as cf:
				cf.seek(offset)
				c_seg_bytes = cf.read(size)
			c_seg = c_seg_bytes.decode("utf-8", errors="ignore")
			ngrams1 = _get_ngrams(c_seg, MINHASH_NGRAMS)
			for band_idx, band in enumerate(bands):
				bucket = lsh_tables[band_idx].get(band, [])
				for kept_idx in bucket:
					k_file, k_offset, k_size, k_sig = kept_segments[kept_idx]
					# 需要时从磁盘读取内容并计算ngrams
					with open(k_file, "rb") as kf:
						kf.seek(k_offset)
						k_seg_bytes = kf.read(k_size)
					k_seg = k_seg_bytes.decode("utf-8", errors="ignore")
					ngrams2 = _get_ngrams(k_seg, MINHASH_NGRAMS)
					if _jaccard(ngrams1, ngrams2) >= MINHASH_JACCARD_THRESHOLD:
						is_dup = True
						break
				if is_dup:
					break
			if is_dup:
				continue
			# 新段落，写入文件
			with open(file_path, "rb") as cf:
				cf.seek(offset)
				seg_bytes = cf.read(size)
			seg = seg_bytes.decode("utf-8", errors="ignore")
			if not first:
				f.write("\n<|endoftext|>\n")
			first = False
			f.write(seg)
			# 更新LSH表和已保留索引
			kept_segments.append((file_path, offset, size, sig))
			for band_idx, band in enumerate(bands):
				lsh_tables[band_idx].setdefault(band, []).append(len(kept_segments)-1)

	print(f"[filter] Final deduplicated corpus written to {FINAL_CORPUS_PATH}")


if __name__ == "__main__":
	main()

