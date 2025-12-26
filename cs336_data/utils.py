import re
import os
from typing import Any

from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import bytes_to_str

from fastwarc.warc import ArchiveIterator, WarcRecordType

import fasttext

import nltk

CC_PATH = '/mnt/mnt/zjdx/commoncrawl/commoncrawl_wet_100g/'

_LANG_MODEL_PATH = '/mnt/mnt/zjdx/tyhcyq/lid.176.bin'
_NSFW_MODEL_PATH = '/mnt/mnt/zjdx/tyhcyq/jigsaw_fasttext_bigrams_nsfw_final.bin'
_TOXIC_MODEL_PATH = '/mnt/mnt/zjdx/tyhcyq/jigsaw_fasttext_bigrams_hatespeech_final.bin'
_QUALITY_MODEL_PATH = '/mnt/mnt/zjdx/tyhcyq/classifier/wiki_vs_cc_fasttext_1.bin'

_LANG_MODEL= None
_NSFW_MODEL= None
_TOXIC_MODEL = None
_QUALITY_MODEL = None


def extract_text_from_html_bytes(html_bytes):
    """Extract text from HTML bytes using resiliparse.

    The input is raw HTML bytes; we first convert them to a
    Unicode string using Resiliparse's encoding detector and
    then extract cleaned plain text.
    """
    try:
        html_str = bytes_to_str(html_bytes)
        return extract_plain_text(html_str)
    except Exception:
        return None

def identify_language(text: str) -> tuple[Any, float]:
    """Identify the language of the given text using FastText.

    Uses the cached language ID model at `_LANG_MODEL_PATH`.
    """

    global _LANG_MODEL
    if _LANG_MODEL is None:
        _LANG_MODEL = fasttext.load_model(_LANG_MODEL_PATH)
    model = _LANG_MODEL

    clean_text = " ".join(text.splitlines()).strip()
    predictions = model.predict(clean_text, k=1)  # Get top prediction
    lang_label = predictions[0][0]  # e.g., '__label__en'
    confidence = float(predictions[1][0])  # Confidence score
    language = lang_label.replace('__label__', '')
    return language, confidence

def mask_emails(text: str) -> tuple[str, int]:
    """Mask all email addresses in the text.

    All detected email addresses are replaced with the string
    "|||EMAIL_ADDRESS|||". The function returns the masked text and the
    number of replacements that were made.
    """

    email_pattern = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
    masked_text, num_masked = email_pattern.subn("|||EMAIL_ADDRESS|||", text)
    return masked_text, num_masked

def mask_phone_numbers(text: str) -> tuple[str, int]:
    """Mask all phone numbers in the text.

    All detected phone numbers are replaced with the string
    "|||PHONE_NUMBER|||". The function returns the masked text and the
    number of replacements that were made.
    """

    phone_pattern = phone_pattern = re.compile(
        r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    )
    masked_text, num_masked = phone_pattern.subn("|||PHONE_NUMBER|||", text)
    return masked_text, num_masked

def mask_ips(text: str) -> tuple[str, int]:
    """Mask all IP addresses in the text.

    All detected IP addresses are replaced with the string
    "|||IP_ADDRESS|||". The function returns the masked text and the
    number of replacements that were made.
    """

    ip_pattern = re.compile(
        r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}"
        r"(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"
    )
    masked_text, num_masked = ip_pattern.subn("|||IP_ADDRESS|||", text)
    return masked_text, num_masked


def classify_nsfw(text: str) -> tuple[str, float]:
    """Classify whether the text is NSFW using a FastText model.

    Uses the Jigsaw NSFW fastText model at `_NSFW_MODEL_PATH` and
    returns (prediction, score) with prediction in {"nsfw", "non-nsfw"}.
    """

    global _NSFW_MODEL
    if _NSFW_MODEL is None:
        _NSFW_MODEL = fasttext.load_model(_NSFW_MODEL_PATH)
    model = _NSFW_MODEL

    clean_text = " ".join(text.splitlines()).strip()
    if not clean_text:
        return "non-nsfw", 0.0

    labels, scores = model.predict(clean_text, k=1)
    label = labels[0]
    score = float(scores[0])
    label_l = label.lower()

    if label_l == '__label__non-nsfw':
        prediction = 'non-nsfw'
    elif label_l == '__label__nsfw':
        prediction = 'nsfw'

    return prediction, score


def classify_toxic_speech(text: str) -> tuple[str, float]:
    """Classify whether the text is toxic/hateful using a FastText model.

    Uses the Jigsaw hate-speech fastText model at `_TOXIC_MODEL_PATH` and
    returns (prediction, score) with prediction in {"toxic", "non-toxic"}.
    """

    global _TOXIC_MODEL
    if _TOXIC_MODEL is None:
        _TOXIC_MODEL = fasttext.load_model(_TOXIC_MODEL_PATH)
    model = _TOXIC_MODEL

    clean_text = " ".join(text.splitlines()).strip()
    if not clean_text:
        return "non-toxic", 0.0

    labels, scores = model.predict(clean_text, k=1)
    label = labels[0]              # 比如 '__label__toxic'
    score = float(scores[0])

    raw = label.replace('__label__', '').lower()  # 得到 'toxic' 或 'non-toxic' 等

    if raw == 'toxic':
        prediction = 'toxic'
    else:
        prediction = 'non-toxic'

    return prediction, score


def gopher_quality_filter(text: str) -> bool:
    """Filter text using simple Gopher-style quality rules.

    Rules (all must pass):
    - Number of non-symbol words in [50, 100000].
    - Average word length between 3 and 10 characters.
    - At most 30% of lines end with an ellipsis ('...').
    - At least 80% of words contain an alphabetic character.
    """

    # Tokenize text into word-like tokens. Use NLTK if available, else fallback.
    try:
        tokens = nltk.word_tokenize(text)
    except Exception:
        tokens = text.split()

    # Consider non-symbol words: tokens that contain at least one
    # alphanumeric character (exclude pure punctuation).
    word_tokens = [t for t in tokens if any(ch.isalnum() for ch in t)]
    num_words = len(word_tokens)

    # 1) Length constraint on number of words.
    if num_words < 50 or num_words > 100000:
        return False

    # Clean leading/trailing punctuation for length calculations.
    cleaned_words = [re.sub(r"^\W+|\W+$", "", w) for w in word_tokens]
    lengths = [len(w) for w in cleaned_words if w]
    if not lengths:
        return False

    avg_len = sum(lengths) / len(lengths)

    # 2) Average word length constraint.
    if avg_len < 3 or avg_len > 10:
        return False

    # 3) Fraction of lines ending with ellipsis.
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if lines:
        ellipsis_lines = [ln for ln in lines if ln.rstrip().endswith("...")]
        frac_ellipsis = len(ellipsis_lines) / len(lines)
        if frac_ellipsis > 0.3:
            return False

        # 新增规则：一行少于10个单词的行占比超过30%则过滤
        # short_lines = [ln for ln in lines if len(ln.split()) < 5]
        # total_short = sum(len(t.split()) for t in short_lines)
        # frac_short = len(short_lines) / len(lines)
        # if frac_short > 0.3 and total_short / num_words > 0.3:
        #     return False
        
        # 新增规则: 检查URL和邮箱出现比例，若超过10%则过滤
        url_pattern = re.compile(r'https?://|www\.')
        email_pattern = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
        url_count = 0
        email_count = 0
        for ln in lines:
            url_count += len(url_pattern.findall(ln))
            email_count += len(email_pattern.findall(ln))
        total_lines = len(lines) if lines else 1
        if (url_count + email_count) / total_lines > 0.3:
            return False
        del lines

    # 4) At least 80% of words contain an alphabetic character.
    alpha_words = [w for w in word_tokens if any(ch.isalpha() for ch in w)]
    if not word_tokens:
        return False
    frac_alpha = len(alpha_words) / len(word_tokens)
    if frac_alpha < 0.8:
        return False
    


    return True


def classify_quality(text: str) -> tuple[str, float]:
    """Classify text quality as Wiki-like or CC-like using fastText.

    Uses the wiki-vs-cc fastText model trained in ``cs336_data.classifier``.
    Returns (prediction, score) where prediction is "wiki" or "cc".
    """

    global _QUALITY_MODEL
    if _QUALITY_MODEL is None:
        _QUALITY_MODEL = fasttext.load_model(_QUALITY_MODEL_PATH)
    model = _QUALITY_MODEL

    clean_text = " ".join(text.splitlines()).strip()
    if not clean_text:
        # 空文本默认视为低质量 CC
        return "cc", 0.0

    labels, scores = model.predict(clean_text, k=1)
    label = labels[0]
    score = float(scores[0])

    raw = label.replace('__label__', '').lower()
    if raw in ("wiki", "cc"):
        prediction = raw
    else:
        # 未知标签时保守地视为 CC
        prediction = "cc"
    # print(prediction, score)
    return prediction, score


def _shingle_text(text: str, ngrams: int) -> set[int]:
    """Convert text into a set of hashed word n-grams.

    Uses Python's built-in ``hash`` over space-joined n-grams as the
    shingle identifier. This is the basic building block for Jaccard
    similarity and MinHash signatures.
    """

    tokens = text.split()
    if not tokens:
        return set()

    if ngrams <= 1 or len(tokens) <= ngrams:
        return {hash(" ".join(tokens))}

    shingles: set[int] = set()
    for i in range(len(tokens) - ngrams + 1):
        ngram = " ".join(tokens[i : i + ngrams])
        shingles.add(hash(ngram))
    return shingles


def _minhash_signature(shingles: set[int], num_hashes: int) -> list[int]:
    """Compute a MinHash signature for a set of hashed shingles.

    We simulate ``num_hashes`` different hash functions by combining each
    shingle with a seed index and taking the minimum value per seed.
    """

    if not shingles:
        return [0] * num_hashes

    signature: list[int] = []
    for seed in range(num_hashes):
        min_val = None
        for sh in shingles:
            h = hash((seed, sh))
            if min_val is None or h < min_val:
                min_val = h
        # ``min_val`` cannot be None here because ``shingles`` is non-empty.
        signature.append(min_val if min_val is not None else 0)
    return signature

def minhash_lsh_deduplication_docs(
    docs: list[tuple[os.PathLike, str, set[int]]],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
) -> None:

    if not docs:
        return

    # Compute MinHash signatures.
    signatures: list[list[int]] = [
        _minhash_signature(shingles, num_hashes) for _, _, shingles in docs
    ]

    # LSH banding: group documents that share identical bands.
    rows_per_band = max(1, num_hashes // num_bands)
    band_buckets: dict[tuple[int, tuple[int, ...]], list[int]] = {}
    candidate_pairs: set[tuple[int, int]] = set()

    for idx, sig in enumerate(signatures):
        for band in range(num_bands):
            start = band * rows_per_band
            end = start + rows_per_band
            if start >= len(sig):
                break
            band_tuple = tuple(sig[start:end])
            key = (band, band_tuple)
            bucket = band_buckets.setdefault(key, [])
            for other_idx in bucket:
                i, j = sorted((idx, other_idx))
                candidate_pairs.add((i, j))
            bucket.append(idx)

    # Greedy selection of non-duplicate documents.
    kept_indices: list[int] = []
    shingle_sets = [sh for _, _, sh in docs]

    for i in range(len(docs)):
        is_dup = False
        for kept_idx in kept_indices:
            if (min(i, kept_idx), max(i, kept_idx)) not in candidate_pairs:
                continue
            a = shingle_sets[i]
            b = shingle_sets[kept_idx]
            if not a and not b:
                is_dup = True
                break
            if not a or not b:
                continue
            inter = len(a & b)
            union = len(a | b)
            jacc = inter / union if union else 0.0
            if jacc >= jaccard_threshold:
                is_dup = True
                break
        if not is_dup:
            kept_indices.append(i)

    # Write kept documents to output_directory.
    os.makedirs(output_directory, exist_ok=True)
    for idx in kept_indices:
        path_like, text, _ = docs[idx]
        assert text is None
        with open(os.fspath(path_like), "r") as f:
            text = f.read()
        in_path = os.fspath(path_like)
        out_path = os.path.join(output_directory, os.path.basename(in_path))
        with open(out_path, "w") as f:
            f.write(text)


def minhash_lsh_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
) -> None:
    """Deduplicate documents using MinHash + LSH over hashed n-grams.

    For each document, we compute a set of hashed word n-grams, then a
    MinHash signature of length ``num_hashes``. We apply banding with
    ``num_bands`` bands to find candidate duplicate pairs, and for each
    candidate pair we compute the Jaccard similarity of their shingle
    sets. If the similarity exceeds ``jaccard_threshold``, we treat one
    of them as a duplicate and keep only the first.

    The resulting deduplicated documents are written to
    ``output_directory`` with the same basenames as the input files.
    """

    # Read documents and build shingle sets.
    docs: list[tuple[os.PathLike, str, set[int]]] = []
    for path_like in input_files:
        path = os.fspath(path_like)
        with open(path, "r") as f:
            text = f.read()
        shingles = _shingle_text(text, ngrams)
        docs.append((path_like, None, shingles))
    minhash_lsh_deduplication_docs(
        docs,
        num_hashes,
        num_bands,
        ngrams,
        jaccard_threshold,
        output_directory,
    )

def exact_line_deduplication(
    input_files: list[os.PathLike], output_directory: os.PathLike
):
    """Perform exact line-level deduplication across documents using hash(line).

    For all input files, we treat two lines as identical if their string
    representations are equal; to speed up counting, we use Python's built-in
    ``hash(line)`` as the key. Any line that appears (exactly equal) in more
    than one document is removed from all documents. The resulting documents
    are written to ``output_directory`` with the same basenames as inputs.
    """

    # First pass: for each unique line (by string equality), count in how many
    # documents it appears. We use hash(line) as the dictionary key.
    line_doc_counts: dict[int, int] = {}
    for path_like in input_files:
        path = os.fspath(path_like)
        with open(path, "r") as f:
            lines = f.readlines()
        # Only count each line once per document.
        seen_hashes = set()
        for line in lines:
            line_hash = hash(line)
            if line_hash in seen_hashes:
                continue
            seen_hashes.add(line_hash)
            line_doc_counts[line_hash] = line_doc_counts.get(line_hash, 0) + 1

    # Second pass: write out documents with lines that are unique across
    # the corpus (i.e., appear in exactly one document).
    os.makedirs(output_directory, exist_ok=True)
    for path_like in input_files:
        path = os.fspath(path_like)
        with open(path, "r") as f:
            lines = f.readlines()

        filtered_lines = [
            ln for ln in lines if line_doc_counts.get(hash(ln), 0) == 1
        ]

        out_path = os.path.join(output_directory, os.path.basename(path))
        with open(out_path, "w") as out_f:
            out_f.writelines(filtered_lines)



if __name__ == '__main__':
    print("This is the main module of the cs336_data package.")