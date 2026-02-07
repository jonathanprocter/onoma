import html
import os
import re
import unicodedata
from importlib import resources

from titlecase import titlecase

ACRONYMS_FALLBACK = {
    "CBT",
    "DBT",
    "ACT",
    "EMDR",
    "REBT",
    "ERP",
    "TF-CBT",
    "CPT",
    "EFT",
    "IFS",
    "IPT",
    "MI",
    "MBT",
    "PE",
    "SFBT",
    "DSM",
    "ICD",
    "PDM",
    "BDI",
    "BAI",
    "PHQ",
    "GAD",
    "PCL",
    "MMPI",
    "MCMI",
    "APA",
    "ACA",
    "NASW",
    "NBCC",
    "WHO",
    "NIH",
    "NIMH",
    "SAMHSA",
    "PDF",
    "EPUB",
    "ISBN",
    "OCR",
    "HTML",
    "CSS",
    "PTSD",
    "ADHD",
    "OCD",
    "BPD",
    "MDD",
    "ASD",
    "SUD",
    "PhD",
    "PsyD",
    "EdD",
    "LCSW",
    "LPC",
    "LMFT",
    "LMHC",
    "NCC",
    "BCBA",
    "ADOS",
    "WAIS",
    "WISC",
    "AMHCA",
}

ALWAYS_LOWER = {"vs", "vs.", "v.", "etc", "et", "al", "e.g.", "i.e."}

ROMAN_RE = re.compile(
    r"^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$",
    re.IGNORECASE,
)
ORDINAL_RE = re.compile(r"^\d+(st|nd|rd|th)$", re.IGNORECASE)

YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
ISBN_RE = re.compile(r"\b97[89][- ]?\d{1,5}[- ]?\d{1,7}[- ]?\d{1,7}[- ]?\d\b", re.IGNORECASE)

KNOWN_EXTS = {
    ".pdf",
    ".epub",
    ".mobi",
    ".azw",
    ".azw3",
    ".kfx",
    ".djvu",
    ".cbr",
    ".cbz",
    ".fb2",
    ".lit",
    ".lrf",
    ".pdb",
    ".prc",
    ".doc",
    ".docx",
    ".odt",
    ".rtf",
    ".txt",
    ".ppt",
    ".pptx",
    ".key",
    ".mp3",
    ".m4a",
    ".m4b",
    ".3gpp",
}


def _load_acronyms(config: dict) -> set[str]:
    path = (config or {}).get("acronyms_path")
    if path:
        try:
            with open(os.path.expanduser(path)) as f:
                return {line.strip() for line in f if line.strip() and not line.startswith("#")}
        except Exception:
            return set(ACRONYMS_FALLBACK)
    try:
        with resources.files("onomatool.resources").joinpath("acronyms.txt").open() as f:
            return {line.strip() for line in f if line.strip() and not line.startswith("#")}
    except Exception:
        return set(ACRONYMS_FALLBACK)


def _acronym_map(acronyms: set[str]) -> dict[str, str]:
    return {a.upper(): a for a in acronyms}


def _normalize_editions(text: str) -> str:
    # Normalize edition markers
    def _edition_repl(match: re.Match) -> str:
        num = match.group(1)
        suffix = (match.group(2) or "").lower()
        if not suffix:
            # infer common suffix
            if num.endswith("1") and not num.endswith("11"):
                suffix = "st"
            elif num.endswith("2") and not num.endswith("12"):
                suffix = "nd"
            elif num.endswith("3") and not num.endswith("13"):
                suffix = "rd"
            else:
                suffix = "th"
        return f"{num}{suffix} Edition"

    text = re.sub(r"\b(\d+)(st|nd|rd|th)?\s*ed(ition)?\b", _edition_repl, text, flags=re.IGNORECASE)
    text = re.sub(r"\bsecond\s+edition\b", "2nd Edition", text, flags=re.IGNORECASE)
    text = re.sub(r"\bthird\s+edition\b", "3rd Edition", text, flags=re.IGNORECASE)
    text = re.sub(r"\bfourth\s+edition\b", "4th Edition", text, flags=re.IGNORECASE)
    text = re.sub(r"\brev\.?\s*ed\.?\b", "Revised Edition", text, flags=re.IGNORECASE)
    text = re.sub(r"\brevised\s+edition\b", "Revised Edition", text, flags=re.IGNORECASE)

    # Normalize volume/part markers
    text = re.sub(r"\bvol(?:\.|ume)?\s*(\d+|[ivx]+)\b", r"Vol. \1", text, flags=re.IGNORECASE)
    text = re.sub(r"\bpt\.?\s*(\d+|[ivx]+)\b", r"Part \1", text, flags=re.IGNORECASE)
    text = re.sub(r"\bpart\s*(\d+|[ivx]+)\b", r"Part \1", text, flags=re.IGNORECASE)
    return text


def _strip_metadata(text: str) -> str:
    # Drop pipe-separated metadata blocks
    if "|" in text:
        text = text.split("|", 1)[0]
    # Remove bracketed year/ISBN/etc blocks
    text = re.sub(r"[\(\[][^)\]]*(19|20)\d{2}[^)\]]*[\)\]]", "", text)
    text = re.sub(r"[\(\[][^)\]]*isbn[^)\]]*[\)\]]", "", text, flags=re.IGNORECASE)
    # Remove trailing "by Author" style suffixes
    text = re.sub(r"\bby\s+[^-–—]+$", "", text, flags=re.IGNORECASE)
    # Remove years anywhere
    text = YEAR_RE.sub("", text)
    return text


def _sanitize_filename(text: str) -> str:
    text = html.unescape(text)
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\x00", "")
    text = text.replace(":", " - ")
    text = text.replace("/", "-").replace("\\", "-")
    text = text.replace('"', "'")
    text = re.sub(r"[?*<>|]", "", text)
    # Normalize spaced dashes (subtitle separators)
    text = re.sub(r"\s+[-–—]\s+", " - ", text)
    text = re.sub(r"( - ){2,}", " - ", text)
    # Collapse whitespace and trim
    text = re.sub(r"\s+", " ", text).strip(" .")
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def _strip_extension_tokens(text: str, original_ext: str | None) -> str:
    ext = (original_ext or "").lower()
    if ext and ext not in KNOWN_EXTS:
        ext = ""
    # Remove any trailing known extensions in the suggestion
    while True:
        base, suffix = os.path.splitext(text)
        if suffix.lower() in KNOWN_EXTS:
            text = base
            continue
        break
    if ext and text.lower().endswith(ext):
        text = text[: -len(ext)]
    return text.strip()


def _titlecase_callback(acronym_map: dict[str, str]):
    def _callback(word: str, **_kwargs):
        upper = word.upper()
        if upper in acronym_map:
            return acronym_map[upper]
        if ROMAN_RE.match(word) and len(word) > 0:
            return word.upper()
        if ORDINAL_RE.match(word):
            return word[:-2] + word[-2:].lower()
        if word.lower() in ALWAYS_LOWER:
            return word.lower()
        return None

    return _callback


def _capitalize_after_subtitle(text: str, separator: str = " - ") -> str:
    if separator not in text:
        return text
    parts = text.split(separator)
    fixed = [parts[0]]
    for seg in parts[1:]:
        seg = seg.lstrip()
        if not seg:
            fixed.append(seg)
            continue
        # Capitalize first alpha character
        m = re.search(r"[A-Za-z]", seg)
        if not m:
            fixed.append(seg)
            continue
        idx = m.start()
        seg = seg[:idx] + seg[idx].upper() + seg[idx + 1 :]
        fixed.append(seg)
    return separator.join(fixed)


def apply_titlecase(raw_title: str, config: dict | None = None, original_ext: str | None = None) -> str:
    cfg = config or {}
    text = raw_title
    if cfg.get("strip_metadata", True):
        text = _strip_metadata(text)
    text = _strip_extension_tokens(text, original_ext)
    text = _normalize_editions(text)
    text = _sanitize_filename(text)

    acronyms = _load_acronyms(cfg)
    acronym_map = _acronym_map(acronyms)

    if cfg.get("enforce_title_case", True):
        text = text.lower()
    text = titlecase(text, callback=_titlecase_callback(acronym_map))
    text = _capitalize_after_subtitle(text, separator=cfg.get("subtitle_separator", " - "))
    # Force known acronyms to canonical case after titlecase
    for upper, canonical in acronym_map.items():
        text = re.sub(rf"\\b{re.escape(upper)}\\b", canonical, text)
    return text.strip()


def evaluate_title(title: str, config: dict | None = None) -> tuple[float, list[str]]:
    cfg = config or {}
    flags: list[str] = []
    score = 0.0

    acronyms = _load_acronyms(cfg)
    acronym_map = _acronym_map(acronyms)

    # Word count (2-15)
    words = [w for w in re.split(r"\s+", title.strip()) if w]
    if 2 <= len(words) <= 15:
        score += 25
    else:
        flags.append("word_count")

    # Illegal characters / cleanliness
    illegal = bool(re.search(r'[\\/:*?"<>|]', title))
    if not illegal:
        score += 25
    else:
        flags.append("illegal_chars")

    # Metadata remnants (year/ISBN)
    if YEAR_RE.search(title) or ISBN_RE.search(title):
        flags.append("metadata")
    else:
        score += 25

    # Acronym consistency
    bad_acronym = False
    for upper, canonical in acronym_map.items():
        if re.search(rf"\b{re.escape(upper)}\b", title, flags=re.IGNORECASE):
            if canonical not in title:
                bad_acronym = True
                break
    if not bad_acronym:
        score += 25
    else:
        flags.append("acronym_case")

    return score, flags
