"""Conservative Easy Data Augmentation for Vietnamese student feedback."""

from __future__ import annotations

from dataclasses import dataclass
import re

from src.augmentation.base import TextAugmenter

TOKEN_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)
WORD_RE = re.compile(r"^\w+$", flags=re.UNICODE)
NO_SPACE_BEFORE = set(".,!?;:%)]}")
NO_SPACE_AFTER = set("([{")

NEGATION_WORDS = {
    "chưa",
    "chẳng",
    "đâu",
    "đừng",
    "không",
    "ko",
    "k",
    "khỏi",
}

SENTIMENT_WORDS = {
    "chán",
    "chậm",
    "dở",
    "dễ",
    "ghét",
    "hay",
    "khó",
    "kém",
    "mệt",
    "nhàm",
    "nhanh",
    "nhiệt",
    "rõ",
    "ổn",
    "tận",
    "tình",
    "tệ",
    "thú",
    "thích",
    "tốt",
    "vui",
    "xuất",
    "yếu",
}

STOPWORDS = {
    "bị",
    "các",
    "cái",
    "cho",
    "của",
    "đã",
    "được",
    "em",
    "hơi",
    "là",
    "mà",
    "mình",
    "này",
    "nên",
    "quá",
    "rất",
    "thì",
    "và",
    "với",
}

SYNONYMS = {
    "bài": ("nội dung",),
    "bận": ("nhiều việc",),
    "buổi": ("tiết",),
    "cần": ("nên",),
    "chậm": ("lâu",),
    "chương": ("phần",),
    "dạy": ("truyền đạt",),
    "đẹp": ("ổn",),
    "đầy": ("đủ",),
    "giảng": ("trình bày",),
    "hay": ("thú vị", "tốt"),
    "hiểu": ("nắm được",),
    "học": ("theo học",),
    "lâu": ("chậm",),
    "mau": ("nhanh",),
    "nhanh": ("mau",),
    "nhiều": ("khá nhiều",),
    "nội": ("phần",),
    "ổn": ("tốt",),
    "phòng": ("lớp",),
    "rõ": ("dễ hiểu",),
    "tài": ("tư liệu",),
    "thích": ("hứng thú",),
    "tốt": ("ổn", "hay"),
}

PHRASE_SYNONYMS = {
    "bài tập": ("bài làm",),
    "cơ sở vật chất": ("trang thiết bị",),
    "dễ hiểu": ("rõ ràng", "dễ nắm bắt"),
    "giảng dạy": ("truyền đạt", "trình bày"),
    "giáo trình": ("tài liệu học tập",),
    "khó hiểu": ("khó nắm bắt",),
    "môn học": ("học phần",),
    "nhiệt tình": ("tận tâm", "chu đáo"),
    "quan tâm": ("chú ý",),
    "sinh viên": ("người học",),
    "tận tình": ("nhiệt tình", "chu đáo"),
    "thực hành": ("luyện tập",),
}


@dataclass(frozen=True)
class EDAConfig:
    """Configuration for one conservative EDA augmenter."""

    alpha_sr: float = 0.1
    alpha_ri: float = 0.1
    alpha_rs: float = 0.1
    p_rd: float = 0.1


class EDAAugmenter(TextAugmenter):
    """Easy Data Augmentation with low-risk Vietnamese operations."""

    method_name = "eda"

    def __init__(
        self,
        seed: int = 42,
        alpha_sr: float = 0.1,
        alpha_ri: float = 0.1,
        alpha_rs: float = 0.1,
        p_rd: float = 0.1,
    ) -> None:
        super().__init__(seed=seed)
        self.config = EDAConfig(
            alpha_sr=max(0.0, float(alpha_sr)),
            alpha_ri=max(0.0, float(alpha_ri)),
            alpha_rs=max(0.0, float(alpha_rs)),
            p_rd=max(0.0, float(p_rd)),
        )

    @staticmethod
    def tokenize(sentence: str) -> list[str]:
        """Tokenize into words and punctuation without external dependencies."""
        return TOKEN_RE.findall(str(sentence))

    @staticmethod
    def detokenize(tokens: list[str]) -> str:
        """Rebuild a sentence with simple Vietnamese-friendly spacing."""
        output = ""
        for token in tokens:
            if not output:
                output = token
            elif token in NO_SPACE_BEFORE or output[-1] in NO_SPACE_AFTER:
                output += token
            else:
                output += f" {token}"
        return TextAugmenter.normalize_text(output)

    @staticmethod
    def is_word(token: str) -> bool:
        """Return True for word tokens and False for punctuation."""
        return bool(WORD_RE.match(token))

    @staticmethod
    def match_case(source: str, replacement: str) -> str:
        """Apply coarse casing from the source token to the replacement."""
        if source.isupper():
            return replacement.upper()
        if source[:1].isupper():
            return replacement[:1].upper() + replacement[1:]
        return replacement

    def synonym_for(self, token: str) -> str | None:
        """Return a deterministic-random synonym for a token when available."""
        choices = SYNONYMS.get(token.lower())
        if not choices:
            return None
        return self.match_case(token, self.rng.choice(list(choices)))

    def word_count(self, tokens: list[str]) -> int:
        """Count word tokens."""
        return sum(1 for token in tokens if self.is_word(token))

    def n_changes(self, tokens: list[str], alpha: float) -> int:
        """Translate EDA alpha into a small number of edits."""
        return max(1, int(round(alpha * max(1, self.word_count(tokens)))))

    def replacement_indices(self, tokens: list[str]) -> list[int]:
        """Indices where synonym replacement is possible."""
        return [
            index
            for index, token in enumerate(tokens)
            if self.is_word(token) and token.lower() in SYNONYMS
        ]

    def content_indices(self, tokens: list[str], protect_sentiment: bool = False) -> list[int]:
        """Indices that can be moved, duplicated, or deleted."""
        protected = set(NEGATION_WORDS)
        if protect_sentiment:
            protected.update(SENTIMENT_WORDS)
        return [
            index
            for index, token in enumerate(tokens)
            if self.is_word(token)
            and token.lower() not in protected
            and token.lower() not in STOPWORDS
        ]

    def synonym_replacement(self, tokens: list[str]) -> list[str]:
        """Replace a small number of words with safe synonyms."""
        output = list(tokens)
        sentence = self.detokenize(output)
        phrase_candidates = [
            phrase for phrase in PHRASE_SYNONYMS if phrase in sentence.lower()
        ]
        if phrase_candidates:
            phrase = self.rng.choice(phrase_candidates)
            replacement = self.rng.choice(list(PHRASE_SYNONYMS[phrase]))
            pattern = re.compile(re.escape(phrase), flags=re.IGNORECASE)
            return self.tokenize(pattern.sub(replacement, sentence, count=1))

        indices = self.replacement_indices(output)
        self.rng.shuffle(indices)
        for index in indices[: self.n_changes(output, self.config.alpha_sr)]:
            synonym = self.synonym_for(output[index])
            if synonym:
                output[index] = synonym
        return output

    def random_insertion(self, tokens: list[str]) -> list[str]:
        """Insert safe synonyms or duplicate content words."""
        output = list(tokens)
        indices = self.content_indices(output, protect_sentiment=True)
        if not indices:
            return output
        for _ in range(self.n_changes(tokens, self.config.alpha_ri)):
            source_index = self.rng.choice(indices)
            token = output[source_index]
            inserted = self.synonym_for(token) or token
            insert_at = min(len(output), source_index + 1)
            output.insert(insert_at, inserted)
        return output

    def random_swap(self, tokens: list[str]) -> list[str]:
        """Swap a small number of non-critical words."""
        output = list(tokens)
        indices = self.content_indices(output, protect_sentiment=True)
        if len(indices) < 2:
            return output
        for _ in range(self.n_changes(tokens, self.config.alpha_rs)):
            left, right = self.rng.sample(indices, 2)
            output[left], output[right] = output[right], output[left]
        return output

    def random_deletion(self, tokens: list[str]) -> list[str]:
        """Delete a few non-critical tokens while keeping the sentence readable."""
        indices = set(self.content_indices(tokens, protect_sentiment=True))
        if len(indices) <= 1 or self.word_count(tokens) <= 3:
            return list(tokens)

        output = [
            token
            for index, token in enumerate(tokens)
            if index not in indices or self.rng.random() > self.config.p_rd
        ]
        if output == tokens:
            drop_index = self.rng.choice(list(indices))
            output = [token for index, token in enumerate(tokens) if index != drop_index]
        if self.word_count(output) < 2:
            return list(tokens)
        return output

    def augment_sentence(self, sentence: str) -> str:
        """Return one EDA variant of a sentence."""
        tokens = self.tokenize(sentence)
        if self.word_count(tokens) < 2:
            return sentence

        operations = [
            (self.config.alpha_sr, self.synonym_replacement),
            (self.config.alpha_ri, self.random_insertion),
            (self.config.alpha_rs, self.random_swap),
            (self.config.p_rd, self.random_deletion),
        ]
        operations = [(weight, op) for weight, op in operations if weight > 0]
        if not operations:
            return sentence

        weights = [weight for weight, _op in operations]
        for _attempt in range(len(operations) * 2):
            operation = self.rng.choices(
                [op for _weight, op in operations],
                weights=weights,
                k=1,
            )[0]
            augmented = self.detokenize(operation(tokens))
            if augmented and augmented != sentence:
                return augmented

        return sentence
