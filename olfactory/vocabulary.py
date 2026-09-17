"""Canonical odor vocabularies used in the paper.

English labels are the internal identifiers. Swedish embeddings and human
matrix labels are translated with the fixed one-to-one mapping below.
"""

from itertools import combinations


ENGLISH_TO_SWEDISH = {
    "almond": "mandel",
    "ammonia": "ammoniak",
    "animal": "djur",
    "anise": "anis",
    "apple": "äpple",
    "bakery": "bageri",
    "banana": "banan",
    "bark": "bark",
    "beer": "öl",
    "berry": "bär",
    "birch": "björk",
    "blackberry": "björnbär",
    "blood": "blod",
    "bread": "bröd",
    "camphor": "kamfer",
    "caramel": "kola",
    "cardboard": "kartong",
    "carrot": "morot",
    "cedar": "ceder",
    "celery": "selleri",
    "cheese": "ost",
    "cherry": "körsbär",
    "chocolate": "choklad",
    "cinnamon": "kanel",
    "citrus": "citrus",
    "clove": "kryddnejlika",
    "coconut": "kokosnöt",
    "coffee": "kaffe",
    "cologne": "cologne",
    "cork": "kork",
    "creosote": "kreosot",
    "cucumber": "gurka",
    "detergent": "rengöringsmedel",
    "disinfectant": "desinfektionsmedel",
    "egg": "ägg",
    "eucalyptus": "eukalyptus",
    "fish": "fisk",
    "garlic": "vitlök",
    "gasoline": "bensin",
    "geranium": "geranium",
    "glue": "lim",
    "grape": "druva",
    "grapefruit": "grapefrukt",
    "grass": "gräs",
    "ham": "skinka",
    "hay": "hö",
    "honey": "honung",
    "incense": "rökelse",
    "kerosene": "fotogen",
    "lavender": "lavendel",
    "leather": "läder",
    "lemon": "citron",
    "licorice": "lakrits",
    "manure": "gödsel",
    "mint": "mint",
    "mothball": "malpåse",
    "mouse": "mus",
    "mushroom": "svamp",
    "mustard": "senap",
    "oak": "ek",
    "onion": "lök",
    "orange": "apelsin",
    "paint": "målarfärg",
    "peach": "persika",
    "peanut": "jordnöt",
    "pear": "päron",
    "peppermint": "pepparmint",
    "perfume": "parfym",
    "pine": "tallbarr",
    "pineapple": "ananas",
    "popcorn": "popcorn",
    "potato": "potatis",
    "raisin": "russin",
    "raspberry": "hallon",
    "rope": "rep",
    "rose": "ros",
    "rubber": "gummi",
    "rum": "rom",
    "smoke": "rök",
    "soap": "tvål",
    "solvent": "lösningsmedel",
    "strawberry": "jordgubbe",
    "syrup": "sirap",
    "tar": "tjära",
    "tea": "te",
    "tobacco": "tobak",
    "turpentine": "terpentin",
    "urine": "urin",
    "vanilla": "vanilj",
    "varnish": "lack",
    "vegetable": "grönsak",
    "vinegar": "vinäger",
    "violet": "violett",
    "walnut": "valnöt",
    "weed": "ogräs",
    "wine": "vin",
}

SWEDISH_TO_ENGLISH = {swedish: english for english, swedish in ENGLISH_TO_SWEDISH.items()}

ODOR_BASED_WORDS_EN = (
    "pineapple", "banana", "gasoline", "lemon", "fish", "coffee", "cinnamon", "clove",
    "licorice", "leather", "peppermint", "rose", "mushroom", "turpentine", "garlic", "apple",
)

DRAVNIEKS_WORDS_EN = (
    "citrus", "lemon", "grapefruit", "orange", "pineapple", "grape", "strawberry", "apple",
    "pear", "peach", "banana", "rose", "violet", "lavender", "cologne", "perfume", "honey",
    "cherry", "berry", "almond", "clove", "cinnamon", "tea", "oak", "cedar", "mothball",
    "peppermint", "camphor", "eucalyptus", "chocolate", "vanilla", "syrup", "caramel", "raisin",
    "coconut", "anise", "licorice", "detergent", "gasoline", "solvent", "turpentine", "pine",
    "geranium", "celery", "vegetable", "weed", "grass", "cucumber", "hay", "bakery", "bread",
    "beer", "leather", "cardboard", "rope", "potato", "mouse", "mushroom", "peanut", "egg",
    "birch", "cork", "incense", "coffee", "tar", "creosote", "disinfectant", "vinegar", "ammonia",
    "urine", "rubber", "kerosene", "paint", "varnish", "popcorn", "cheese", "garlic", "onion",
    "blood", "animal", "manure", "bark",
)

BERT_WORDS_EN = tuple(ENGLISH_TO_SWEDISH)
BERT_WORDS_SV = tuple(ENGLISH_TO_SWEDISH[word] for word in BERT_WORDS_EN)


def words_for_language(words_en: tuple[str, ...], language: str) -> tuple[str, ...]:
    if language == "en":
        return words_en
    if language == "sv":
        return tuple(ENGLISH_TO_SWEDISH[word] for word in words_en)
    raise ValueError(f"Unsupported language: {language!r}")


def canonical_word(word: str, language: str) -> str:
    word = word.strip().lower()
    if language == "en":
        return word
    if language == "sv":
        try:
            return SWEDISH_TO_ENGLISH[word]
        except KeyError as exc:
            raise KeyError(f"No Swedish-to-English mapping for {word!r}") from exc
    raise ValueError(f"Unsupported language: {language!r}")


def paper_decoder_pairs(language: str = "en") -> tuple[tuple[str, str], ...]:
    """Return the 3,336 unique pairs queried for decoder models in the paper."""
    dravnieks = words_for_language(DRAVNIEKS_WORDS_EN, language)
    odor_based = words_for_language(ODOR_BASED_WORDS_EN, language)
    pairs = {
        tuple(sorted(pair))
        for vocabulary in (dravnieks, odor_based)
        for pair in combinations(vocabulary, 2)
    }
    return tuple(sorted(pairs))


assert len(ENGLISH_TO_SWEDISH) == 96
assert len(ODOR_BASED_WORDS_EN) == 16
assert len(DRAVNIEKS_WORDS_EN) == 82
assert len(paper_decoder_pairs("en")) == 3336
