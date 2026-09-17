"""Verbatim prompt templates used by the paper's model pipelines."""


BERT_PROMPTS = {
    "en": {
        "prompt0": "Research suggests that exposure to high levels of {word} may have adverse effects on human health.",
        "prompt1": "I could clearly detect the {word} scent in the mixture of smells around me.",
        "prompt2": "Among the various fragrances, the {word} smell stood out prominently and was hard to miss.",
        "prompt3": "The olfactory experience was dominated by the characteristic {word} scent that lingered in the air.",
        "prompt4": "As I took a deep breath, the distinct aroma of {word} filled my nostrils, overwhelming my senses with its pungent fragrance.",
        "prompt5": "I breathed in the unique smell of {word} its unmistakable fragrance filling my senses. [SEP] I closed my eyes and was transported by the intriguing aroma and memories it brought.",
        "prompt6": "The scent of {word} was so powerful that it engulfed me, immersing me in its unique and unforgettable olfactory experience.",
    },
    "sv": {
        "prompt0": "Forskning tyder på att exponering för höga nivåer av {word} kan ha skadliga effekter på människors hälsa.",
        "prompt1": "Jag kunde tydligt upptäcka doften av {word} i blandningen av lukter omkring mig.",
        "prompt2": "Bland de olika dofterna var lukten av {word} framträdande och svår att missa.",
        "prompt3": "Doftupplevelsen dominerades av den karakteristiska doften av {word} som hängde kvar i luften.",
        "prompt4": "När jag tog ett djupt andetag fyllde den distinkta aromen av {word} mina näsborrar och överväldigade mina sinnen med sin skarpa karaktär.",
        "prompt5": "Jag andades in den unika doften av {word} och dess omisskännliga doft som fyllde mina sinnen. [SEP] Jag slöt ögonen och transporterades av den spännande doften och de minnen den gav.",
        "prompt6": "Doften av {word} var så kraftfull att den uppslukade mig och försjönk mig i dess unika och oförglömliga doftupplevelse.",
    },
}

DECODER_PROMPTS = {
    "en": (
        "Based on general knowledge and human perception of scents, estimate the similarity score between "
        "the two odors provided. Rate the similarity on a scale from 0 to 1, with 0 being completely "
        "dissimilar and 1 being identical.\n"
        "Please keep the answer concise and round to three decimal points.\n"
        "Odors: {word1} and {word2}\n"
        "Similarity score is:"
    ),
    "sv": (
        "Föreställ dig att du har data från ett experiment där människor bedömde likheten mellan två "
        "distinkta lukter. Baserat på din förståelse av mänsklig perception, förutsäg likhetspoängen "
        "mellan följande lukter på en skala från 0 till 1, där 0 betyder helt olik och 1 betyder identisk.\n"
        "Ange ditt svar upp till 3 decimaler och ange bara din poäng.\n"
        "Lukter: {word1} och {word2}\n"
        "Ditt likhetspoäng:"
    ),
}

DECODER_PROMPT_ID = "paper_similarity_v1"


def format_decoder_prompt(word1: str, word2: str, language: str = "en") -> str:
    try:
        template = DECODER_PROMPTS[language]
    except KeyError as exc:
        raise ValueError(f"Unsupported prompt language: {language!r}") from exc
    return template.format(word1=word1, word2=word2)
