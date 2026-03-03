import stanza

def init_stanza():
    stanza.download("uk")
    nlp = stanza.Pipeline(
        lang="uk",
        processors="tokenize,mwt,pos,lemma",
        use_gpu=False
    )
    return nlp


def extract_ling_features(text, nlp):
    doc = nlp(text)

    lemmas = []
    pos_tags = []

    for sent in doc.sentences:
        for word in sent.words:
            lemmas.append(word.lemma)
            pos_tags.append(word.upos)

    return {
        "lemma_text": " ".join(lemmas),
        "pos_seq": pos_tags
    }


def filter_pos(lemmas, pos_tags, allowed=("NOUN", "ADJ")):
    filtered = [
        lemma for lemma, pos in zip(lemmas, pos_tags)
        if pos in allowed
    ]
    return " ".join(filtered)