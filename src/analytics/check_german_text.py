from src.util import load_meta_data


def check_sentences():
    meta_data, _ = load_meta_data("Zivadiliring_enriched_DID_DE_included.txt")
    count = 0
    for entry in meta_data:
        length = len(entry.de_text)

        if length > 390:
            print(f"{len(entry.de_text)}: {entry.de_text}")
            count += 1

    print(count)

if __name__ == '__main__':
    check_sentences()