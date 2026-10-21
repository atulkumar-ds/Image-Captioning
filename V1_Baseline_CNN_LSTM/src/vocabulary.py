from collections import Counter


def build_vocabulary(caption_dict):
    all_words = []

    for image_name in caption_dict:
        for caption in caption_dict[image_name]:
            words = caption.split()
            all_words.extend(words)

    words_count = Counter(all_words)

    return words_count


if __name__ == "__main__":
    from dataset import load_captions, clean_all_captions

    captions_dict = load_captions()
    captions_dict = clean_all_captions(captions_dict)

    word_counts = build_vocabulary(captions_dict)

    print("Unique Words:", len(word_counts))

    print("\n20 Most Common Words:")

    for word, count in word_counts.most_common(20):
        print(word, ":", count)

    for threshold in [1, 2, 3, 5, 10]:
        count = sum(
            1 for frequency in word_counts.values()
            if frequency >= threshold
        )

        print(f"Words occurring >= {threshold} times: {count}")
