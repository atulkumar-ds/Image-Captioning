def find_max_length(caption_dict):
    max_length = 0
    for image_name in caption_dict:
        for caption in caption_dict[image_name]:
            length = len(caption.split())
            if length > max_length:
                max_length = length
    return max_length


def pad_sequence(
    sequence,
    max_length,
    padding_value=0,
    pad_at="post",
    preserve_end_token=False,
    end_token=None
):
    """
    Pads or truncates a sequence to guarantee exact length of max_length.

    Parameters
    ----------
    sequence : list or array-like
        The sequence of token indices to pad or truncate.
    max_length : int
        The exact target length.
    padding_value : int, default 0
        The integer value to use for padding.
    pad_at : str, default 'post'
        Where to pad: 'post' (append padding at the end) or 'pre'.
    preserve_end_token : bool, default False
        If True and len(sequence) > max_length, preserves the end token
        (either the last token of sequence or end_token if specified)
        at index max_length - 1.
    end_token : int or None, default None
        Explicit token index representing the <end> marker. If specified and
        preserve_end_token is True, this value is placed at the final position
        upon truncation.

    Returns
    -------
    list
        Padded or truncated sequence of exact length max_length.
    """
    if not isinstance(sequence, list):
        sequence = list(sequence)

    if len(sequence) > max_length:
        if max_length <= 0:
            return []
        if preserve_end_token:
            if end_token is not None:
                chosen_end = end_token
            elif len(sequence) > 0:
                chosen_end = sequence[-1]
            else:
                chosen_end = padding_value
            return sequence[:max_length - 1] + [chosen_end]
        return sequence[:max_length]
    elif len(sequence) == max_length:
        return list(sequence)
    else:
        pad_count = max_length - len(sequence)
        if pad_at == "pre":
            return [padding_value] * pad_count + sequence
        return sequence + [padding_value] * pad_count


def create_training_sequences(sequence, max_length):
    input_sequences = []
    target_words = []

    for i in range(1, len(sequence)):
        input_sequence = sequence[:i]
        target_word = sequence[i]

        input_sequence = pad_sequence(
            input_sequence,
            max_length
        )

        input_sequences.append(input_sequence)
        target_words.append(target_word)

    return input_sequences, target_words


if __name__ == "__main__":
    from dataset import load_captions, clean_all_captions
    from vocabulary import build_vocabulary
    from tokenizer import create_tokenizer, encode_caption

    caption_dict = load_captions()
    caption_dict = clean_all_captions(caption_dict)

    max_length = find_max_length(caption_dict)
    print("Maximum Caption Length:", max_length)
