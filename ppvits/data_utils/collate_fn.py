import paddle


class TextAudioSpeakerCollate(object):
    """ Zero-pads model inputs and targets
    """

    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        """Collate's training batch from normalized text, audio and speaker identities
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized, sid]
        """
        # Right zero-pad all one-hot text sequences to max input length
        ids_sorted_decreasing = paddle.argsort(paddle.to_tensor([x[1].shape[1] for x in batch], dtype=paddle.int64),
                                               axis=0, descending=True)
        max_text_len = max([len(x[0]) for x in batch])
        max_spec_len = max([x[1].shape[1] for x in batch])
        max_wav_len = max([x[2].shape[1] for x in batch])

        text_lengths = paddle.zeros([len(batch)], dtype=paddle.int64)
        spec_lengths = paddle.zeros([len(batch)], dtype=paddle.int64)
        wav_lengths = paddle.zeros([len(batch)], dtype=paddle.int64)
        sid = paddle.zeros([len(batch)], dtype=paddle.int64)

        text_padded = paddle.empty(shape=[len(batch), max_text_len], dtype=paddle.int64)
        spec_padded = paddle.empty(shape=[len(batch), batch[0][1].shape[0], max_spec_len], dtype=paddle.float32)
        wav_padded = paddle.empty(shape=[len(batch), 1, max_wav_len], dtype=paddle.float32)

        text_padded.zero_()
        spec_padded.zero_()
        wav_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]
            text = row[0]
            text_padded[i, :text.shape[0]] = text
            text_lengths[i] = text.shape[0]
            spec = row[1]
            spec_padded[i, :, :spec.shape[1]] = spec
            spec_lengths[i] = spec.shape[1]
            wav = row[2]
            wav_padded[i, :, :wav.shape[1]] = wav
            wav_lengths[i] = wav.shape[1]
            sid[i] = row[3]
        if self.return_ids:
            return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, sid, ids_sorted_decreasing
        return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, sid
