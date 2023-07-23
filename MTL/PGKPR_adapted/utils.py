vocab_path = 'mscoco.vocab'

PAD_TAG = '[pad]'  # 0
UNK_TAG = '[unk]'
UN_KEY_TAG = '[unkey]'
START_TAG = '<s>'
END_TAG = '</s>'
SEP_TAG = '<sep>'

POS_TAG = ['[VB]',
           '[NNS]',
           '[RBR]',
           '[DT]',
           '[IN]',
           '[WRB]',
           '[VBZ]',
           '[NNP]',
           '[CD]',
           '[(]',
           '[NN]',
           '[VBG]',
           '[``]',
           '[PRP$]',
           "['']",
           '[VBD]',
           '[PDT]',
           '[CC]',
           '[POS]',
           '[LS]',
           '[$]',
           '[JJ]',
           '[.]',
           '[,]',
           '[#]',
           '[RB]',
           '[RP]',
           '[:]',
           '[VBP]',
           '[MD]',
           '[PRP]',
           '[)]',
           '[VBN]',
           '[FW]',
           '[SYM]']



def loadVocab(vocab_fpath, padTag=PAD_TAG, oov_mark=False, lowercase=True):
    vocab = [PAD_TAG, UNK_TAG, START_TAG, END_TAG, SEP_TAG, UN_KEY_TAG] + [i.lower() for i in POS_TAG]
    if not lowercase:
        vocab = [PAD_TAG, UNK_TAG, START_TAG, END_TAG, SEP_TAG, UN_KEY_TAG] + [i for i in POS_TAG]
    # if oov_mark:
    #     vocab += OOV_MAK
    vocab += [line.split()[0] for line in open(vocab_fpath, 'r', encoding="utf8").read().splitlines()]

    token2idx = {token: idx for idx, token in enumerate(vocab)}
    idx2token = {idx: token for idx, token in enumerate(vocab)}
    assert idx2token[0] == PAD_TAG
    return token2idx, idx2token