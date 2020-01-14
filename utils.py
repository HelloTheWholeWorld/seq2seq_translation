from config import ENG_PATH, FREN_PATH, ENG_TRAIN, ENG_TEST, ENG_VAL, FREN_TRAIN, FREN_VAL, FREN_TEST, ENG_DATA_PATH, FREN_DATA_PATH
from torchtext import data, datasets


UNKNOWN_WORD = '<UNK>'
END_WORD = '<EOS>'
START_WORD = '<SOS>'

def read_data(num):
    f1 = open(ENG_PATH, 'r', encoding='utf-8')
    f2= open(FREN_PATH, 'r', encoding='utf-8')
    engs, frens = [], []
    for _ in range(num):
        engs.append(f1.readline())
        frens.append(f2.readline())
    f1.close()
    f2.close()
    return engs, frens

def build_new_data(sum_num=30000, train=0.9, test=0.05, val=0.05):
    engs, frens = read_data(sum_num)
    with open(ENG_DATA_PATH, 'w+', encoding='utf-8') as f:
        for eng in engs:
            f.write(eng)
    with open(FREN_DATA_PATH, 'w+', encoding='utf-8') as f:
        for fren in frens:
            f.write(fren)
    lindex = 0
    rindex = int(train * sum_num)
    with open(ENG_TRAIN, 'w+', encoding='utf-8') as f:
        for eng in engs[lindex : rindex]:
            f.write(eng)
    with open(FREN_TRAIN, 'w+', encoding='utf-8') as f:
        for fren in frens[lindex : rindex]:
            f.write(fren)
    lindex += int(train * sum_num)
    rindex += int(test * sum_num)
    with open(ENG_TEST, 'w+', encoding='utf-8') as f:
        for eng in engs[lindex : rindex]:
            f.write(eng)
    with open(FREN_TEST, 'w+', encoding='utf-8') as f:
        for fren in frens[lindex : rindex]:
            f.write(fren)
    lindex += int(test * sum_num)
    rindex += int(val * sum_num)
    with open(ENG_VAL, 'w+', encoding='utf-8') as f:
        for eng in engs[lindex : rindex]:
            f.write(eng)
    with open(FREN_VAL, 'w+', encoding='utf-8') as f:
        for fren in frens[lindex : rindex]:
            f.write(fren)

build_new_data()

def load_data(max_length=10):
    tokenize = lambda x : x.split()
    # eng_field = data.Field(eos_token="<eos>",
    #             include_lengths=True, batch_first=True)
    # fren_field = data.Field(init_token="<sos>",
    #             eos_token="<eos>", include_lengths=True, batch_first=True)
    eng_field = data.Field(sequential=True, # 序列化数据
                        use_vocab=True, # 确认使用词典
                        init_token=START_WORD,
                        eos_token=END_WORD,
                        fix_length=max_length, # 最大长度
                        tokenize=tokenize, # token方法
                        unk_token=UNKNOWN_WORD, # 未出现的词
                        batch_first=True, #是否先生成批次维度的张量
                        include_lengths=True # 返回填充的小批量的元祖和包含每个示例的列表
                        )
    fren_field = data.Field(sequential=True, # 序列化数据
                        use_vocab=True, # 确认使用词典
                        init_token=START_WORD,
                        eos_token=END_WORD,
                        fix_length=max_length, # 最大长度
                        tokenize=tokenize, # token方法
                        unk_token=UNKNOWN_WORD,
                        batch_first = True,
                        include_lengths=True
                        )
    dataset = datasets.TranslationDataset(
        path='./data/small',
        exts=('.en','.fr'),
        fields=(eng_field, fren_field)
    )
    train, val, test = dataset.splits(exts=('.en','.fr'), fields=(eng_field, fren_field),
                                    path='./data/')
    eng_field.build_vocab(train.src)
    fren_field.build_vocab(train.trg)
    print(len(eng_field.vocab), len(fren_field.vocab))
    return eng_field, fren_field, (train, val, test)

