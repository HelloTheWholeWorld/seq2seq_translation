from config import ENG_PATH, FREN_PATH, ENG_TRAIN, ENG_TEST, ENG_VAL, FREN_TRAIN, FREN_VAL, FREN_TEST, ENG_DATA_PATH, FREN_DATA_PATH
from torchtext import data, datasets
import spacy

UNKNOWN_WORD = '<UNK>'
END_WORD = '<EOS>'
START_WORD = '<SOS>'

def read_data(num, max_length):
    f1 = open(ENG_PATH, 'r', encoding='utf-8')
    f2= open(FREN_PATH, 'r', encoding='utf-8')
    engs, frens = [], []
    count = 0
    while count < num:
        eng = f1.readline()
        fren = f2.readline()
        if len(eng.split()) > max_length or len(fren.split()) > max_length:
            continue
        engs.append(eng)
        frens.append(fren)
        count += 1
    f1.close()
    f2.close()
    return engs, frens

def build_new_data(sum_num=10000, max_length=10, train=0.9, test=0.05, val=0.05):
    engs, frens = read_data(sum_num, max_length)
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
    print('build new data successfully!\ntrain:{0}  test:{1}  val:{2}  max_length:{3}'.format(
        int(train * sum_num),
        int(test * sum_num),
        int(val * sum_num),
        max_length
        ))

def load_data(sum_num=30000, max_length=10):
    spacy_fr = spacy.load('fr_core_news_sm')
    spacy_en = spacy.load("en_core_web_sm")
    tokenize_eng = lambda text : [tok.text for tok in spacy_en.tokenizer(text)][::-1] #TODO：为什么是反序
    tokenize_fren = lambda text : [tok.text for tok in spacy_fr.tokenizer(text)]
    build_new_data(sum_num=sum_num, max_length=max_length)
    temp_tokenizer = lambda x : x.strip().split()

    # eng_field = data.Field(tokenize = tokenize_eng, 
    #         init_token = '<sos>', 
    #         eos_token = '<eos>', 
    #         lower = True)

    # fren_field = data.Field(tokenize = tokenize_fren, 
    #         init_token = '<sos>', 
    #         eos_token = '<eos>', 
    #         lower = True)
    
    # train, val, test = datasets.Multi30k.splits(exts = ('.de', '.en'), 
    #                                                 fields = (eng_field, fren_field))
    
    # eng_field.build_vocab(train.src, min_freq=3)
    # fren_field.build_vocab(train.trg, min_freq=3)
    # if True: return eng_field, fren_field, (train, val, test)



    eng_field = data.Field(
        tokenize=tokenize_eng,
        init_token = START_WORD, 
        eos_token = END_WORD
    )
    fren_field = data.Field(
        tokenize=tokenize_fren,
        init_token=START_WORD,
        eos_token=END_WORD
    )

    # eng_field = data.Field(sequential=True, # 序列化数据
    #                     use_vocab=True, # 确认使用词典
    #                     init_token=START_WORD,
    #                     eos_token=END_WORD,
    #                     fix_length=max_length, # 最大长度
    #                     tokenize=tokenize, # token方法
    #                     unk_token=UNKNOWN_WORD, # 未出现的词
    #                     batch_first=True, #是否先生成批次维度的张量
    #                     include_lengths=True # 返回填充的小批量的元祖和包含每个示例的列表
    #                     )
    # fren_field = data.Field(sequential=True, # 序列化数据
    #                     use_vocab=True, # 确认使用词典
    #                     init_token=START_WORD,
    #                     eos_token=END_WORD,
    #                     fix_length=max_length, # 最大长度
    #                     tokenize=tokenize, # token方法
    #                     unk_token=UNKNOWN_WORD,
    #                     batch_first = True,
    #                     include_lengths=True
    #                     )
    dataset = datasets.TranslationDataset(
        path='./data/small',
        exts=('.en','.fr'),
        fields=(eng_field, fren_field)
    )
    train, val, test = dataset.splits(exts=('.en','.fr'),
                                      fields=(eng_field, fren_field),
                                      path='./data/')
    print('len(train.examples)',len(train.examples))
    print('len(val.examples)',len(val.examples))
    print('len(test.examples)',len(test.examples))
    eng_field.build_vocab(train.src, min_freq=2)
    fren_field.build_vocab(train.trg, min_freq=2)
    print('len(src_field.vocab)', len(eng_field.vocab))
    print('len(trg_field.vocab)', len(fren_field.vocab))
    return eng_field, fren_field, (train, val, test)

# eng_field, fren_field, (train, val, test) = load_data()
# train_iterator, val_iterator, test_iterator = data.BucketIterator.splits((train, val, test), batch_size=16)
# for i, batch in enumerate(train_iterator):
#     print(batch.src.transpose(0, 1))
#     print(batch.trg.transpose(0, 1))
#     break

