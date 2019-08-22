import numpy as np
import pandas as pd
import opencc as oc
import jieba

LENS = 30

def read2df(mnt_txt):
    cc = oc.OpenCC("t2s")
    with open(mnt_txt,"r", encoding="utf-8") as f:
        data = f.read()
    data_list = data.split("\n")
    eng_list,chn_list = [],[]
    df = pd.DataFrame()
    for dl in data_list[:-1]:
        dls = dl.split("\t")
        #print(dls)
        eng_list.append(split_dot(dls[0]))
        chn_list.append(cc.convert(dls[1]))
    df["eng"] = eng_list
    df["chn"] = chn_list
    print(df.head(5))
    df.to_csv("cmn.csv",index=None)
    print("save csv")
    return(df)

def split_dot(strs,dots=", . ! ?"):
    for d in dots.split(" "):
        #print(d)
        strs = strs.replace(d," "+d)
        #print(strs)
    return(strs)

def get_eng_dicts(datas):
    w_all_dict = {}
    for sample in datas:
        for token in sample.split(" "):
            if token not in w_all_dict.keys():
                w_all_dict[token] = 1
            else:
                w_all_dict[token] += 1
 
    sort_w_list = sorted(w_all_dict.items(),  key=lambda d: d[1], reverse=True)


    w_keys = [x for x,_ in sort_w_list[:7000-2]]
    w_keys.insert(0,"<PAD>")
    w_keys.insert(0,"<UNK>")
    
 
    w_dict = { x:i for i,x in enumerate(w_keys) }
    i_dict = { i:x for i,x in enumerate(w_keys) }
    return w_dict,i_dict

def get_chn_dicts(datas):
    w_all_dict = {}
    for sample in datas:
        for token in jieba.cut(sample):
            if token not in w_all_dict.keys():
                w_all_dict[token] = 1
            else:
                w_all_dict[token] += 1
 
    sort_w_list = sorted(w_all_dict.items(),  key=lambda d: d[1], reverse=True)

    w_keys = [x for x,_ in sort_w_list[:10000-4]]
    w_keys.insert(0,"<EOS>")
    w_keys.insert(0,"<GO>")
    w_keys.insert(0,"<PAD>")
    w_keys.insert(0,"<UNK>")
    w_dict = { x:i for i,x in enumerate(w_keys) }
    i_dict = { i:x for i,x in enumerate(w_keys) }
    return w_dict,i_dict
 
def get_val(keys,dicts):
    if keys in dicts.keys():
        val = dicts[keys]
    else:
        keys = "<UNK>"
        val = dicts[keys]
    return(val)

def padding(lists,lens=LENS):
    list_ret = []
    for l in lists:
        
        while(len(l)<lens):
            l.append(1)

        if len(l)>lens:
            l = l[:lens]
        list_ret.append(l)
    
    return(list_ret)

# =======预定义模型参数========
EN_VOCAB_SIZE = 7000
CH_VOCAB_SIZE = 10000
HIDDEN_SIZE = 256

LEARNING_RATE = 0.001
BATCH_SIZE = 50
EPOCHS = 100

# ======================================keras model==================================
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding,CuDNNLSTM
from keras.optimizers import Adam
import numpy as np

def get_model():
    # ==============encoder=============
    encoder_inputs = Input(shape=(None,))
    emb_inp = Embedding(output_dim=128, input_dim=EN_VOCAB_SIZE)(encoder_inputs)
    encoder_h1, encoder_state_h1, encoder_state_c1 = CuDNNLSTM(HIDDEN_SIZE, return_sequences=True, return_state=True)(emb_inp)
    encoder_h2, encoder_state_h2, encoder_state_c2 = CuDNNLSTM(HIDDEN_SIZE, return_state=True)(encoder_h1)

    # ==============decoder=============
    decoder_inputs = Input(shape=(None,))

    emb_target = Embedding(output_dim=128, input_dim=CH_VOCAB_SIZE)(decoder_inputs)
    lstm1 = CuDNNLSTM(HIDDEN_SIZE, return_sequences=True, return_state=True)
    lstm2 = CuDNNLSTM(HIDDEN_SIZE, return_sequences=True, return_state=True)
    decoder_dense = Dense(CH_VOCAB_SIZE, activation='softmax')

    decoder_h1, _, _ = lstm1(emb_target, initial_state=[encoder_state_h1, encoder_state_c1])
    decoder_h2, _, _ = lstm2(decoder_h1, initial_state=[encoder_state_h2, encoder_state_c2])
    decoder_outputs = decoder_dense(decoder_h2)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # encoder模型和训练相同
    encoder_model = Model(encoder_inputs, [encoder_state_h1, encoder_state_c1, encoder_state_h2, encoder_state_c2])

    # 预测模型中的decoder的初始化状态需要传入新的状态
    decoder_state_input_h1 = Input(shape=(HIDDEN_SIZE,))
    decoder_state_input_c1 = Input(shape=(HIDDEN_SIZE,))
    decoder_state_input_h2 = Input(shape=(HIDDEN_SIZE,))
    decoder_state_input_c2 = Input(shape=(HIDDEN_SIZE,))

    # 使用传入的值来初始化当前模型的输入状态
    decoder_h1, state_h1, state_c1 = lstm1(emb_target, initial_state=[decoder_state_input_h1, decoder_state_input_c1])
    decoder_h2, state_h2, state_c2 = lstm2(decoder_h1, initial_state=[decoder_state_input_h2, decoder_state_input_c2])
    decoder_outputs = decoder_dense(decoder_h2)

    decoder_model = Model([decoder_inputs, decoder_state_input_h1, decoder_state_input_c1, decoder_state_input_h2, decoder_state_input_c2], 
                        [decoder_outputs, state_h1, state_c1, state_h2, state_c2])


    return(model,encoder_model,decoder_model)




import keras.backend as K
from keras.models import load_model
 
def my_acc(y_true, y_pred):
    acc = K.cast(K.equal(K.max(y_true,axis=-1),K.cast(K.argmax(y_pred,axis=-1),K.floatx())),K.floatx())
    return acc


Train = False

if __name__ == "__main__":
    df = read2df("cmn-eng/cmn.txt")
    eng_dict,id2eng = get_eng_dicts(df["eng"])
    chn_dict,id2chn = get_chn_dicts(df["chn"])
    print(list(eng_dict.keys())[:20])
    print(list(chn_dict.keys())[:20])

    enc_in = [[get_val(e,eng_dict) for e in eng.split(" ")] for eng in df["eng"]]
    dec_in = [[get_val("<GO>",chn_dict)]+[get_val(e,chn_dict) for e in jieba.cut(eng)]+[get_val("<EOS>",chn_dict)] for eng in df["chn"]]
    dec_out = [[get_val(e,chn_dict) for e in jieba.cut(eng)]+[get_val("<EOS>",chn_dict)] for eng in df["chn"]]

    enc_in_ar = np.array(padding(enc_in,32))
    dec_in_ar = np.array(padding(dec_in,30))
    dec_out_ar = np.array(padding(dec_out,30))

    #dec_out_ar = covt2oh(dec_out_ar)


    
    if Train:


        model,encoder_model,decoder_model = get_model()

        model.load_weights('e2c1.h5')

        opt = Adam(lr=LEARNING_RATE, beta_1=0.9, beta_2=0.99, epsilon=1e-08)
        model.compile(optimizer=opt, loss='sparse_categorical_crossentropy',metrics=[my_acc])
        model.summary()
        print(dec_out_ar.shape)
        model.fit([enc_in_ar, dec_in_ar], np.expand_dims(dec_out_ar,-1),
                batch_size=50,
                epochs=64,
                initial_epoch=56,
                validation_split=0.1)
        model.save('e2c1.h5')
        encoder_model.save("enc1.h5")
        decoder_model.save("dec1.h5")
    
    else:


        encoder_model,decoder_model = load_model("enc1.h5",custom_objects={"my_acc":my_acc}),load_model("dec1.h5",custom_objects={"my_acc":my_acc})

        for k in range(16000-20,16000):
            test_data = enc_in_ar[k:k+1]
            h1, c1, h2, c2 = encoder_model.predict(test_data)
            target_seq = np.zeros((1,1))
            
            outputs = []
            target_seq[0, len(outputs)] = chn_dict["<GO>"]
            while True:
                output_tokens, h1, c1, h2, c2 = decoder_model.predict([target_seq, h1, c1, h2, c2])
                sampled_token_index = np.argmax(output_tokens[0, -1, :])
                #print(sampled_token_index)
                outputs.append(sampled_token_index)
                #target_seq = np.zeros((1, 30))
                target_seq[0, 0] = sampled_token_index
                #print(target_seq)
                if sampled_token_index == chn_dict["<EOS>"] or len(outputs) > 28: break
            
            print("> "+df["eng"][k])
            print("< "+' '.join([id2chn[i] for i in outputs[:-1]]))
            print()


#Save model
#model.save('s2s.h5')








    


    
