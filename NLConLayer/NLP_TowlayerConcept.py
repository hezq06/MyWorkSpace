
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import matplotlib
matplotlib.use('qt5agg')
# %matplotlib inline


# In[2]:


import numpy as np
import random
import copy
import matplotlib.pyplot as plt
from ncautil.ncalearn import *
from ncautil.seqgen import SeqGen
from ncautil.seqmultip import *
from ncautil.nlputil import NLPutil
import torch
import collections
from ncautil.datautil import *
import nltk
from nltk.tokenize import word_tokenize
import cProfile


# ## Getting Data From Text8 Corpus

# In[3]:


nVcab=10000
nlp=NLPutil()
nlp.get_data("text8_sub_aa")
w2id,id2w=nlp.build_vocab(Vsize=nVcab)
w2v_dict=nlp.build_w2v(mode="torchnlp",Nvac=nVcab)
subctm=nlp.build_textmat(nlp.sub_corpus)
# nlp.plot_txtmat(subctm.T,start=0,length=200,text=nlp.sub_corpus)
# nlp.plot_txtmat(subctm.T,start=0,length=200)
dataset=[]
for wrd in nlp.corpus:
    dataset.append(nlp.word_to_id.get(wrd,nlp.word_to_id["UNK"]))
print(len(dataset))
id_2_vec=dict([])
for k,v in w2v_dict.items():
    id_2_vec[nlp.word_to_id[k]]=v
nlpb=NLPutil()
nlpb.get_data("text8_sub_ab")
datasetb=[]
for wrd in nlpb.corpus:
    datasetb.append(nlp.word_to_id.get(wrd,nlp.word_to_id["UNK"]))
print(len(datasetb))
nlpb=NLPutil()
nlpb.get_data("text8_sub_ac")
datasetc=[]
for wrd in nlpb.corpus:
    datasetc.append(nlp.word_to_id.get(wrd,nlp.word_to_id["UNK"]))
print(len(datasetc))


# ## Tagging using nltk_tagger and transfer dataset

# In[4]:


startt = time.time()
pos_corpus=nltk.pos_tag(nlp.corpus)
endt = time.time()
print("Time used in training:", endt - startt)


# In[5]:


startt = time.time()
pos_corpusb=nltk.pos_tag(nlpb.corpus)
endt = time.time()
print("Time used in training:", endt - startt)


# In[6]:


pos_only=[item[1] for item in pos_corpusb]
nlp_pos=NLPutil()
nlp_pos.set_corpus(pos_only)
w2id_pos,id2w_pos=nlp_pos.build_vocab(Vsize=nVcab)
pos_only=[w2id_pos[item[1]] for item in pos_corpusb]


# In[7]:


pos_onlyb=[item[1] for item in pos_corpusb]
pos_onlyb=[w2id_pos[item[1]] for item in pos_corpusb]


# In[8]:


len(pos_only)


# In[9]:


lsize=len(w2id_pos)
rnn_pos=LSTM_NLP(lsize,20,lsize)
para={"seqtrain":True,"loss_clip":50,"cuda_flag":True}
ptA=PyTrain(pos_only, [lsize,lsize], rnn_pos, 4000, learning_rate=0.5e-2, batch=30, window=50, para=para)
ptA.run_training()


# In[10]:


ptA.data(pos_onlyb)
ptA.do_eval(1000)


# ## Anchoring Pretraining

# In[11]:


# Pure projectio seems not working, adding achoring pretraining
widl_sup=[]
labell_sup=[]
length=len(pos_corpus)
for tup in pos_corpus:
    try:
        wid=nlp.word_to_id[tup[0].lower()]
        label=nlp_pos.word_to_id[tup[1]]
        widl_sup.append(wid)
        labell_sup.append(label)
    except:
        pass
widl_sup=np.array(widl_sup)   
labell_sup=np.array(labell_sup)


# In[12]:


seqlab=True
lsize_in=lsize_out=len(nlp.word_to_id)
rnn_se=GRU_SerialCon_SharedAssociation(rnn_pos,lsize_in,10)


# In[13]:


dataset_sup={"dataset":widl_sup,"label":labell_sup}
para={"seqtrain":seqlab,"pre_training":True,"supervise_mode":True,"cuda_flag":True}
pt1=PyTrain(dataset_sup, [lsize_in,lsize_in], rnn_se, 1000, learning_rate=0.5e-2, batch=30, window=50, para=para)
# pt1.run_training()
cProfile.run('pt1.run_training()', 'restats')


# In[14]:


import pstats
p = pstats.Stats('restats')
p.strip_dirs()
p.sort_stats('cumulative')
p.print_stats()


# ## Serial Mode Training

# In[15]:


para={"seqtrain":seqlab,"loss_clip":2000,"cuda_flag":True}
pt_se=PyTrain(dataset, [lsize_in,lsize_in], rnn_se, 8000, learning_rate=0.5e-2, batch=30, window=50, para=para)
# pt_se.run_training()
cProfile.run('pt_se.run_training()', 'restats')


# In[16]:


import pstats
p = pstats.Stats('restats')
p.strip_dirs()
p.sort_stats('cumulative')
p.print_stats()


# In[17]:


pt_se.do_eval()


# In[18]:


inputlabl=pt_se.inputlabl
conceptl=pt_se.conceptl


# In[19]:


pstep=int(np.random.rand()*300)
pbatch=int(np.random.rand()*30)
sent=[]
pos=[]
for ii in range(len(inputlabl[pstep])):
    sent.append((ii,nlp.id_to_word[inputlabl[pstep][ii,pbatch]]))
    pos.append((ii,nlp_pos.id_to_word[np.argmax(conceptl[pstep][ii,pbatch,:])]))


# In[20]:


print(sent)


# In[21]:


print(pos)


# In[22]:


## POS Correct rate evaluation
cnt_tot=0
cnt_hit=0
for iit in range(500):
    pstep=int(np.random.rand()*300)
    pbatch=int(np.random.rand()*30)
    tres=[]
    for ii in range(len(inputlabl[pstep])):
        tres.append((inputlabl[pstep][ii,pbatch],np.argmax(conceptl[pstep][ii,pbatch,:])))
    sent=[nlp.id_to_word[wid] for wid in inputlabl[pstep][:,pbatch]]
    pos_sent=nltk.pos_tag(sent)
    nltk_res=[(nlp.word_to_id[item[0]],nlp_pos.word_to_id.get(item[1],None)) for item in pos_sent]
    for ii in range(len(tres)):
        assert tres[ii][0]==nltk_res[ii][0]
        cnt_tot=cnt_tot+1
        if tres[ii][1]==nltk_res[ii][1]:
            cnt_hit=cnt_hit+1
print("Hit rate:",cnt_hit/cnt_tot)


# In[23]:


# # Run 1: pre_training 500
# Hit rate: 0.73108 (Perp=200.53)
# # Run 2: pre_training 2000
# Hit rate: 0.80064 (Perp=223.53)
# # Run 3: pre_training 2000, no second training
# Hit rate: 0.84104
# # Maximum Baseline
# Hit rate: 0.84316
    
# # Inner feed back pre_training 500
# Hit rate: 0.72532 
# # Inner feed back pre_training 500, dot product
# Hit rate: 0.724 (Perp=205) 
# # Inner feed back pre_training 2000, dot product
# Hit rate: 0.80816 (Perp=228) 


# ## Fix layer1 and train second layer

# In[24]:


lsize2=30
rnn2=LSTM_NLP(lsize2,20,lsize2)
rnn_pl=GRU_TwoLayerCon_SharedAssociation(rnn_se,rnn2,lsize_in,10)


# In[25]:


para={"seqtrain":seqlab,"loss_clip":2000,"cuda_flag":True}
pt_pl=PyTrain(dataset, [lsize_in,lsize_in], rnn_pl, 8000, learning_rate=0.5e-2, batch=30, window=50, para=para)
# pt_pl.run_training()
cProfile.run('pt_pl.run_training()', 'restats')


# In[26]:


import pstats
p = pstats.Stats('restats')
p.strip_dirs()
p.sort_stats('cumulative')
p.print_stats()


# In[27]:


pt_pl.do_eval()


# In[28]:


pt_pl.do_eval(layer_sep_mode=1)


# In[29]:


wrd_spec_ent,wrd_spec_cnt=pt_pl.do_eval_conditioned_ave(min_sh=25, max_sh=1000,layer_sep_mode=None)


# In[30]:


wrd_spec_ent0,wrd_spec_cnt0=pt_pl.do_eval_conditioned_ave(min_sh=25, max_sh=1000,layer_sep_mode=0)
# cProfile.run('wrd_spec_ent0,wrd_spec_cnt0=pt_pl.do_eval_conditioned(step_eval=1000,layer_sep_mode=0)', 'restats')


# In[31]:


# import pstats
# p = pstats.Stats('restats')
# p.strip_dirs()
# p.sort_stats('cumulative')
# p.print_stats()


# In[32]:


# wrd_spec_ent1,wrd_spec_cnt1=pt_pl.do_eval_conditioned(step_eval=1000,layer_sep_mode=1)
wrd_spec_ent1,wrd_spec_cnt1=pt_pl.do_eval_conditioned_ave(min_sh=25, max_sh=1000,layer_sep_mode=1)


# In[33]:


plt.plot(wrd_spec_ent,"g")
plt.plot(wrd_spec_ent0,"b")
plt.plot(wrd_spec_ent1,"r")
plt.show()


# In[34]:


plt.plot(wrd_spec_cnt0)
plt.show()


# In[35]:


text=[]
for ii in range(len(wrd_spec_ent)):
    text.append(nlp.id_to_word[ii])
plot_data_w_text(wrd_spec_ent,text)


# In[ ]:


def plot_data_w_text(data,text):
    fig,ax=plt.subplots()
    fig=ax.plot(data)
    st,end=ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(st,end,1))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    labels=[item.get_text() for item in ax.get_xticklabels()]
    for ii in range(len(labels)):
        try:
            labels[ii]=str(text[ii])
        except:
            print("Failing label "+str(ii))
    ax.set_xticklabels(labels,rotation=70)
    plt.show()


# In[ ]:


id_to_con=pt_pl.rnn.build_concept_map(1000,nlp.prior,pt_pl.device,switch=0)


# In[ ]:


pl_conceptbubblecloud(id_to_con,nlp.id_to_word,nlp.prior,nlp.w2v_dict, pM=None)


# In[ ]:


maxmat=np.zeros((lsize_in,lsize))
for ii in range(len(dataset_sup["dataset"])):
    wid=dataset_sup["dataset"][ii]
    posid=dataset_sup["label"][ii]
    maxmat[wid,posid]=maxmat[wid,posid]+1
maxvec=np.argmax(maxmat,axis=-1)


# In[ ]:


## POS Correct rate evaluation
cnt_tot=0
cnt_hit=0
for iit in range(500):
    pstep=int(np.random.rand()*300)
    pbatch=int(np.random.rand()*30)
    tres=[]
    for ii in range(len(inputlabl[pstep])):
        tres.append((inputlabl[pstep][ii,pbatch],maxvec[inputlabl[pstep][ii,pbatch]]))
    sent=[nlp.id_to_word[wid] for wid in inputlabl[pstep][:,pbatch]]
    pos_sent=nltk.pos_tag(sent)
    nltk_res=[(nlp.word_to_id[item[0]],nlp_pos.word_to_id[item[1]]) for item in pos_sent]
    for ii in range(len(tres)):
        assert tres[ii][0]==nltk_res[ii][0]
        cnt_tot=cnt_tot+1
        if tres[ii][1]==nltk_res[ii][1]:
            cnt_hit=cnt_hit+1
print("Baseline rate:",cnt_hit/cnt_tot)


# ## Save / restore

# In[36]:


save_data(ptA,"NLP_TowlayerConcept_trained_ptA.pickle")
save_data(pt_se,"NLP_TowlayerConcept_trained_pt_se.pickle")
save_data(pt_pl,"NLP_TowlayerConcept_trained_pt_pl.pickle")
wrd_spec_data=[(wrd_spec_ent,wrd_spec_cnt),(wrd_spec_ent0,wrd_spec_cnt0),(wrd_spec_ent1,wrd_spec_cnt1)]
save_data(wrd_spec_data,"NLP_TowlayerConcept_trained_wrd_spec_data.pickle")


# In[ ]:


ptA=load_data("NLP_TowlayerConcept_trained_ptA.pickle")
pt_se=load_data("NLP_TowlayerConcept_trained_pt_se.pickle")
ptA=load_data("NLP_TowlayerConcept_trained_pt_pl.pickle")
wrd_spec_data=load_data("NLP_TowlayerConcept_trained_wrd_spec_data.pickle")

