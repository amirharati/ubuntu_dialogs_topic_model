# Introduction
In this project, I have used a small sub section of Ubuntu dialogs (section 4) to test topic modeling using two models:
* baseline LDA model.
* doc2vec clustering model.

# Requirements
* Python 3.6
* Numpy
* NLTK
* Gensim

# How to Use
There are few files that are used to train and test LDA and doc2vec models.
After training (see below), use display_top_topics_baseline.py to see the top topics for LDA and display_top_topics_doc2vec.py to see top topics for doc2vec model. Use topic_detector_lda_baseline.py and topic_detector_doc2vec.py to see the topic detected for a new tsv file.

*Example:*

python  topic_detector_lda_baseline.py data/dialogs/4/1.tsv

python  topic_detector_doc2vec.py data/dialogs/4/1.tsv

python  display_top_topics_baseline.py


*before training run:*

python data_prep.py

This will create a file name data/dialogs_4.txt that contains all dialogs for section 4 and used in the next steps.

# Baseline Model
Baseline model is a LDA with a simple NLP pipeline (e.g. remving stopwords, tfidf etc). To train the model:

*First create an iterable corpus object and store it:*

python create_corpus.py

This will save the corpus  in tmp directory.


*Train:*

python train_topic_model_lda_baseline.py


# Doc2Vec Model
For doc2vec model we first train the doc2vec model to convert docs into dense vectors. For this project, I choose a small dimension of 10. Then we convert the data into vectors and finally we use a clustering algorithm to find the clusters/topics.

*First create the corpus:*

python create_tagged_corpus.py

This save the corpus in tmp directory.


*Second extract features:*

python extract_feature_doc2vec.py

This will save the features in tmp directory.


*Third cluster:*

python cluster_doc2vec_features.py


Now you should have the trianed models in tmp directory.


# Samples outputs

## LDA top topics

**(py36) [bash]:python display_top_topics_baseline.py **

2018-03-18 17:49:08,912 : INFO : loaded corpus index from tmp/dialogs4-corpus.mm.index

2018-03-18 17:49:08,912 : INFO : initializing cython corpus reader from tmp/dialogs4-corpus.mm

2018-03-18 17:49:08,913 : INFO : accepted corpus with 268895 documents, 51147 features, 4530364 non-zero entries

2018-03-18 17:49:08,913 : INFO : loading LdaModel object from tmp/lda_topics.model

2018-03-18 17:49:08,914 : INFO : loading expElogbeta from tmp/lda_topics.model.expElogbeta.npy with mmap=None

2018-03-18 17:49:08,942 : INFO : setting ignored attribute state to None

2018-03-18 17:49:08,943 : INFO : setting ignored attribute dispatcher to None

2018-03-18 17:49:08,943 : INFO : setting ignored attribute id2word to None

2018-03-18 17:49:08,943 : INFO : loaded tmp/lda_topics.model

2018-03-18 17:49:08,943 : INFO : loading LdaState object from tmp/lda_topics.model.state

2018-03-18 17:49:09,085 : INFO : loaded tmp/lda_topics.model.state

2018-03-18 17:49:09,145 : INFO : topic #0 (0.010): 0.037*"upgrade" + 0.030*"update" + 0.024*"apt" + 0.024*"sources"

2018-03-18 17:49:09,145 : INFO : topic #45 (0.010): 0.059*"bit" + 0.024*"core" + 0.021*"intel" + 0.019*"yup"

2018-03-18 17:49:09,146 : INFO : topic #62 (0.010): 0.042*"kill" + 0.034*"ps" + 0.027*"process" + 0.021*"processes"

2018-03-18 17:49:09,146 : INFO : topic #96 (0.010): 0.031*"ping" + 0.020*"sessions" + 0.018*"dns" + 0.017*"pay"

2018-03-18 17:49:09,146 : INFO : topic #13 (0.010): 0.042*"ftp" + 0.030*"nick" + 0.028*"rpm" + 0.016*"alien"

2018-03-18 17:49:09,147 : INFO : topic #46 (0.010): 0.027*"ls" + 0.024*"dir" + 0.020*"automatix" + 0.017*"directories"

2018-03-18 17:49:09,147 : INFO : topic #57 (0.010): 0.027*"modules" + 0.024*"module" + 0.016*"cron" + 0.015*"modprobe"

2018-03-18 17:49:09,147 : INFO : topic #7 (0.010): 0.036*"noob" + 0.028*"screenshot" + 0.021*"split" + 0.018*"unknown"

2018-03-18 17:49:09,148 : INFO : topic #59 (0.010): 0.026*"suspend" + 0.025*"spanish" + 0.024*"hibernate" + 0.023*"force"

2018-03-18 17:49:09,148 : INFO : topic #3 (0.010): 0.033*"edition" + 0.027*"base" + 0.019*"banned" + 0.018*"hopefully"

**TOP TOPICS**

topic  0   upgrade-update-apt-sources

topic  45   bit-core-intel-yup

topic  62   kill-ps-process-processes

topic  96   ping-sessions-dns-pay

topic  13   ftp-nick-rpm-alien

topic  46   ls-dir-automatix-directories

topic  57   modules-module-cron-modprobe

topic  7   noob-screenshot-split-unknown

topic  59   suspend-spanish-hibernate-force

topic  3   edition-base-banned-hopefully



## doc2vec top topics

(py36) [bash]:python display_top_topics_doc2vec.py

**TOP TOPICS**

95  ---  http-paste-com-enter

66  ---  channel-offtopic-support-people

28  ---  upgrade-version-release-install

43  ---  channel-irc-join-client

1  ---  gnome-install-kde-desktop

50  ---  linux-channel-support-offtopic

5  ---  root-password-sudo-user

94  ---  wireless-card-get-work

10  ---  gnome-firefox-get-system

63  ---  install-boot-windows-drive


## topic detection

**tsv file**

(py36) [bash]:cat data/dialogs/4/1000.tsv

2012-09-25T14:48:00.000Z        Kingsy          so does anyone in here use the xorg-edgers ppa ? out of curiousity ?

2012-09-25T14:48:00.000Z        winxpvbox       Kingsy  ppas are unsupported 3rd party packages

2012-09-25T14:49:00.000Z        winxpvbox       Kingsy  if something goes wrong you are on your own

2012-09-25T14:49:00.000Z        Kingsy  winxpvbox       I know, I am asking if anyone in here uses it..

### LDA model
(py36) [bash]:python topic_detector_.py data/dialogs/4/1000.tsv

topic_detector_doc2vec.py       topic_detector_lda_baseline.py

(py36) [bash]:python topic_detector_lda_baseline.py  data/dialogs/4/1000.tsv

2018-03-18 17:52:11,142 : INFO : loaded corpus index from tmp/dialogs4-corpus.mm.index

2018-03-18 17:52:11,142 : INFO : initializing cython corpus reader from tmp/dialogs4-corpus.mm

2018-03-18 17:52:11,143 : INFO : accepted corpus with 268895 documents, 51147 features, 4530364 non-zero entries

2018-03-18 17:52:11,143 : INFO : loading TfidfModel object from tmp/tfidf.model

2018-03-18 17:52:11,194 : INFO : loaded tmp/tfidf.model

2018-03-18 17:52:11,195 : INFO : loading LdaModel object from tmp/lda_topics.model

2018-03-18 17:52:11,196 : INFO : loading expElogbeta from tmp/lda_topics.model.expElogbeta.npy with mmap=None

2018-03-18 17:52:11,225 : INFO : setting ignored attribute state to None

2018-03-18 17:52:11,225 : INFO : setting ignored attribute dispatcher to None

2018-03-18 17:52:11,225 : INFO : setting ignored attribute id2word to None

2018-03-18 17:52:11,226 : INFO : loaded tmp/lda_topics.model

2018-03-18 17:52:11,226 : INFO : loading LdaState object from tmp/lda_topics.model.state

2018-03-18 17:52:11,333 : INFO : loaded tmp/lda_topics.model.state

*TOPICS:*

topic  79   vmware-virtualbox-virtual-irssi  probablity: 0.2207199

topic  58   pidgin-gaim-msn-opera  probablity: 0.19551224

topic  54   headers-clock-compiling-operating  probablity: 0.11093143

topic  69   j-bootable-perfect-distribution  probablity: 0.094964825

topic  48   ssh-server-remote-secure  probablity: 0.08262363

topic  49   ppa-nfs-txt-classic  probablity: 0.048476804

topic  72   xorg-conf-x-resolution  probablity: 0.041110646

### doc2vec
(py36) [bash]:python topic_detector_doc2vec.py data/dialogs/4/1000.tsv

2018-03-18 17:54:17,310 : INFO : loading Doc2Vec object from tmp/doc2vec.model

2018-03-18 17:54:17,528 : INFO : loading vocabulary recursively from tmp/doc2vec.model.vocabulary.* with mmap=None

2018-03-18 17:54:17,528 : INFO : loading trainables recursively from tmp/doc2vec.model.trainables.* with mmap=None

2018-03-18 17:54:17,528 : INFO : loading wv recursively from tmp/doc2vec.model.wv.* with mmap=None

2018-03-18 17:54:17,528 : INFO : loading docvecs recursively from tmp/doc2vec.model.docvecs.* with mmap=None

2018-03-18 17:54:17,528 : INFO : loaded tmp/doc2vec.model

*TOPICS:*

topic  28   upgrade-version-release-install  probablity: 100.0

**tsv file**

(py36) [bash]:cat data/dialogs/4/100000.tsv

2008-11-26T06:50:00.000Z        |MUSE|          I just installed ubuntu-server. What would I need to install to get a graphical application to run over ssh, like: ssh -X 10.10.10.10 psp.

2008-11-26T06:52:00.000Z        |MUSE|          fiXXXerMet: this application does not have a command-line. :/

2008-11-26T06:53:00.000Z        n8tuser |MUSE|  -> your psp has to have an Xserver also

2008-11-26T06:54:00.000Z        n8tuser |MUSE|  -> provide a better information

### LDA model
(py36) [bash]:python topic_detector_lda_baseline.py  data/dialogs/4/100000.tsv

2018-03-18 17:55:19,132 : INFO : loaded corpus index from tmp/dialogs4-corpus.mm.index

2018-03-18 17:55:19,132 : INFO : initializing cython corpus reader from tmp/dialogs4-corpus.mm

2018-03-18 17:55:19,132 : INFO : accepted corpus with 268895 documents, 51147 features, 4530364 non-zero entries

2018-03-18 17:55:19,132 : INFO : loading TfidfModel object from tmp/tfidf.model

2018-03-18 17:55:19,183 : INFO : loaded tmp/tfidf.model

2018-03-18 17:55:19,183 : INFO : loading LdaModel object from tmp/lda_topics.model

2018-03-18 17:55:19,184 : INFO : loading expElogbeta from tmp/lda_topics.model.expElogbeta.npy with mmap=None

2018-03-18 17:55:19,203 : INFO : setting ignored attribute state to None

2018-03-18 17:55:19,203 : INFO : setting ignored attribute dispatcher to None

2018-03-18 17:55:19,203 : INFO : setting ignored attribute id2word to None

2018-03-18 17:55:19,203 : INFO : loaded tmp/lda_topics.model

2018-03-18 17:55:19,203 : INFO : loading LdaState object from tmp/lda_topics.model.state

2018-03-18 17:55:19,300 : INFO : loaded tmp/lda_topics.model.state

*TOPICS:*

topic  48   ssh-server-remote-secure  probablity: 0.39898354

topic  66   fonts-unity-es-font  probablity: 0.160864

topic  39   compiz-beryl-effects-fusion  probablity: 0.12017661

topic  13   ftp-nick-rpm-alien  probablity: 0.051770627

topic  72   xorg-conf-x-resolution  probablity: 0.051737484


### doc2vec
(py36) [bash]:python topic_detector_doc2vec.py data/dialogs/4/100000.tsv

2018-03-18 17:55:32,448 : INFO : loading Doc2Vec object from tmp/doc2vec.model

2018-03-18 17:55:32,656 : INFO : loading vocabulary recursively from tmp/doc2vec.model.vocabulary.* with mmap=None

2018-03-18 17:55:32,656 : INFO : loading trainables recursively from tmp/doc2vec.model.trainables.* with mmap=None

2018-03-18 17:55:32,656 : INFO : loading wv recursively from tmp/doc2vec.model.wv.* with mmap=None

2018-03-18 17:55:32,656 : INFO : loading docvecs recursively from tmp/doc2vec.model.docvecs.* with mmap=None

2018-03-18 17:55:32,656 : INFO : loaded tmp/doc2vec.model

*TOPICS:*

topic  6   server-ssh-desktop-x  probablity: 100.0


# TODO
* Develope metric and tools to directly compare the performance: By inspection, we can see doc2vec produce relatively interesting results but I have not conducted any scientific comparsion between two models.
* Doc2Vec model is very sensetive to some of its hyper-params. I find out we I train it for more passes the over results degrades. I think the model overfit due to small dataset. However, this needs more investigations.
* Tune the algorithms: I have not spend a lot of time to tune these algorithms. There are a lot of directions (e.g. hyper-params, NLP pipeline etc) to tune the performnace.
* Use sequential models: Both models are using bag of words. Using sequential models like LSTM (for classification) might be helpful.
* Train on large dataset: This subset of ubuntu dialog is relatively small and not very good for word2vec and doc2vec models.
* Use  averge word2vec model on pretrained Google News corpus: This might actually help sinc we will have good word2vec models.
