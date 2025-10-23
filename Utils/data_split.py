import pandas as pd
import numpy as np
import torch



def data_split(args, synergy, rd_seed=0):
    synergy_pos = pd.DataFrame([i for i in synergy if i[3] == 1])
    synergy_neg = pd.DataFrame([i for i in synergy if i[3] == 0])
    # -----split synergy into 5CV,test set
    train_size = 0.9
    synergy_cv_pos, synergy_test_pos = np.split(np.array(synergy_pos.sample(frac=1, random_state=rd_seed)),
                                                [int(train_size * len(synergy_pos))])
    synergy_cv_neg, synergy_test_neg = np.split(np.array(synergy_neg.sample(frac=1, random_state=rd_seed)),
                                                [int(train_size * len(synergy_neg))])
    # --CV set
    synergy_cv_data = np.concatenate((np.array(synergy_cv_neg), np.array(synergy_cv_pos)), axis=0)
    # --test set
    synergy_test = np.concatenate((np.array(synergy_test_neg), np.array(synergy_test_pos)), axis=0)
    pd.DataFrame(synergy_test).to_csv('test.csv')
    np.random.shuffle(synergy_cv_data)
    np.random.shuffle(synergy_test)
    # np.savetxt(path + 'test_y_true.txt', synergy_test[:, 3])
    test_label = torch.from_numpy(np.array(synergy_test[:, 3], dtype='float32')).to(args.cuda)
    test_ind = torch.from_numpy(synergy_test).to(args.cuda)
    return synergy_cv_data, test_ind, test_label




