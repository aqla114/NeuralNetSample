import numpy as np

def make_minibatch(datas_x, datas_t, batch_size):
    """
    datas_xにはdata配列を、datas_tにはその正解ラベル（ワンホット）配列を入れる。
    datas_x[mask], datas_t[mask] (それぞれbatch_size組のデータ)を返す。
    """
    train_size = datas.shape[0]
    batch_mask = np.random.choice(train_size, batch_size)
    
    return datas_x[batch_mask], datas_t[batch_mask]