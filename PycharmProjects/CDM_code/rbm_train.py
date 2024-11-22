from RBM import RBM
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim


batch_size = 2
n_epochs = 5

response = np.loadtxt('./data/response_8th_reduced.csv',delimiter=',')
init_hidden = np.loadtxt('./bert_base_mix.csv',delimiter=',')

lr = 0.01
n_hid = 9
n_vis = response.shape[1]

model = RBM(n_vis,n_hid,k=1)



nb_epoch = 10
for epoch in range(1, nb_epoch + 1):
    loss_ = []
    s = 0.
    train_op = optim.Adam(model.parameters(), lr)
    for id_user in range(0, nb_users - batch_size, batch_size):
        vk = response_set[id_user:id_user+batch_size]
        h0 = hidden_set[id_user:id_user+batch_size]

        #
        # v0 = response_set[id_user:id_user+batch_size]
        v0 = model.hidden_to_visible(h0)
        v, v_gibbs = model(h0)
        loss = model.free_energy(v) - model.free_energy(v_gibbs)
        loss_.append(loss.item())
        train_op.zero_grad()
        loss.backward()
        train_op.step()

    print('Epoch %d\t Loss=%.4f' % (epoch, np.mean(loss_)))



def train(model, response,init_hidden,n_epochs=20, lr=0.01):

    model.train()
    response_set = torch.FloatTensor(response)
    hidden_set = torch.FloatTensor(init_hidden)
    nb_users = (response_set.shape[1])
    #
    # train_loader = DataLoader(
    #     response_set,
    #     batch_size=batch_size,
    #     shuffle=False,
    # )
    for epoch in range(1, n_epochs + 1):
        loss_ = []
        train_op = optim.Adam(model.parameters(), lr)
        for id_user in range(0, nb_users - batch_size, batch_size):
            h0 = hidden_set[id_user:id_user + batch_size]
            v0 = response_set[id_user:id_user+batch_size]

            v0 = model.hidden_to_visible(h0)
            v, v_gibbs = model(h0)
            loss = model.free_energy(v) - model.free_energy(v_gibbs)
            loss_.append(loss.item())
            train_op.zero_grad()
            loss.backward()
            train_op.step()

        print('Epoch %d\t Loss=%.4f' % (epoch, np.mean(loss_)))
    return model
