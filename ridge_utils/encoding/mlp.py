

# def get_model(args):
#     if args.encoding_model == 'mlp':
#         return NeuralNetRegressor(
#             encoding_models.MLP(
#                 dim_inputs=stim_train_delayed.shape[1],
#                 dim_hidden=args.mlp_dim_hidden,
#                 dim_outputs=resp_train.shape[1]
#             ),
#             max_epochs=3000,
#             lr=1e-5,
#             optimizer=torch.optim.Adam,
#             callbacks=[EarlyStopping(patience=30)],
#             iterator_train__shuffle=True,
#             # device='cuda',
#         )

# elif args.encoding_model == 'mlp':
#     stim_train_delayed = stim_train_delayed.astype(np.float32)
#     resp_train = resp_train.astype(np.float32)
#     stim_test_delayed = stim_test_delayed.astype(np.float32)
#     net = get_model(args)
#     net.fit(stim_train_delayed, resp_train)
#     preds = net.predict(stim_test_delayed)
#     corrs_test = []
#     for i in range(preds.shape[1]):
#         corrs_test.append(np.corrcoef(resp_test[:, i], preds[:, i])[0, 1])
#     corrs_test = np.array(corrs_test)
#     r[corrs_key_test] = corrs_test
#     model_params_to_save = {
#         'weights': net.module_.state_dict(),
#     }
# torch.save(net.module_.state_dict(), join(save_dir, 'weights.pt'))
