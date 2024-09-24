
from Model.unsupHash_bert import unsupHash_bert



if __name__ == "__main__":
    argparser = unsupHash_bert.get_model_specific_argparser()
    hparams = argparser.parse_args()
    model = unsupHash_bert(hparams)
    if hparams.continue_training:
        model.load(hparams)
        print('Loaded model with: %s' % model.flag_hparams())
        model.hparams = hparams
        model.run_training_sessions()
    if hparams.train:
        model.run_training_sessions()
    else:
        model.load()
        model.hparams.hashing_alpha =hparams.hashing_alpha

        print('Loaded model with: %s' % model.flag_hparams())
        model.hparams = hparams
        # model.run_coarse_topN(N=500)
        if hparams.is_coarse:
            val_perf, test_perf = model.run_coarse_test(True)
        else:
            val_perf, test_perf = model.run_test()
            # val_perf, test_perf = model.run_coarse_test(True)

        print('Val:  {:8.2f}'.format(val_perf))
        print('Test: {:8.2f}'.format(test_perf))