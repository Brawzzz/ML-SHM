#============================================================================================================================#
#---------------------------------------------------------- IMPORT ----------------------------------------------------------#
#============================================================================================================================#
import numpy as np 

import MLP
import CAE

import data
import tools 
import validation as val
import setup as stp

    
#============================================================================================================================#
#---------------------------------------------------------- MAIN ------------------------------------------------------------#
#============================================================================================================================#
if __name__ == '__main__':

    args = tools.arg_parse()
    
    #---------------------------------------------
    if args.train is not None:

        config = stp.get_config(config_path=args.train)

        stp.set_config(config_data=config)
        stp.hyperparameters()

        #------------------------------
        if stp.DATASET == "UTAH":

            stp.UTAH_FILES                          = stp.UTAH_paths(nb_sample=stp.UTAH_FILES_SAMPLE)
            (X_train, X_test, scaler, labels_test)  = data.UTAH_data(stp.UTAH_FILES, path_index=3)

        elif stp.DATASET == "OGW":

            (X_train, X_test, scaler, labels_test)  = data.OGW_load(stp.UTAH_FILES, path_index=3)

        else :
            raise ValueError(f"no dataset recognized : {stp.DATASET}")

        #------------------------------
        if stp.MODEL == "CAE":
            training_outputs = CAE.CAE_train(X_uncrack=X_train, X_crack=X_test)

        elif stp.MODEL == "MLP":
            training_outputs = MLP.MLP_train(X_uncrack=X_train, X_crack=X_test)

        else :
            training_outputs = MLP.MLP_train(X_uncrack=X_train, X_crack=X_test)

        #------------------------------
        (model, train_losses)    = training_outputs[0], training_outputs[3]
        (threshold, recons)      = training_outputs[1], training_outputs[2]
        (healthy_mse, crack_mse) = training_outputs[4], training_outputs[5]

        tools.save_model(model, scaler, model_name=stp.MODEL_NAME)
        np.savez_compressed(f"./models/{stp.MODEL_NAME}_metrics.npz",
                            train_losses = train_losses,
                            healthy_mse  = healthy_mse,
                            crack_mse    = crack_mse,
                            threshold    = threshold)

    #---------------------------------------------
    elif args.test is not None:

        stp.set_config(config_data=stp.get_config(config_path=args.test))

        # model = tools.load_model(model_path   = "./models/CAE_UTAH_shm.pth",
        #                          model_type   = "PyTorch",
        #                          model_class  = CAE.ConvAutoEncoder,
        #                          scaler_path  = "./models/CAE_UTAH_shm_scaler.pkl",
        #                          model_kwargs = {"signal_length": 2000})

        
        # signal      = X_test[args.test]
        # threshold   = 1.0560758859483052e-05

        # (recons, erreurs, diagnostic) = CAE.CAE_inference(model, X_input=signal, n_threshold=threshold)

    #---------------------------------------------
    elif args.plot is not None:

        metrics = np.load(args.plot)
        val.validation_report(metrics["healthy_mse"], metrics["crack_mse"], float(metrics["threshold"]))

        # metrics = np.load(args.plot)

        # tools.model_perf(
        #     train_losses = metrics["train_losses"],
        #     healthy_mse  = metrics["healthy_mse"],
        #     crack_mse    = metrics["crack_mse"],
        #     threshold    = metrics["threshold"]
        # )

    #---------------------------------------------
    # else :
    #   print(f"No command found")