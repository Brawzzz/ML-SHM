#============================================================================================================================#
#---------------------------------------------------------- IMPORT ----------------------------------------------------------#
#============================================================================================================================#
import numpy as np 

import CAE

import metrics
import data
import tools 
import setup as stp

    
#============================================================================================================================#
#---------------------------------------------------------- MAIN ------------------------------------------------------------#
#============================================================================================================================#
if __name__ == '__main__':

    args = tools.arg_parse()

    #---------------------------------------------
    if args.train is not None:

        config  = stp.get_config(config_path=args.train)
    
        stp.set_config(config_data=config)
        stp.configuration()

        #------------------------------
        if stp.DATASET == "UTAH":
            (X_train, X_test, scaler, labels_test)  = data.UTAH_load(nb_sample=5, path_index=3)

        elif stp.DATASET == "OGW":
            (X_train, X_test, scaler, labels_test)  = data.OGW_load(nb_cycles=5, path_index=3)

        else :
            raise ValueError(f"no dataset recognized : {stp.DATASET}")

        #------------------------------
        if stp.MODEL == "CAE":
            training_outputs = CAE.CAE_train(X_uncrack=X_train, X_crack=X_test)

        elif stp.MODEL == "MLP":
            raise NotImplementedError("MLP_train not implemented yet")

        else :
            training_outputs = CAE.CAE_train(X_uncrack=X_train, X_crack=X_test)

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
    # elif args.test is not None:

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

        config  = stp.get_config(config_path=args.plot)
            
        stp.set_config(config_data=config)
        stp.model()
        stp.datas()

        #------------------------------
        metrics_data = np.load(file=f"{stp.MODELS_DIR}{stp.MODEL_NAME}_metrics.npz")
        
        healthy_mse  = metrics_data["healthy_mse"]
        crack_mse    = metrics_data["crack_mse"]
        threshold    = float(metrics_data["threshold"])
        train_losses = metrics_data["train_losses"]

        model_metrics = metrics.Metrics.from_mse(healthy_mse  = healthy_mse, 
                                                 crack_mse    = crack_mse, 
                                                 name         = stp.MODEL_NAME)
        
        model_metrics.summary(thr=threshold, plot=True)

        fig = metrics.model_report(model_metrics, threshold, show=True)

    #---------------------------------------------
    else :
      raise ValueError(f"No command found : {args}")