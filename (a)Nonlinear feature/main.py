from param import parameter_parser
import load_data
from model import GraphEmbedding
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier,HistGradientBoostingClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

import evaluation_scores
import time
# import xgboost as xgb
import gpboost as gpb
#import catboost
# import lightgbm as lgb
import pandas as pd
def GMCB(n_fold):
    args = parameter_parser()
    dataset, ld_pairs = load_data.dataset(args)
 
    
    kf = KFold(n_splits = n_fold, shuffle = True)
    model = GraphEmbedding(args)
    
    ave_acc = 0
    ave_prec = 0
    ave_sens = 0
    ave_f1_score = 0
    ave_mcc = 0
    ave_auc = 0
    ave_auprc = 0
        
        for train_index, test_index in kf.split(ld_pairs):
            l_dmatix,train_ld_pairs,test_ld_pairs = load_data.L_Dmatix(ld_pairs,train_index,test_index)
            dataset['l_d']=l_dmatix
            score, lnc_fea, dis_fea = load_data.feature_representation(model, args, dataset)

            # # 保存 lnc_fea
            # pd.DataFrame(lnc_fea).to_csv(args.dataset_path +"lnc_fea.csv", index=False, header=False)
            #
            # # 保存 dis_fea
            # pd.DataFrame(dis_fea).to_csv(args.dataset_path +"dis_fea.csv", index=False, header=False)

            train_dataset = load_data.new_dataset(lnc_fea, dis_fea, train_ld_pairs)
            test_dataset = load_data.new_dataset(lnc_fea, dis_fea, test_ld_pairs)
            X_train, y_train = train_dataset[:,:-2], train_dataset[:,-1:]
            X_test, y_test = test_dataset[:,:-2], test_dataset[:,-1:]
            print(X_train.shape,X_test.shape)
            # clf = RandomForestClassifier(n_estimators=200,n_jobs=11,max_depth=8)
            clf = HistGradientBoostingClassifier( max_iter=400, max_depth=9)
            # clf = AdaBoostClassifier(n_estimators=200)
            # clf = xgb.XGBClassifier( max_depth=8, n_estimators=200 )

            # clf =MLPClassifier(hidden_layer_sizes=3)
            # clf = GradientBoostingClassifier(n_estimators=200, max_depth=9)
            # clf = gpb.GPBoostClassifier(n_estimators=200, max_depth=9, learning_rate=0.1)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_pred = y_pred
            y_prob = clf.predict_proba(X_test)
            y_prob = y_prob[:, 1]
            tp, fp, tn, fn, acc, prec, sens, f1_score, MCC, AUC,AUPRC = evaluation_scores.calculate_performace(len(y_pred), y_pred, y_prob, y_test)
            print('RF: \n  Acc = \t', acc, '\n  prec = \t', prec, '\n  sens = \t', sens, '\n  f1_score = \t', f1_score, '\n  MCC = \t', MCC, '\n  AUC = \t', AUC,'\n  AUPRC = \t', AUPRC)
            f.write('RF: \t  tp = \t'+ str(tp) + '\t fp = \t'+ str(fp) + '\t tn = \t'+ str(tn)+ '\t fn = \t'+ str(fn)+'\t  Acc = \t'+ str(acc)+'\t  prec = \t'+ str(prec)+ '\t  sens = \t'+str(sens)+'\t  f1_score = \t'+str(f1_score)+ '\t  MCC = \t'+str(MCC)+'\t  AUC = \t'+ str(AUC)+'\t  AUPRC = \t'+ str(AUPRC)+'\n')
            ave_acc += acc
            ave_prec += prec
            ave_sens += sens
            ave_f1_score += f1_score
            ave_mcc += MCC
            ave_auc += AUC
            ave_auprc  += AUPRC
            
        ave_acc /= n_fold
        ave_prec /= n_fold
        ave_sens /= n_fold
        ave_f1_score /= n_fold
        ave_mcc /= n_fold
        ave_auc /= n_fold
        ave_auprc /= n_fold
        print('Final: \t  tp = \t'+ str(tp) + '\t fp = \t'+ str(fp) + '\t tn = \t'+ str(tn)+ '\t fn = \t'+ str(fn)+'\t  Acc = \t'+ str(ave_acc)+'\t  prec = \t'+ str(ave_prec)+ '\t  sens = \t'+str(ave_sens)+'\t  f1_score = \t'+str(ave_f1_score)+ '\t  MCC = \t'+str(ave_mcc)+'\t  AUC = \t'+ str(ave_auc)+'\t  AUPRC = \t'+ str(ave_auprc)+'\n')
        f.write('Final: \t  tp = \t'+ str(tp) + '\t fp = \t'+ str(fp) + '\t tn = \t'+ str(tn)+ '\t fn = \t'+ str(fn)+'\t  Acc = \t'+ str(ave_acc)+'\t  prec = \t'+ str(ave_prec)+ '\t  sens = \t'+str(ave_sens)+'\t  f1_score = \t'+str(ave_f1_score)+ '\t  MCC = \t'+str(ave_mcc)+'\t  AUC = \t'+ str(ave_auc)+'\t  AUPRC = \t'+ str(ave_auprc)+'\n')
        

if __name__ == "__main__":
    
    n_fold = 5
    for i in range(1):
        GMCB(n_fold)
