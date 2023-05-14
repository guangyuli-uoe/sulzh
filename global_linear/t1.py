from src import dataloader
from src import glm
from src import glm02
import numpy as np



if __name__ == '__main__':
    train_path = './src/train.conll'
    dev_path = './src/dev.conll'

    train_dataset = dataloader.DataLoader(train_path, batch_size=10)
    print(train_dataset)
    dev_dataset = dataloader.DataLoader(dev_path, batch_size=10)
    print(dev_dataset)
    print(train_dataset.tag_dict)

    # myglm = glm.GlobalLinearModel(train_dataset, dev_dataset)
    # print(myglm.epsilon)

    myglm02 = glm02.GlobalLinearModel(train_dataset, dev_dataset)
    # print(myglm02.epsilon)
    myglm02.viterbi_predict(train_dataset.sent_word_list[0])
    print(myglm02.bigram_features[0])
    print(len(myglm02.bigram_features[0]))
    print(np.array(myglm02.bigram_features).shape)
    '''
        [
        [['01:AD_AD'], ['01:AD_AS'], ['01:AD_BA'], ['01:AD_CC'], ['01:AD_CD'], ['01:AD_CS'], ['01:AD_DEC'], ['01:AD_DEG'], ['01:AD_DER'], ['01:AD_DEV'], ['01:AD_DT'], ['01:AD_ETC'], ['01:AD_FW'], ['01:AD_JJ'], ['01:AD_LB'], ['01:AD_LC'], ['01:AD_M'], ['01:AD_MSP'], ['01:AD_NN'], ['01:AD_NR'], ['01:AD_NT'], ['01:AD_OD'], ['01:AD_P'], ['01:AD_PN'], ['01:AD_PU'], ['01:AD_SB'], ['01:AD_SP'], ['01:AD_VA'], ['01:AD_VC'], ['01:AD_VE'], ['01:AD_VV']], 
        [['01:AS_AD'], ['01:AS_AS'], ['01:AS_BA'], ['01:AS_CC'], ['01:AS_CD'], ['01:AS_CS'], ['01:AS_DEC'], ['01:AS_DEG'], ['01:AS_DER'], ['01:AS_DEV'], ['01:AS_DT'], ['01:AS_ETC'], ['01:AS_FW'], ['01:AS_JJ'], ['01:AS_LB'], ['01:AS_LC'], ['01:AS_M'], ['01:AS_MSP'], ['01:AS_NN'], ['01:AS_NR'], ['01:AS_NT'], ['01:AS_OD'], ['01:AS_P'], ['01:AS_PN'], ['01:AS_PU'], ['01:AS_SB'], ['01:AS_SP'], ['01:AS_VA'], ['01:AS_VC'], ['01:AS_VE'], ['01:AS_VV']], 
        [['01:BA_AD'], ['01:BA_AS'], ['01:BA_BA'], ['01:BA_CC'], ['01:BA_CD'], ['01:BA_CS'], ['01:BA_DEC'], ['01:BA_DEG'], ['01:BA_DER'], ['01:BA_DEV'], ['01:BA_DT'], ['01:BA_ETC'], ['01:BA_FW'], ['01:BA_JJ'], ['01:BA_LB'], ['01:BA_LC'], ['01:BA_M'], ['01:BA_MSP'], ['01:BA_NN'], ['01:BA_NR'], ['01:BA_NT'], ['01:BA_OD'], ['01:BA_P'], ['01:BA_PN'], ['01:BA_PU'], ['01:BA_SB'], ['01:BA_SP'], ['01:BA_VA'], ['01:BA_VC'], ['01:BA_VE'], ['01:BA_VV']], [['01:CC_AD'], ['01:CC_AS'], ['01:CC_BA'], ['01:CC_CC'], ['01:CC_CD'], ['01:CC_CS'], ['01:CC_DEC'], ['01:CC_DEG'], ['01:CC_DER'], ['01:CC_DEV'], ['01:CC_DT'], ['01:CC_ETC'], ['01:CC_FW'], ['01:CC_JJ'], ['01:CC_LB'], ['01:CC_LC'], ['01:CC_M'], ['01:CC_MSP'], ['01:CC_NN'], ['01:CC_NR'], ['01:CC_NT'], ['01:CC_OD'], ['01:CC_P'], ['01:CC_PN'], ['01:CC_PU'], ['01:CC_SB'], ['01:CC_SP'], ['01:CC_VA'], ['01:CC_VC'], ['01:CC_VE'], ['01:CC_VV']], [['01:CD_AD'], ['01:CD_AS'], ['01:CD_BA'], ['01:CD_CC'], ['01:CD_CD'], ['01:CD_CS'], ['01:CD_DEC'], ['01:CD_DEG'], ['01:CD_DER'], ['01:CD_DEV'], ['01:CD_DT'], ['01:CD_ETC'], ['01:CD_FW'], ['01:CD_JJ'], ['01:CD_LB'], ['01:CD_LC'], ['01:CD_M'], ['01:CD_MSP'], ['01:CD_NN'], ['01:CD_NR'], ['01:CD_NT'], ['01:CD_OD'], ['01:CD_P'], ['01:CD_PN'], ['01:CD_PU'], ['01:CD_SB'], ['01:CD_SP'], ['01:CD_VA'], ['01:CD_VC'], ['01:CD_VE'], ['01:CD_VV']], [['01:CS_AD'], ['01:CS_AS'], ['01:CS_BA'], ['01:CS_CC'], ['01:CS_CD'], ['01:CS_CS'], ['01:CS_DEC'], ['01:CS_DEG'], ['01:CS_DER'], ['01:CS_DEV'], ['01:CS_DT'], ['01:CS_ETC'], ['01:CS_FW'], ['01:CS_JJ'], ['01:CS_LB'], ['01:CS_LC'], ['01:CS_M'], ['01:CS_MSP'], ['01:CS_NN'], ['01:CS_NR'], ['01:CS_NT'], ['01:CS_OD'], ['01:CS_P'], ['01:CS_PN'], ['01:CS_PU'], ['01:CS_SB'], ['01:CS_SP'], ['01:CS_VA'], ['01:CS_VC'], ['01:CS_VE'], ['01:CS_VV']], [['01:DEC_AD'], ['01:DEC_AS'], ['01:DEC_BA'], ['01:DEC_CC'], ['01:DEC_CD'], ['01:DEC_CS'], ['01:DEC_DEC'], ['01:DEC_DEG'], ['01:DEC_DER'], ['01:DEC_DEV'], ['01:DEC_DT'], ['01:DEC_ETC'], ['01:DEC_FW'], ['01:DEC_JJ'], ['01:DEC_LB'], ['01:DEC_LC'], ['01:DEC_M'], ['01:DEC_MSP'], ['01:DEC_NN'], ['01:DEC_NR'], ['01:DEC_NT'], ['01:DEC_OD'], ['01:DEC_P'], ['01:DEC_PN'], ['01:DEC_PU'], ['01:DEC_SB'], ['01:DEC_SP'], ['01:DEC_VA'], ['01:DEC_VC'], ['01:DEC_VE'], ['01:DEC_VV']], [['01:DEG_AD'], ['01:DEG_AS'], ['01:DEG_BA'], ['01:DEG_CC'], ['01:DEG_CD'], ['01:DEG_CS'], ['01:DEG_DEC'], ['01:DEG_DEG'], ['01:DEG_DER'], ['01:DEG_DEV'], ['01:DEG_DT'], ['01:DEG_ETC'], ['01:DEG_FW'], ['01:DEG_JJ'], ['01:DEG_LB'], ['01:DEG_LC'], ['01:DEG_M'], ['01:DEG_MSP'], ['01:DEG_NN'], ['01:DEG_NR'], ['01:DEG_NT'], ['01:DEG_OD'], ['01:DEG_P'], ['01:DEG_PN'], ['01:DEG_PU'], ['01:DEG_SB'], ['01:DEG_SP'], ['01:DEG_VA'], ['01:DEG_VC'], ['01:DEG_VE'], ['01:DEG_VV']], [['01:DER_AD'], ['01:DER_AS'], ['01:DER_BA'], ['01:DER_CC'], ['01:DER_CD'], ['01:DER_CS'], ['01:DER_DEC'], ['01:DER_DEG'], ['01:DER_DER'], ['01:DER_DEV'], ['01:DER_DT'], ['01:DER_ETC'], ['01:DER_FW'], ['01:DER_JJ'], ['01:DER_LB'], ['01:DER_LC'], ['01:DER_M'], ['01:DER_MSP'], ['01:DER_NN'], ['01:DER_NR'], ['01:DER_NT'], ['01:DER_OD'], ['01:DER_P'], ['01:DER_PN'], ['01:DER_PU'], ['01:DER_SB'], ['01:DER_SP'], ['01:DER_VA'], ['01:DER_VC'], ['01:DER_VE'], ['01:DER_VV']], [['01:DEV_AD'], ['01:DEV_AS'], ['01:DEV_BA'], ['01:DEV_CC'], ['01:DEV_CD'], ['01:DEV_CS'], ['01:DEV_DEC'], ['01:DEV_DEG'], ['01:DEV_DER'], ['01:DEV_DEV'], ['01:DEV_DT'], ['01:DEV_ETC'], ['01:DEV_FW'], ['01:DEV_JJ'], ['01:DEV_LB'], ['01:DEV_LC'], ['01:DEV_M'], ['01:DEV_MSP'], ['01:DEV_NN'], ['01:DEV_NR'], ['01:DEV_NT'], ['01:DEV_OD'], ['01:DEV_P'], ['01:DEV_PN'], ['01:DEV_PU'], ['01:DEV_SB'], ['01:DEV_SP'], ['01:DEV_VA'], ['01:DEV_VC'], ['01:DEV_VE'], ['01:DEV_VV']], [['01:DT_AD'], ['01:DT_AS'], ['01:DT_BA'], ['01:DT_CC'], ['01:DT_CD'], ['01:DT_CS'], ['01:DT_DEC'], ['01:DT_DEG'], ['01:DT_DER'], ['01:DT_DEV'], ['01:DT_DT'], ['01:DT_ETC'], ['01:DT_FW'], ['01:DT_JJ'], ['01:DT_LB'], ['01:DT_LC'], ['01:DT_M'], ['01:DT_MSP'], ['01:DT_NN'], ['01:DT_NR'], ['01:DT_NT'], ['01:DT_OD'], ['01:DT_P'], ['01:DT_PN'], ['01:DT_PU'], ['01:DT_SB'], ['01:DT_SP'], ['01:DT_VA'], ['01:DT_VC'], ['01:DT_VE'], ['01:DT_VV']], [['01:ETC_AD'], ['01:ETC_AS'], ['01:ETC_BA'], ['01:ETC_CC'], ['01:ETC_CD'], ['01:ETC_CS'], ['01:ETC_DEC'], ['01:ETC_DEG'], ['01:ETC_DER'], ['01:ETC_DEV'], ['01:ETC_DT'], ['01:ETC_ETC'], ['01:ETC_FW'], ['01:ETC_JJ'], ['01:ETC_LB'], ['01:ETC_LC'], ['01:ETC_M'], ['01:ETC_MSP'], ['01:ETC_NN'], ['01:ETC_NR'], ['01:ETC_NT'], ['01:ETC_OD'], ['01:ETC_P'], ['01:ETC_PN'], ['01:ETC_PU'], ['01:ETC_SB'], ['01:ETC_SP'], ['01:ETC_VA'], ['01:ETC_VC'], ['01:ETC_VE'], ['01:ETC_VV']], [['01:FW_AD'], ['01:FW_AS'], ['01:FW_BA'], ['01:FW_CC'], ['01:FW_CD'], ['01:FW_CS'], ['01:FW_DEC'], ['01:FW_DEG'], ['01:FW_DER'], ['01:FW_DEV'], ['01:FW_DT'], ['01:FW_ETC'], ['01:FW_FW'], ['01:FW_JJ'], ['01:FW_LB'], ['01:FW_LC'], ['01:FW_M'], ['01:FW_MSP'], ['01:FW_NN'], ['01:FW_NR'], ['01:FW_NT'], ['01:FW_OD'], ['01:FW_P'], ['01:FW_PN'], ['01:FW_PU'], ['01:FW_SB'], ['01:FW_SP'], ['01:FW_VA'], ['01:FW_VC'], ['01:FW_VE'], ['01:FW_VV']], [['01:JJ_AD'], ['01:JJ_AS'], ['01:JJ_BA'], ['01:JJ_CC'], ['01:JJ_CD'], ['01:JJ_CS'], ['01:JJ_DEC'], ['01:JJ_DEG'], ['01:JJ_DER'], ['01:JJ_DEV'], ['01:JJ_DT'], ['01:JJ_ETC'], ['01:JJ_FW'], ['01:JJ_JJ'], ['01:JJ_LB'], ['01:JJ_LC'], ['01:JJ_M'], ['01:JJ_MSP'], ['01:JJ_NN'], ['01:JJ_NR'], ['01:JJ_NT'], ['01:JJ_OD'], ['01:JJ_P'], ['01:JJ_PN'], ['01:JJ_PU'], ['01:JJ_SB'], ['01:JJ_SP'], ['01:JJ_VA'], ['01:JJ_VC'], ['01:JJ_VE'], ['01:JJ_VV']], [['01:LB_AD'], ['01:LB_AS'], ['01:LB_BA'], ['01:LB_CC'], ['01:LB_CD'], ['01:LB_CS'], ['01:LB_DEC'], ['01:LB_DEG'], ['01:LB_DER'], ['01:LB_DEV'], ['01:LB_DT'], ['01:LB_ETC'], ['01:LB_FW'], ['01:LB_JJ'], ['01:LB_LB'], ['01:LB_LC'], ['01:LB_M'], ['01:LB_MSP'], ['01:LB_NN'], ['01:LB_NR'], ['01:LB_NT'], ['01:LB_OD'], ['01:LB_P'], ['01:LB_PN'], ['01:LB_PU'], ['01:LB_SB'], ['01:LB_SP'], ['01:LB_VA'], ['01:LB_VC'], ['01:LB_VE'], ['01:LB_VV']], [['01:LC_AD'], ['01:LC_AS'], ['01:LC_BA'], ['01:LC_CC'], ['01:LC_CD'], ['01:LC_CS'], ['01:LC_DEC'], ['01:LC_DEG'], ['01:LC_DER'], ['01:LC_DEV'], ['01:LC_DT'], ['01:LC_ETC'], ['01:LC_FW'], ['01:LC_JJ'], ['01:LC_LB'], ['01:LC_LC'], ['01:LC_M'], ['01:LC_MSP'], ['01:LC_NN'], ['01:LC_NR'], ['01:LC_NT'], ['01:LC_OD'], ['01:LC_P'], ['01:LC_PN'], ['01:LC_PU'], ['01:LC_SB'], ['01:LC_SP'], ['01:LC_VA'], ['01:LC_VC'], ['01:LC_VE'], ['01:LC_VV']], [['01:M_AD'], ['01:M_AS'], ['01:M_BA'], ['01:M_CC'], ['01:M_CD'], ['01:M_CS'], ['01:M_DEC'], ['01:M_DEG'], ['01:M_DER'], ['01:M_DEV'], ['01:M_DT'], ['01:M_ETC'], ['01:M_FW'], ['01:M_JJ'], ['01:M_LB'], ['01:M_LC'], ['01:M_M'], ['01:M_MSP'], ['01:M_NN'], ['01:M_NR'], ['01:M_NT'], ['01:M_OD'], ['01:M_P'], ['01:M_PN'], ['01:M_PU'], ['01:M_SB'], ['01:M_SP'], ['01:M_VA'], ['01:M_VC'], ['01:M_VE'], ['01:M_VV']], [['01:MSP_AD'], ['01:MSP_AS'], ['01:MSP_BA'], ['01:MSP_CC'], ['01:MSP_CD'], ['01:MSP_CS'], ['01:MSP_DEC'], ['01:MSP_DEG'], ['01:MSP_DER'], ['01:MSP_DEV'], ['01:MSP_DT'], ['01:MSP_ETC'], ['01:MSP_FW'], ['01:MSP_JJ'], ['01:MSP_LB'], ['01:MSP_LC'], ['01:MSP_M'], ['01:MSP_MSP'], ['01:MSP_NN'], ['01:MSP_NR'], ['01:MSP_NT'], ['01:MSP_OD'], ['01:MSP_P'], ['01:MSP_PN'], ['01:MSP_PU'], ['01:MSP_SB'], ['01:MSP_SP'], ['01:MSP_VA'], ['01:MSP_VC'], ['01:MSP_VE'], ['01:MSP_VV']], [['01:NN_AD'], ['01:NN_AS'], ['01:NN_BA'], ['01:NN_CC'], ['01:NN_CD'], ['01:NN_CS'], ['01:NN_DEC'], ['01:NN_DEG'], ['01:NN_DER'], ['01:NN_DEV'], ['01:NN_DT'], ['01:NN_ETC'], ['01:NN_FW'], ['01:NN_JJ'], ['01:NN_LB'], ['01:NN_LC'], ['01:NN_M'], ['01:NN_MSP'], ['01:NN_NN'], ['01:NN_NR'], ['01:NN_NT'], ['01:NN_OD'], ['01:NN_P'], ['01:NN_PN'], ['01:NN_PU'], ['01:NN_SB'], ['01:NN_SP'], ['01:NN_VA'], ['01:NN_VC'], ['01:NN_VE'], ['01:NN_VV']], [['01:NR_AD'], ['01:NR_AS'], ['01:NR_BA'], ['01:NR_CC'], ['01:NR_CD'], ['01:NR_CS'], ['01:NR_DEC'], ['01:NR_DEG'], ['01:NR_DER'], ['01:NR_DEV'], ['01:NR_DT'], ['01:NR_ETC'], ['01:NR_FW'], ['01:NR_JJ'], ['01:NR_LB'], ['01:NR_LC'], ['01:NR_M'], ['01:NR_MSP'], ['01:NR_NN'], ['01:NR_NR'], ['01:NR_NT'], ['01:NR_OD'], ['01:NR_P'], ['01:NR_PN'], ['01:NR_PU'], ['01:NR_SB'], ['01:NR_SP'], ['01:NR_VA'], ['01:NR_VC'], ['01:NR_VE'], ['01:NR_VV']], [['01:NT_AD'], ['01:NT_AS'], ['01:NT_BA'], ['01:NT_CC'], ['01:NT_CD'], ['01:NT_CS'], ['01:NT_DEC'], ['01:NT_DEG'], ['01:NT_DER'], ['01:NT_DEV'], ['01:NT_DT'], ['01:NT_ETC'], ['01:NT_FW'], ['01:NT_JJ'], ['01:NT_LB'], ['01:NT_LC'], ['01:NT_M'], ['01:NT_MSP'], ['01:NT_NN'], ['01:NT_NR'], ['01:NT_NT'], ['01:NT_OD'], ['01:NT_P'], ['01:NT_PN'], ['01:NT_PU'], ['01:NT_SB'], ['01:NT_SP'], ['01:NT_VA'], ['01:NT_VC'], ['01:NT_VE'], ['01:NT_VV']], [['01:OD_AD'], ['01:OD_AS'], ['01:OD_BA'], ['01:OD_CC'], ['01:OD_CD'], ['01:OD_CS'], ['01:OD_DEC'], ['01:OD_DEG'], ['01:OD_DER'], ['01:OD_DEV'], ['01:OD_DT'], ['01:OD_ETC'], ['01:OD_FW'], ['01:OD_JJ'], ['01:OD_LB'], ['01:OD_LC'], ['01:OD_M'], ['01:OD_MSP'], ['01:OD_NN'], ['01:OD_NR'], ['01:OD_NT'], ['01:OD_OD'], ['01:OD_P'], ['01:OD_PN'], ['01:OD_PU'], ['01:OD_SB'], ['01:OD_SP'], ['01:OD_VA'], ['01:OD_VC'], ['01:OD_VE'], ['01:OD_VV']], [['01:P_AD'], ['01:P_AS'], ['01:P_BA'], ['01:P_CC'], ['01:P_CD'], ['01:P_CS'], ['01:P_DEC'], ['01:P_DEG'], ['01:P_DER'], ['01:P_DEV'], ['01:P_DT'], ['01:P_ETC'], ['01:P_FW'], ['01:P_JJ'], ['01:P_LB'], ['01:P_LC'], ['01:P_M'], ['01:P_MSP'], ['01:P_NN'], ['01:P_NR'], ['01:P_NT'], ['01:P_OD'], ['01:P_P'], ['01:P_PN'], ['01:P_PU'], ['01:P_SB'], ['01:P_SP'], ['01:P_VA'], ['01:P_VC'], ['01:P_VE'], ['01:P_VV']], [['01:PN_AD'], ['01:PN_AS'], ['01:PN_BA'], ['01:PN_CC'], ['01:PN_CD'], ['01:PN_CS'], ['01:PN_DEC'], ['01:PN_DEG'], ['01:PN_DER'], ['01:PN_DEV'], ['01:PN_DT'], ['01:PN_ETC'], ['01:PN_FW'], ['01:PN_JJ'], ['01:PN_LB'], ['01:PN_LC'], ['01:PN_M'], ['01:PN_MSP'], ['01:PN_NN'], ['01:PN_NR'], ['01:PN_NT'], ['01:PN_OD'], ['01:PN_P'], ['01:PN_PN'], ['01:PN_PU'], ['01:PN_SB'], ['01:PN_SP'], ['01:PN_VA'], ['01:PN_VC'], ['01:PN_VE'], ['01:PN_VV']], [['01:PU_AD'], ['01:PU_AS'], ['01:PU_BA'], ['01:PU_CC'], ['01:PU_CD'], ['01:PU_CS'], ['01:PU_DEC'], ['01:PU_DEG'], ['01:PU_DER'], ['01:PU_DEV'], ['01:PU_DT'], ['01:PU_ETC'], ['01:PU_FW'], ['01:PU_JJ'], ['01:PU_LB'], ['01:PU_LC'], ['01:PU_M'], ['01:PU_MSP'], ['01:PU_NN'], ['01:PU_NR'], ['01:PU_NT'], ['01:PU_OD'], ['01:PU_P'], ['01:PU_PN'], ['01:PU_PU'], ['01:PU_SB'], ['01:PU_SP'], ['01:PU_VA'], ['01:PU_VC'], ['01:PU_VE'], ['01:PU_VV']], [['01:SB_AD'], ['01:SB_AS'], ['01:SB_BA'], ['01:SB_CC'], ['01:SB_CD'], ['01:SB_CS'], ['01:SB_DEC'], ['01:SB_DEG'], ['01:SB_DER'], ['01:SB_DEV'], ['01:SB_DT'], ['01:SB_ETC'], ['01:SB_FW'], ['01:SB_JJ'], ['01:SB_LB'], ['01:SB_LC'], ['01:SB_M'], ['01:SB_MSP'], ['01:SB_NN'], ['01:SB_NR'], ['01:SB_NT'], ['01:SB_OD'], ['01:SB_P'], ['01:SB_PN'], ['01:SB_PU'], ['01:SB_SB'], ['01:SB_SP'], ['01:SB_VA'], ['01:SB_VC'], ['01:SB_VE'], ['01:SB_VV']], [['01:SP_AD'], ['01:SP_AS'], ['01:SP_BA'], ['01:SP_CC'], ['01:SP_CD'], ['01:SP_CS'], ['01:SP_DEC'], ['01:SP_DEG'], ['01:SP_DER'], ['01:SP_DEV'], ['01:SP_DT'], ['01:SP_ETC'], ['01:SP_FW'], ['01:SP_JJ'], ['01:SP_LB'], ['01:SP_LC'], ['01:SP_M'], ['01:SP_MSP'], ['01:SP_NN'], ['01:SP_NR'], ['01:SP_NT'], ['01:SP_OD'], ['01:SP_P'], ['01:SP_PN'], ['01:SP_PU'], ['01:SP_SB'], ['01:SP_SP'], ['01:SP_VA'], ['01:SP_VC'], ['01:SP_VE'], ['01:SP_VV']], [['01:VA_AD'], ['01:VA_AS'], ['01:VA_BA'], ['01:VA_CC'], ['01:VA_CD'], ['01:VA_CS'], ['01:VA_DEC'], ['01:VA_DEG'], ['01:VA_DER'], ['01:VA_DEV'], ['01:VA_DT'], ['01:VA_ETC'], ['01:VA_FW'], ['01:VA_JJ'], ['01:VA_LB'], ['01:VA_LC'], ['01:VA_M'], ['01:VA_MSP'], ['01:VA_NN'], ['01:VA_NR'], ['01:VA_NT'], ['01:VA_OD'], ['01:VA_P'], ['01:VA_PN'], ['01:VA_PU'], ['01:VA_SB'], ['01:VA_SP'], ['01:VA_VA'], ['01:VA_VC'], ['01:VA_VE'], ['01:VA_VV']], [['01:VC_AD'], ['01:VC_AS'], ['01:VC_BA'], ['01:VC_CC'], ['01:VC_CD'], ['01:VC_CS'], ['01:VC_DEC'], ['01:VC_DEG'], ['01:VC_DER'], ['01:VC_DEV'], ['01:VC_DT'], ['01:VC_ETC'], ['01:VC_FW'], ['01:VC_JJ'], ['01:VC_LB'], ['01:VC_LC'], ['01:VC_M'], ['01:VC_MSP'], ['01:VC_NN'], ['01:VC_NR'], ['01:VC_NT'], ['01:VC_OD'], ['01:VC_P'], ['01:VC_PN'], ['01:VC_PU'], ['01:VC_SB'], ['01:VC_SP'], ['01:VC_VA'], ['01:VC_VC'], ['01:VC_VE'], ['01:VC_VV']], [['01:VE_AD'], ['01:VE_AS'], ['01:VE_BA'], ['01:VE_CC'], ['01:VE_CD'], ['01:VE_CS'], ['01:VE_DEC'], ['01:VE_DEG'], ['01:VE_DER'], ['01:VE_DEV'], ['01:VE_DT'], ['01:VE_ETC'], ['01:VE_FW'], ['01:VE_JJ'], ['01:VE_LB'], ['01:VE_LC'], ['01:VE_M'], ['01:VE_MSP'], ['01:VE_NN'], ['01:VE_NR'], ['01:VE_NT'], ['01:VE_OD'], ['01:VE_P'], ['01:VE_PN'], ['01:VE_PU'], ['01:VE_SB'], ['01:VE_SP'], ['01:VE_VA'], ['01:VE_VC'], ['01:VE_VE'], ['01:VE_VV']], [['01:VV_AD'], ['01:VV_AS'], ['01:VV_BA'], ['01:VV_CC'], ['01:VV_CD'], ['01:VV_CS'], ['01:VV_DEC'], ['01:VV_DEG'], ['01:VV_DER'], ['01:VV_DEV'], ['01:VV_DT'], ['01:VV_ETC'], ['01:VV_FW'], ['01:VV_JJ'], ['01:VV_LB'], ['01:VV_LC'], ['01:VV_M'], ['01:VV_MSP'], ['01:VV_NN'], ['01:VV_NR'], ['01:VV_NT'], ['01:VV_OD'], ['01:VV_P'], ['01:VV_PN'], ['01:VV_PU'], ['01:VV_SB'], ['01:VV_SP'], ['01:VV_VA'], ['01:VV_VC'], ['01:VV_VE'], ['01:VV_VV']]]
        
        [
        [['01:AD_AD'], ['01:AD_AS'], ['01:AD_BA'], ['01:AD_CC'], ['01:AD_CD'], ['01:AD_CS'], ['01:AD_DEC'], ['01:AD_DEG'], ['01:AD_DER'], ['01:AD_DEV'], ['01:AD_DT'], ['01:AD_ETC'], ['01:AD_FW'], ['01:AD_JJ'], ['01:AD_LB'], ['01:AD_LC'], ['01:AD_M'], ['01:AD_MSP'], ['01:AD_NN'], ['01:AD_NR'], ['01:AD_NT'], ['01:AD_OD'], ['01:AD_P'], ['01:AD_PN'], ['01:AD_PU'], ['01:AD_SB'], ['01:AD_SP'], ['01:AD_VA'], ['01:AD_VC'], ['01:AD_VE'], ['01:AD_VV']]
]
    '''