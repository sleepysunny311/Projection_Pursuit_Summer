import os
import pickle as pkl
import matplotlib.pyplot as plt
import glob

def makeplots(res_log, sep=False, savefig = True):
    if not os.path.exists('./images'):
        os.makedirs('./images')
        
    # res_log can either be a object name or path to the file
    if isinstance(res_log, str):
        path = os.path.join('./memory/', res_log)
        with open(path, 'rb') as f:
            res_log = pkl.load(f)
    
    # Extract parameters
    parameters = res_log['parameters']
    noise_level_best_K = res_log['noise_level_best_K']
    noise_level_lowest_MSE = res_log['noise_level_lowest_MSE']
    
    K_lst = parameters['K_lst']
    trial_num = res_log['parameters']['trial_num']
    print('Parameters: ', parameters)
    
    # Get folder name for saving images
    imgfolderPath = str(parameters['N']) + '_' + str(parameters['d']) + '_' + str(parameters['m']) + '_' + str(parameters['trial_num']) + '_' + str(parameters['cv_num'])
    if savefig:
        if not os.path.exists('./images/' + imgfolderPath):
            os.makedirs('./images/' + imgfolderPath)
    
    # Generate plots
    ## First plot: Iteration number vs. CV prediction error for each noise level
    noise_level_lst = parameters['noise_level_lst']
    if sep:
        if not os.path.exists('./images/' + imgfolderPath + '/noise_sep'):
            os.makedirs('./images/' + imgfolderPath + '/noise_sep')
        # Plot the cv error for each K for each noise level in separate plots
        # Each noise level has trial_num trials and we put them in the same plot
        for i in range(len(res_log['log'])):
            fig, ax = plt.subplots()
            for j in range(trial_num):
                plt.plot(K_lst, res_log['log'][i]['cv_error_lst'][j], label = "Trial " + str(j + 1))
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            plt.xlabel("Iteration number")
            plt.ylabel("CV prediction error")
            plt.title("CV prediction error for noise level " + str(res_log['log'][i]['noise_level']) + " with m = " + str(parameters['m']) + " and d = " + str(parameters['d']))
            if savefig:
                plt.savefig('./images/' + imgfolderPath + '/noise_sep/' + str(res_log['log'][i]['noise_level']) + '.png', dpi=200, bbox_inches='tight')
            plt.show()
    else:
        tmp = []
        for i in range(len(res_log['log'])):
            tmp.append([res_log['log'][i]['noise_level'], res_log['log'][i]['trial'], res_log['log'][i]['cv_error_lst']])
        fig, ax = plt.subplots()
        # Plot the cv error for each K for each noise level
        for i in range(len(tmp)):    
            if i % trial_num == 0:
                label = tmp[i][0]
            else:
                label = ""
            # Assign color to different noise level
            plt.plot(K_lst, tmp[i][2], label = label, color = plt.cm.Set2_r(i // trial_num))
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        ax.get_legend().set_title("Noise std")

        
        plt.xlabel("Iteration number")
        plt.ylabel("CV prediction error")
        plt.title("CV prediction error for each noise level with m = " + str(parameters['m']) + " and d = " + str(parameters['d']))
        if savefig:
            plt.savefig('./images/' + imgfolderPath + '/cv_error.png', dpi=200, bbox_inches='tight')
        plt.show()

    ## Second plot: Noise std vs. best iteration number
    plt.plot(noise_level_lst, noise_level_best_K)
    plt.title("Best iteration number for each noise level with m = " + str(parameters['m']) + " and d = " + str(parameters['d']))
    plt.xlabel("Noise std")
    plt.ylabel("Best iteration number")
    if savefig:
        plt.savefig('./images/' + imgfolderPath + '/noise_vs_bestK.png', dpi=200, bbox_inches='tight')
    plt.show()
    
    ## Third plot: Noise std vs. CV prediction error for best iteration number
    plt.plot(noise_level_lst, noise_level_lowest_MSE)
    plt.xlabel("Noise std")
    plt.ylabel("CV prediction error for best K")
    plt.title("CV prediction error for best iteration number for each noise level with m = " + str(parameters['m']) + " and d = " + str(parameters['d']))
    if savefig:
        plt.savefig('./images/' + imgfolderPath + '/noise_vs_bestMSE.png', dpi=200, bbox_inches='tight')
    plt.show()

# # Extracing All information
# pkl_files = glob.glob("memory/*.pkl")
# parameter_dict_list = []
# for i in range(len(pkl_files)):
#     with open(pkl_files[i], 'rb') as f:
#         res_log = pkl.load(f)
#     makeplots(res_log, savefig = True)