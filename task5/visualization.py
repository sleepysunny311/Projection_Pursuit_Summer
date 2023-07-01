import os
import pickle as pkl
import matplotlib.pyplot as plt

def makeplots(res_log, sep=False, savefig = True, ylim = None):
    if not os.path.exists('./images'):
        os.makedirs('./images')
        
    # res_log can either be a object name or path to the file
    if isinstance(res_log, str):
        path = os.path.join('./', res_log)
        with open(path, 'rb') as f:
            res_log = pkl.load(f)
    
    # Extract parameters
    parameters = res_log['parameters']
    noise_level_best_K = res_log['noise_level_best_alpha']
    noise_level_lowest_MSE = res_log['noise_level_lowest_MSE']
    
    alpha_lst = parameters['alpha_lst']
    trial_num = res_log['parameters']['trial_num']
    print('Parameters: ', parameters)
    
    # Get folder name for saving images
    imgfolderPath = 'Lasso' + str(parameters['N']) + '_' + str(parameters['d']) + '_' + str(parameters['m']) + '_' + str(parameters['trial_num']) + '_' + str(parameters['cv_num'])
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
        for i, noise_level in enumerate(noise_level_lst):
            fig, ax = plt.subplots()
            for j in range(trial_num):
                plt.plot(alpha_lst, res_log['log'][i * trial_num + j]['cv_error_lst'])
            plt.xlabel("Iteration number")
            plt.ylabel("CV prediction error")
            plt.title("CV prediction error\nn = " + str(parameters['d']) + ", p = " + str(parameters['N'])+ ", m = " + str(parameters['m'])+"\nNoise std: " + str(noise_level))    
            if savefig:
                plt.savefig('./images/' + imgfolderPath + '/noise_sep/' + str(noise_level) + '_' + str(j) + '.png')
            plt.show()
    # Plot all noise level together no matter whther sep or not
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
        plt.plot(alpha_lst, tmp[i][2], label = label, color = plt.cm.Set2_r(i // trial_num))
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    ax.get_legend().set_title("Noise std")
    if ylim:
        plt.ylim(ylim)
    else:
        plt.ylim([0, 0.5])
    
    plt.xlabel("Lambda")
    plt.ylabel("CV prediction error")
    plt.title("CV prediction error for each noise level\n n = " + str(parameters['d']) + ", p = " + str(parameters['N'])+ ", m = " + str(parameters['m'])) 
    if savefig:
        plt.savefig('./images/' + imgfolderPath + '/cv_error.png', dpi=200, bbox_inches='tight')
    plt.show()

    ## Second plot: Noise std vs. best iteration number
    plt.plot(noise_level_lst, noise_level_best_K)
    plt.title("Best lambda for each noise level\n n = " + str(parameters['d']) + ", p = " + str(parameters['N'])+ ", m = " + str(parameters['m']))
    plt.xlabel("Noise std")
    plt.ylabel("Best lambda")
    if savefig:
        plt.savefig('./images/' + imgfolderPath + '/noise_vs_bestK.png', dpi=200, bbox_inches='tight')
    plt.show()
    
    ## Third plot: Noise std vs. CV prediction error for best iteration number
    plt.plot(noise_level_lst, noise_level_lowest_MSE)
    plt.xlabel("Noise std")
    plt.ylabel("CV prediction error for best K")
    plt.title("CV prediction error for best iteration number for each noise level\n n = " + str(parameters['d']) + ", p = " + str(parameters['N']) + ", m = " + str(parameters['m']))
    if savefig:
        plt.savefig('./images/' + imgfolderPath + '/noise_vs_bestMSE.png', dpi=200, bbox_inches='tight')
    plt.show()
    
    
def makeplotsgrid(res_log, savefig = True, colnum = 3, fig_size = None, title = True):
    if not os.path.exists('./images'):
        os.makedirs('./images')
        
    # res_log can either be a object name or path to the file
    if isinstance(res_log, str):
        path = os.path.join('./', res_log)
        with open(path, 'rb') as f:
            res_log = pkl.load(f)
    
    # Extract parameters
    parameters = res_log['parameters']
    noise_level_best_K = res_log['noise_level_best_alpha']
    noise_level_lowest_MSE = res_log['noise_level_lowest_MSE']
    
    K_lst = parameters['alpha_lst']
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
    
    if not os.path.exists('./images/' + imgfolderPath + '/noise_sep'):
        os.makedirs('./images/' + imgfolderPath + '/noise_sep')

    if fig_size:
        fig, ax = plt.subplots(len(noise_level_lst) // colnum + 1, colnum, figsize = fig_size)
    else:
        fig, ax = plt.subplots(len(noise_level_lst) // colnum + 1, colnum, figsize = (10, 12))
    for i, noise_level in enumerate(noise_level_lst):
        row = int(i/colnum)
        column = i % colnum
        for j in range(trial_num):
            ax[row, column].plot(K_lst, res_log['log'][i * trial_num + j]['cv_error_lst'])
        ax[row, column].set_xlabel("Lambda")
        ax[row, column].set_ylabel("CV prediction error")
        ax[row, column].set_title("Noise std: " + str(noise_level))  
    plt.tight_layout()
    if title:
        fig.suptitle("CV prediction error\nn = " + str(parameters['d']) + ", p = " + str(parameters['N'])+ ", m = " + str(parameters['m']), fontsize="x-large", y = 1.05)
    if savefig:
        plt.savefig('./images/' + imgfolderPath + '/noise_sep/' + 'grid.png')  
    plt.show()
  