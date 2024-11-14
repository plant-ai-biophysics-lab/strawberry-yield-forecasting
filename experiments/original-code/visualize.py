import matplotlib.pyplot as plt
from scipy.stats import linregress
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os

def kfold_plot(model_path,_X_per_t,_y_per_t,samples_dim,y_Data,year,k_fold,block_size,err_metric,phenological=True,normalize=True,save=True):
    # Get the current directory where your script is running
    current_dir = os.path.dirname(__file__)
    # Move up one directory to the 'strawberry_forecasting' level
    base_dir = os.path.abspath(os.path.join(current_dir, '..'))
    images_dir = os.path.join(base_dir, 'images')
    models_dir = os.path.join(base_dir, 'models')
    if phenological:
        phen_str='XT'
    else:
        phen_str='X'
    _r2 = []
    _metric = []
    _y_hat = []
    _y_true = []
    if k_fold == 0:
        n = 0#samples_dim[1]-1
        m = 0
        t = samples_dim[0]+1
    else:
        n = samples_dim[1]
        m = samples_dim[1]
        t = samples_dim[0]*2
    fig,(p1,p2) = plt.subplots(1,2,figsize=(10,5))
    scatter_marker = ['.','d','*','x','+','o','.']
    scatter_color = ['black','gray','orange','steelblue','darkviolet','blue','pink']
    best_model = load_model(models_dir+'/'+model_path+'_'+str(year)+'_'+str(block_size)+'_'+str(samples_dim[0])+str(samples_dim[1])+'_'+str(k_fold+1)+'_'+phen_str+'.h5')
    flattened_y = y_Data.flatten().reshape(-1, 1)
    for X_t,y_t in zip(_X_per_t[k_fold],_y_per_t[k_fold]):
        model_output = best_model.predict(np.array(X_t))
        y_hat=model_output[:, :1]  # Take only the first column of the output tensor
        if normalize:
            y_gt_scaler = MinMaxScaler(feature_range=(0, 1))
            y_fit = y_gt_scaler.fit(flattened_y)
            #y_t[:] = np.array(y_t)
            y_hat_reshaped = np.zeros((flattened_y.shape[0], 1))
            y_hat_reshaped[:y_hat.shape[0], :] = y_hat
            #y_t_reshaped = np.zeros((flattened_y.shape[0], 1))
            #y_t_reshaped[:len(y_t), :] = y_t
            Oy_hat = y_fit.inverse_transform(y_hat_reshaped)
            #Oy_t = y_fit.inverse_transform(y_t_reshaped)
            cy_hat = Oy_hat[:y_hat.shape[0]]
            #cy_true = Oy_t[:y_hat.shape[0]-1]
            cy_true = y_t
            print(np.array(cy_hat).shape,np.array(cy_true).shape)
            slope, intercept, r_value, p_value, std_err = linregress(np.ravel(cy_true), np.ravel(cy_hat))
            error_metric = mean_squared_error(np.ravel(cy_true), np.ravel(cy_hat))
            metric = np.sqrt(error_metric)   
            _y_hat.append(cy_hat)
            _y_true.append(cy_true)
        else:
            _y_hat.append(y_hat)
            _y_true.append(y_t)
            slope, intercept, r_value, p_value, std_err = linregress(np.ravel(y_t), np.ravel(y_hat))
            error_metric = mean_squared_error(np.ravel(y_t), np.ravel(y_hat))
            metric = np.sqrt(error_metric)
        r_squared = r_value ** 2
        _r2.append(r_squared)
        _metric.append(metric)
        if normalize:
            p1.scatter(cy_true, cy_hat,s=10,alpha=1, color=scatter_color[n-m],marker=scatter_marker[n-m],label = r'$t_{{{}}}$ (R$^{{2}}$={:.2f}, RMSE={:.2f})'.format(n + t, _r2[n - m], metric))#label=f'$t_{n+t}$ (R$^{2}$={_r2[n-m]:.2f}, RMSE={metric:.2f})')
        else:
            p1.scatter(y_t, y_hat,s=10,alpha=1, color=scatter_color[n-m],marker=scatter_marker[n-m],label = r'$t_{{{}}}$ (R$^{{2}}$={:.2f}, RMSE={:.2f})'.format(n + t, _r2[n - m], metric))#label=f'$t_{n+t}$ (R$^{2}$={_r2[n-m]:.2f}, RMSE={metric:.2f})')
        n+=1

    #print('Metric =',np.mean(_metric))
    #print('R2 =',np.mean(_r2))
    p1.set_xlabel('$y$ [count]')
    p1.set_ylabel('$\hat{y}$ [count]')
    p1.set_title('Ground truth vs predicted scatter plot')
    p1.axis('equal')
    p1.set_aspect('equal')
    p1.legend()
 
#
    p2.boxplot(_r2, widths=0.2,positions=[1])  # Box plot for R2 values
    p2.set_ylabel('R$^2$ values')
    p2.set_xticks([])
#
    p2_mse = p2.twinx()  # Create another twin y-axis for MSE values
    p2_mse.boxplot(_metric, widths=0.2,positions=[2])
    p2_mse.set_ylabel(err_metric+' values')
    p2_mse.set_xticks([])
    p2.set_title('R$^2$ and '+err_metric+' Box plots')
    p2.text(0.33, 0.15, f'Mean R$^{2}$= {np.mean(_r2):.2f}\nMean RMSE={np.mean(_metric):.2f}', 
        horizontalalignment='left', 
        verticalalignment='top', 
        transform=plt.gca().transAxes,
        bbox=dict(facecolor='white', alpha=0.5))
    plt.tight_layout()
    if save:
        plt.savefig(images_dir+'/'+model_path+str(year)+'_'+str(block_size)+'_'+str(samples_dim[0])+str(samples_dim[1])+'_'+str(k_fold+1)+phen_str+'.pdf', dpi=300)
    plt.show()
    return _metric, _r2#_y_true, _y_hat

def yearly_plot(model_path,_X_per_t,_y_per_t,samples_dim,y_Data,year,k_fold,block_size,err_metric,phenological=True,normalize=True,save=True):
    # Get the current directory where your script is running
    current_dir = os.path.dirname(__file__)
    # Move up one directory to the 'strawberry_forecasting' level
    base_dir = os.path.abspath(os.path.join(current_dir, '..'))
    images_dir = os.path.join(base_dir, 'images')
    models_dir = os.path.join(base_dir, 'models')
    if phenological:
        phen_str='XT'
    else:
        phen_str='X'
    _r2 = []
    _metric = []
    _y_hat = []
    _y_true = []
    n = samples_dim[0]
    fig,(p1,p2) = plt.subplots(1,2,figsize=(18,10))
    scatter_marker = ['.','d','*','x','+','.','o','d','*','x','+','.','d','*']
    scatter_color = ['black','gray','slategray','steelblue','turquoise','blue','darkviolet','pink','coral','orange','sandybrown','brown','gold']
    best_model = load_model(models_dir+'/'+model_path+'_'+str(year)+'_'+str(block_size)+'_'+str(samples_dim[0])+str(samples_dim[1])+'_'+str(k_fold)+'_'+phen_str+'.h5')
    flattened_y = y_Data.flatten().reshape(-1, 1)
    _r2_45 = []
    _rmse_45 = []
    for X_t,y_t in zip(_X_per_t,_y_per_t):
        #X_t = np.array(_X_per_t)
        #y_t = np.array(_y_per_t)
        model_output = best_model.predict(np.array(X_t))
        y_hat=model_output[:, :1]  # Take only the first column of the output tensor
        if normalize:
            y_gt_scaler = MinMaxScaler(feature_range=(0, 1))
            y_fit = y_gt_scaler.fit(flattened_y)
            y_t[:] = np.array(y_t)
            y_hat_reshaped = np.zeros((flattened_y.shape[0], 1))
            y_hat_reshaped[:y_hat.shape[0], :] = y_hat
            y_t_reshaped = np.zeros((flattened_y.shape[0], 1))
            y_t_reshaped[:len(y_t), :] = y_t
            Oy_hat = y_fit.inverse_transform(y_hat_reshaped)
            Oy_t = y_fit.inverse_transform(y_t_reshaped)
            cy_hat = np.ravel(Oy_hat[:y_hat.shape[0]])
            #cy_true = Oy_t[:y_hat.shape[0]-1]
            cy_true = np.ravel(y_t)

            slope, intercept, r_value, p_value, std_err = linregress(cy_true, cy_hat)
            # R^2 for line through the origin
            #beta = np.sum(cy_true * cy_hat) / np.sum(cy_hat**2)  # Constrain to origin
            y_pred_45 = cy_hat  # Predicted values based on this slope

            # Compute residual sum of squares for the origin-constrained line
            SS_res = np.sum((cy_true - y_pred_45)**2)
            SS_tot = np.sum((cy_true)**2)
            r2_origin = 1 - (SS_res / SS_tot)
            rmse_45 = np.sqrt(np.mean((cy_true - y_pred_45)**2))
            _r2_45.append(r2_origin)
            _rmse_45.append(rmse_45)
            print(r2_origin, rmse_45)
            error_metric = mean_squared_error(cy_true, cy_hat)
            metric = np.sqrt(error_metric)     
            _y_hat.append(cy_hat)
            _y_true.append(cy_true)
        r_squared = r_value ** 2
        _r2.append(r_squared)
        _metric.append(metric)
        print(n)
        if normalize:
            p1.scatter(cy_true, cy_hat,s=10,alpha=1, color=scatter_color[n-samples_dim[0]],marker=scatter_marker[n-samples_dim[0]],label = r'$t_{{{}}}$ (R$^{{2}}$={:.2f}, RMSE={:.2f})'.format(n, r_squared, metric))#label=f'$t_{n+t}$ (R$^{2}$={_r2[n-m]:.2f}, RMSE={metric:.2f})')
        else:
            p1.scatter(y_t, y_hat,s=10,alpha=1, color=scatter_color[n-samples_dim[0]],marker=scatter_marker[n-samples_dim[0]],label = r'$t_{{{}}}$ (R$^{{2}}$={:.2f}, RMSE={:.2f})'.format(n, r_squared, metric))#label=f'$t_{n+t}$ (R$^{2}$={_r2[n-m]:.2f}, RMSE={metric:.2f})')
        n+=1
    #
    p1.set_xlabel('$y$ [count]')
    p1.set_ylabel('$\hat{y}$ [count]')
    p1.set_title('Ground truth vs predicted scatter plot')
    p1.axis('equal')
    p1.set_aspect('equal')
    p1.legend()
#
    p2.boxplot(_r2, widths=0.2,positions=[1])  # Box plot for R2 values
    p2.set_ylabel('R$^2$ values')
    p2.set_xticks([])
#
    p2_mse = p2.twinx()  # Create another twin y-axis for MSE values
    p2_mse.boxplot(_metric, widths=0.2,positions=[2])
    p2_mse.set_ylabel(err_metric+' values')
    p2_mse.set_xticks([])
    p2.set_title('R$^2$ and '+err_metric+' Box plots')
    p2.text(0.33, 0.15, f'Mean R$^{2}$= {np.mean(_r2):.2f}\nMean RMSE={np.mean(_metric):.2f}', 
        horizontalalignment='left', 
        verticalalignment='top', 
        transform=plt.gca().transAxes,
        bbox=dict(facecolor='white', alpha=0.5))
    print(np.mean(_r2_45),np.mean(_rmse_45))
    plt.tight_layout()
    if save:
        plt.savefig(images_dir+'/'+model_path+str(year)+'_'+str(block_size)+'_'+str(samples_dim[0])+str(samples_dim[1])+'_'+str(k_fold)+phen_str+'.pdf', dpi=300)
    plt.show()
    return _metric, _r2#_y_true, _y_hat