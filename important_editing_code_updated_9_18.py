#cond=(pred==0)&(height<3500.0)&(cf_res>0.5)&(albedo>0.0)&(h>0.0)
#from class_misr import *
#no need to import modules
#have change the directory for the new computer

class Albedo_data:

    def __init__(self, h5py_file):
        import h5py
        h5py_file=h5py.File(h5py_file,'r')

        self.cf_threshold = h5py_file['Cloud_Fraction_thres']
        self.feature_vector = h5py_file['Feature_Vector_24_Components_White_No_Scaling']
        self.lat = h5py_file['Lat']
        self.lon = h5py_file['Lon']
        self.probability = h5py_file['Model_Uncertainty_12_Features_Whiten']
        self.predictions = h5py_file['Model_Pred_24_Features_Whiten']
        self.scaling_params = h5py_file['Scaling_Law_Coefs']
        self.scaling_uncert = h5py_file['Scaling_Law_Var']
        self.h_index = h5py_file['Heterogenity_thres']
        # here are the relevant level 1 parameters

        self.overcast = h5py_file['mean_ConsensusOvercastMaskFineResolution_BestWind_cf']
        self.res_corr_cf = h5py_file['mean_A17CorrectedCloudFraction[4]_cf']
        self.n_high_pixels = h5py_file['high_cl_present_albedo']
        self.cloud_edge_var = h5py_file['var_CloudEdgeFraction[4]_cf']
        self.mean_cloud_edge = h5py_file['mean_CloudEdgeFraction[3]_cf']
        self.hom_counts = h5py_file['hom_tc_albedo_albedo']
        self.l_albedo_var = h5py_file['variance_l_albedo_albedo']
        self.var_height_cf = h5py_file['std_AverageCloudHeightAboveSurface_cf']
        self.mean_height_alb = h5py_file['mean_height_albedo']
        self.het_counts = h5py_file['het_albedo_albedo']
        self.mean_l_albedo = h5py_file['average_l_albedo_albedo']
        self.mean_r_albedo = h5py_file['average_r_albedo_albedo']
        self.mean_e_albedo = h5py_file['average_e_albedo_albedo']
        #   self.max_l_albedo = h5py_file['max_alb_l_albedo']
        #   self.max_l_albedo = h5py_file['max_alb_r_albedo']
        self.weighted_height_cf = h5py_file['weighted_mean_1_AverageCloudHeight_cf']
        self.old_prediction = h5py_file['model_2_pred']
        self.mean_max_region_height = h5py_file['mean_MaxRegionalHeightFineResolution_ZeroWind_cf']
        self.sza = h5py_file['sza_albedo']
        self.var_cloud_height_cf = h5py_file['var_AverageCloudHeightAboveSurface_cf']
        self.cloud_fraction_albedo = h5py_file['cloud_fraction_albedo']

        # the above show the important level 2 parameters
        self.lat_tc=h5py_file['lat_tc_albedo']
        self.lon_tc=h5py_file['lon_tc_albedo']

        #the above show the important level 2 parameters

        self.year=h5py_file['year']
        self.month=h5py_file['month']
        self.day=h5py_file['day']




f_name = r'C:\Users\neele\Desktop\PDF_Research_Folder\HeartLab\MISR_Paper\needs_to_be_gridded_updated_interpolated_f-DESKTOP-7GJLSIB.h5'

#f_name=r'needs_to_be_gridded_updated_interpolated_f-DESKTOP-7GJLSIB.h5'
data1 = Albedo_data(f_name)
cf_res=data1.res_corr_cf[:]
cf_thres=data1.cf_threshold[:,2]/1000.0
pred=data1.predictions[:,-1]
height=data1.mean_height_alb[:]
albedo=data1.mean_r_albedo[:]
h_i=data1.hom_counts[:]*1.0/8100.0
sza=data1.sza[:]
h_i=data1.h_index[:,-1]
import pylab as py
import numpy as np
from scipy.stats import binned_statistic_dd

#restrictions on cloud fraction
cond2=(sza<50.0)&(cf_res>-0.01)&(height<3500.0)

# cond3=(sza<50.0)&(cf_res<0.1)&(height<3500.0)
# #len(pred[cond2])*1.0/len(pred[cond3])
# len(pred[cond3])*1.0/len(pred[cond2])

#7.5 percent are closed-cell
#clear sky approximately 1%
#10.4% are open celled
#11.2% are stratus







a, x, y = binned_statistic_dd(sample=np.vstack([cf_res[cond2]]).T, values=albedo[cond2][::],
                              statistic=np.nanmedian, bins=[np.arange(0, 1.006, 0.005)])

a_75, x, y = binned_statistic_dd(sample=np.vstack([cf_res[cond2]]).T, values=albedo[cond2][::],
                                   statistic=lambda a: np.nanpercentile(a,75), bins=[np.arange(0, 1.006, 0.005)])

a_25, x, y = binned_statistic_dd(sample=np.vstack([cf_res[cond2]]).T, values=albedo[cond2][::],
                                 statistic=lambda a: np.nanpercentile(a,25), bins=[np.arange(0, 1.006, 0.005)])

fig,ax=py.subplots(3,figsize=(7,9))
ax[0].plot(x[0][:-1]*100,a,'k-',label='Median Albedo')
ax[0].plot(x[0][:-1][::5]*100,a_75[::5],'k--',alpha=0.3)
ax[0].plot(x[0][:-1][::5]*100,a_25[::5],'k--',alpha=0.3)
ax[0].plot(np.linspace(x[0][0]*100,x[0][-2]*100,10),np.linspace(a[0],a[-1],10),'k-*',label='Linear Relationship')
ax[0].set_xlim(0,100.001)
ax[0].set_xlabel('Cloud Fraction (%)')
ax[0].set_ylabel(r'Domain Albedo ($\alpha$)')
ax[0].legend()
ax[0].set_title('(a)',fontsize=10)
ax[0].set_ylim(0,0.6)
ax[0].set_xticks([0.0,25.0,50.0,75.0,100.0])
ax[0].set_xticklabels('%.0f'% f for f in [0.0,25.0,50.0,75.0,100.0])

cond2=(sza<50.0)&(cf_res>-0.01)&(height<3500.0)
a, x, y = binned_statistic_dd(sample=np.vstack([h_i[cond2]]).T, values=albedo[cond2][::],
                              statistic=np.nanmedian, bins=[np.arange(0, 0.3, 0.003)])

a_75, x, y = binned_statistic_dd(sample=np.vstack([h_i[cond2]]).T, values=albedo[cond2][::],
                                 statistic=lambda a: np.nanpercentile(a,75), bins=[np.arange(0, 0.3, 0.003)])

a_25, x, y = binned_statistic_dd(sample=np.vstack([h_i[cond2]]).T, values=albedo[cond2][::],
                                 statistic=lambda a1: np.nanpercentile(a1,25), bins=[np.arange(0, 0.3, 0.003)])

ax[1].plot(x[0][:-1][::-1],a,'k-',label=r'Median Albedo')
ax[1].plot(x[0][:-1][::][::-1],a_75[::][::],'k--',alpha=0.3)
ax[1].plot(x[0][:-1][::][::-1],a_25[::][::],'k--',alpha=0.3)
#ax[0].plot([x[0][0]*100,x[0][-2]*100],[a[0],a[-1]],'b',label='Linear Relationship')
#ax[0].set_xlim(0,100.001)
ax[1].set_xticks(np.array([0,0.05,0.10,0.15,0.20,0.25,0.30]))
ax[1].set_xticklabels(['%.2f' %f for f in np.array([0,0.05,0.10,0.15,0.20,0.25,0.30])[::-1]])
ax[1].set_xlabel(r'$\bar{H_\sigma}$')
ax[1].set_ylabel(r'Domain Albedo ($\alpha$)')
ax[1].legend()
ax[1].set_title('(b)',fontsize=10)
ax[1].set_ylim(0,0.6)
#py.show()

cond2=(sza<50.0)&(cf_res>-0.01)&(height<3500.0)
a, x, y = binned_statistic_dd(sample=np.vstack([cf_res[cond2]]).T, values=h_i[cond2][::],
                              statistic=np.nanmedian, bins=[np.arange(0, 1.007, 0.007)])

a_75, x, y = binned_statistic_dd(sample=np.vstack([cf_res[cond2]]).T, values=h_i[cond2][::],
                                 statistic=lambda a: np.nanpercentile(a,75), bins=[np.arange(0, 1.007, 0.007)])

a_25, x, y = binned_statistic_dd(sample=np.vstack([cf_res[cond2]]).T, values=h_i[cond2][::],
                                 statistic=lambda a1: np.nanpercentile(a1,25), bins=[np.arange(0, 1.007, 0.007)])

ax[2].plot(x[0][:-1]*100,a,'k-',label=r'Median $\bar{H_\sigma}$')
ax[2].plot(x[0][:-1][::]*100,a_75[::][::],'k--',alpha=0.3)
ax[2].plot(x[0][:-1][::]*100,a_25[::][::],'k--',alpha=0.3)
#ax[0].plot([x[0][0]*100,x[0][-2]*100],[a[0],a[-1]],'b',label='Linear Relationship')
#ax[0].set_xlim(0,100.001)
ax[2].set_xticks([0.0,25.0,50.0,75.0,100.0])
ax[2].set_xticklabels('%.0f'% f for f in [0.0,25.0,50.0,75.0,100.0])
ax[2].set_yticks(np.array([0,0.05,0.10,0.15,0.20,0.25,0.30]))
ax[2].set_yticklabels(['%.2f' %f for f in np.array([0,0.05,0.10,0.15,0.20,0.25,0.30])])
ax[2].set_ylabel(r'$\bar{H_\sigma}$')
ax[2].set_xlabel(r'Cloud Fraction (%)')
ax[2].set_xlim(0,100.001)
ax[2].legend()
ax[2].set_title('(c)',fontsize=10)
py.show()
fig.savefig('Generic_Relationships.pdf',dpi=300)

"""lets do a quadratic fit"""

"""Figure 1: Plot of Domain Albedo as a function of cloud fraction with percentage contours"""
"""Subfigure, the albedo, with reversed lables as a function of heterogeneity. """
#now lets compute the median albedo curve



"""Figure 2: Explore how these relationships are a function of heterogeneity levels """

fig,ax=py.subplots(1,2,figsize=(10,5))
cond2=(sza<50.0)&(cf_res>-0.0001)&(height<3500.0)&(h_i>0.15)
a, x, y = binned_statistic_dd(sample=np.vstack([cf_res[cond2]]).T, values=albedo[cond2][::],
                              statistic=np.nanmedian, bins=[np.arange(-0.07, 1.12, 0.06)])

ax[0].plot(x[0][1:]*100,a,'k-D',label=r'$\bar{H_\sigma}\geq 0.2$')
cond2=(sza<50.0)&(cf_res>-0.0001)&(height<3500.0)&(h_i<0.15)&(h_i>0.1)

a, x, y = binned_statistic_dd(sample=np.vstack([cf_res[cond2]]).T, values=albedo[cond2][::],
                              statistic=np.nanmedian, bins=[np.arange(-0.07, 1.12, 0.06)])


ax[0].plot(x[0][1:]*100,a,'r-s',label=r'$0.10\leq\bar{H_\sigma}\leq 0.15$')

cond2=(sza<50.0)&(cf_res>-0.0001)&(height<3500.0)&(h_i<0.1)&(h_i>0.05)

a, x, y = binned_statistic_dd(sample=np.vstack([cf_res[cond2]]).T, values=albedo[cond2][::],
                              statistic=np.nanmedian, bins=[np.arange(-0.07, 1.12, 0.06)])


ax[0].plot(x[0][1:]*100,a,'b-o',label=r'$0.05\leq\bar{H_\sigma}\leq 0.10$')

cond2=(sza<50.0)&(cf_res>-0.0001)&(height<3500.0)&(h_i<0.05)#&(h_i>0.05)

a, x, y = binned_statistic_dd(sample=np.vstack([cf_res[cond2]]).T, values=albedo[cond2][::],
                              statistic=np.nanmedian, bins=[np.arange(-0.07, 1.12, 0.06)])


ax[0].plot(x[0][1:]*100,a,'g-v',label=r'$\bar{H_\sigma}\leq 0.05$')
cond2=(sza<50.0)&(cf_res>-0.01)&(height<3500.0)

a, x, y = binned_statistic_dd(sample=np.vstack([cf_res[cond2]]).T, values=albedo[cond2][::],
                              statistic=np.nanmedian, bins=[np.arange(0, 1.006, 0.005)])

ax[0].plot(np.linspace(x[0][0]*100,x[0][-2]*100,10),np.linspace(a[0],a[-1],10),'k--',label='Linear Relationship',alpha=0.7)
ax[0].set_xlabel('Cloud Fraction (%)')
ax[0].set_ylabel(r'Domain Albedo ($\alpha$)')
ax[0].set_ylim(0.08,0.45)
ax[0].set_yticks([0.1,0.2,0.3,0.4,0.5])
ax[0].legend()






cond2=(sza<50.0)&(cf_res>0.5)&(height<3500.0)&(pred==0)
a, x, y = binned_statistic_dd(sample=np.vstack([cf_res[cond2]]).T, values=albedo[cond2][::],
                              statistic=np.nanmedian, bins=[np.arange(-0.07, 1.12, 0.06)])

ax[1].plot(x[0][1:]*100,a,'k-P',label=r'CC')
cond2=(sza<50.0)&(cf_res>0.5)&(height<3500.0)&(pred==1)

a, x, y = binned_statistic_dd(sample=np.vstack([cf_res[cond2]]).T, values=albedo[cond2][::],
                              statistic=np.nanmedian, bins=[np.arange(-0.07, 1.12, 0.06)])


ax[1].plot(x[0][1:]*100,a,'r-D',label=r'NC')

cond2=(sza<50.0)&(cf_res>-0.0001)&(height<3500.0)&(pred==2)

a, x, y = binned_statistic_dd(sample=np.vstack([cf_res[cond2]]).T, values=albedo[cond2][::],
                              statistic=np.nanmedian, bins=[np.arange(-0.07, 1.12, 0.06)])


ax[1].plot(x[0][1:]*100,a,'b-s',label=r'DC')

cond2=(sza<50.0)&(cf_res>-0.0001)&(height<3500.0)&(pred==3)#&(h_i>0.05)

a, x, y = binned_statistic_dd(sample=np.vstack([cf_res[cond2]]).T, values=albedo[cond2][::],
                              statistic=np.nanmedian, bins=[np.arange(-0.07, 1.12, 0.06)])


ax[1].plot(x[0][1:]*100,a,'g-x',label=r'OC')
cond2=(sza<50.0)&(cf_res>-0.01)&(height<3500.0)

a, x, y = binned_statistic_dd(sample=np.vstack([cf_res[cond2]]).T, values=albedo[cond2][::],
                              statistic=np.nanmedian, bins=[np.arange(0, 1.006, 0.005)])

ax[1].plot(np.linspace(x[0][0]*100,x[0][-2]*100,10),np.linspace(a[0],a[-1],10),'k--',label='Linear Relationship',alpha=0.7)
ax[1].set_xlabel('Cloud Fraction (%)')
ax[1].set_ylabel(r'Domain Albedo ($\alpha$)')
ax[1].set_ylim(0.08,0.45)
ax[1].set_yticks([0.1,0.2,0.3,0.4,0.5])
ax[1].legend()
ax[1].set_title('(b)')
ax[0].set_title('(a)')
py.show()
fig.savefig('Albedo_as_func_het_presentation.pdf',dpi=300)


"""Time to do a Box Plot"""
self=data1
albedo=self.mean_r_albedo[:]
height=self.weighted_height_cf[:]
pred=self.predictions[:,0]
proba=self.probability[:,0]
hom=self.hom_counts[:]/8100.0
h_index=self.h_index[:,-1]
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# use latex for font rendering

cf=self.res_corr_cf[:]#resolution corrected cf
sza=self.sza[:]
lat=self.lat[:]
lon=self.lon[:]
import matplotlib.pyplot as plt
import numpy as np

fig,ax=plt.subplots(2,1,figsize=(5,7))
ax=ax.ravel()
titles=['(a)','(b)']
tit=['CC','NC','DC','OC']
labels=[r'Albedo ($\alpha$)',r'$\bar{H_{\sigma}}$']

#fig.canvas.set_window_title('A Boxplot Example')
#fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)

data=np.vstack([albedo,h_index])
for i in range(2):
    data_box=[]
    #print i
    for j in range(4):
        #print j
        if j>1.0:
            index=(pred==j)&(sza<50.0)&(height<3500.0)&(data[i]>=-0.01)&(cf>0.05)#&(lat<-40)&(sza<55)
            box=data[i]
            box=np.ma.masked_array(box,mask=np.isnan(box))
        else:
            index=(pred==j)&(sza<50.0)&(height<3500.0)&(data[i]>=-0.01)&(cf>0.5)
            box = data[i]
            box=np.ma.masked_array(box,mask=np.isnan(box))


        data_box.append(box[index])

    bp = ax[i].boxplot(np.array(data_box), notch=0, sym='+', vert=1, whis=1.5)
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='white', marker=None)
    ax[i].set_xlim(0.5, 4.5)
    ax[i].set_xticklabels(tit,
                          rotation=0, fontsize=9)
    ax[i].set_ylabel(labels[i])

    ax[i].set_title(titles[i],fontsize=9)
    ax[i].yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                     alpha=0.5)

ax[1].set_ylim(0.0,0.3)
ax[0].set_ylim(0.0,0.7)
fig.show()
fig.savefig('Box_and_Whisker_Statistics.pdf',dpi=300)




"""Illustrating the separation between the categories"""

cond2=(sza<50.0)&(cf_res>0.9)&(height<3500.0)&(pred==0)&(cf_res<0.91)

cond3=(sza<50.0)&(cf_res>0.9)&(height<3500.0)&(pred==3)&(cf_res<0.91)
import pylab as py

fig,ax=py.subplots(figsize=(5,5))
ax.plot(h_index[cond2],albedo[cond2],'ro',alpha=0.3,label='CC ($90\% \leq f \leq 91\%$)')

ax.plot(h_index[cond3][::],albedo[cond3][::],'kD',alpha=0.3,label='OC ($90 \%\leq f \leq 91\%$)')
ax.set_xlabel(r'$\bar{H_\sigma}$')
ax.set_ylabel(r'Domain Albedo ($\alpha$)')
ax.set_ylim(0.1,0.5)
#ax.set_xticks()
ax.legend()
ax.set_yticks([0.1,0.2,0.3,0.4,0.5])
ax.set_xticks([0.00,0.05,0.10,0.15,0.20])
py.gca().invert_xaxis()
fig.show()


def residual_adjustmnet(cond2=cond2):
    a, x, y = binned_statistic_dd(sample=np.vstack([cf_res[cond2]]).T, values=albedo[cond2][::],
                                  statistic=np.nanmean, bins=[np.arange(-0.004, 1.004, 0.003)])
    bins_cf=x[0]
    mask_for_fitting=~np.isnan(a)
    #params=curve_fit(f,x[0][1:][mask_for_fitting],a[mask_for_fitting])
    from scipy.interpolate import interp1d
    f = interp1d(x[0][1:][mask_for_fitting],a[mask_for_fitting],bounds_error=False,kind='linear')
    pred_albedo=f(cf_res)#,*params[0])
    a, x, y = binned_statistic_dd(sample=np.vstack([cf_res[cond2]]).T, values=pred_albedo[cond2]-albedo[cond2][::],
                                  statistic=np.nanstd, bins=[np.arange(0, 1.06, 0.05)])



    a2, x2, y = binned_statistic_dd(sample=np.vstack([h_index[cond2]]).T, values=-pred_albedo[cond2]+albedo[cond2],
                                    statistic=np.nanmedian, bins=[np.arange(-0.03, 0.3, 0.02)])

    mask_for_fitting=~np.isnan(a2)
    f=interp1d(x2[0][1:][mask_for_fitting],a2[mask_for_fitting],bounds_error=False)
    pred_resid_albedo=f(h_index)
    new_deviation=(albedo[cond2]-pred_albedo[cond2])-pred_resid_albedo[cond2]
    return new_deviation.tolist(),cf_res[cond2].tolist()
cond2=(sza<35.0)&(cf_res>0.1)&(height<3500.0)&(pred==1)
x1,cf1=residual_adjustmnet(cond2)
cond2=(sza<55.0)&(cf_res>0.1)&(height<3500.0)&(pred==2)
x2,cf2=residual_adjustmnet(cond2)

cond2=(sza<55.0)&(cf_res>0.1)&(height<3500.0)&(pred==3)
x3,cf3=residual_adjustmnet(cond2)
cond2=(sza<55.0)&(cf_res>0.1)&(height<3500.0)&(pred==0)
x4,cf4=residual_adjustmnet(cond2)


x_final=np.array(x1+x2+x3+x4)
cf_final=np.array(cf1+cf2+cf3+cf4)

"""Ordinary Least Squares fitting"""

def f(x,a,b,c,d,e,f):
    return c*(1-x)+b*x**3+a*x**2+d*x**4+e*x**5+f*x**6
from scipy.optimize import curve_fit
from scipy.stats import binned_statistic_dd

cond2=(sza<55.0)&(cf_res>0.1)&(height<3500.0)#&(pred==1)
#note only observations with SZA less than 40 have been fitted.
a, x, y = binned_statistic_dd(sample=np.vstack([cf_res[cond2]]).T, values=albedo[cond2][::],
                              statistic=np.nanmean, bins=[np.arange(-0.004, 1.004, 0.003)])
bins_cf=x[0]
mask_for_fitting=~np.isnan(a)
#params=curve_fit(f,x[0][1:][mask_for_fitting],a[mask_for_fitting])
from scipy.interpolate import interp1d
f = interp1d(x[0][1:][mask_for_fitting],a[mask_for_fitting],bounds_error=False,kind='linear')
pred_albedo=f(cf_res)#,*params[0])

fig,ax=plt.subplots(2,figsize=(5,7))
a, x, y = binned_statistic_dd(sample=np.vstack([cf_res[cond2]]).T, values=pred_albedo[cond2]-albedo[cond2][::],
                              statistic=np.nanstd, bins=[np.arange(0, 1.06, 0.05)])



a2, x2, y = binned_statistic_dd(sample=np.vstack([h_index[cond2]]).T, values=-pred_albedo[cond2]+albedo[cond2],
                              statistic=np.nanmedian, bins=[np.arange(-0.03, 0.3, 0.02)])

mask_for_fitting=~np.isnan(a2)
f=interp1d(x2[0][1:][mask_for_fitting],a2[mask_for_fitting],bounds_error=False)
pred_resid_albedo=f(h_index)

# fig=py.figure()
# py.plot(x2[0][1:],f(x2[0][1:],*params[0]),'r--')
# py.plot(x2[0][1:],a2,'b--')
# fig.show()

new_deviation=(albedo[cond2]-pred_albedo[cond2])-pred_resid_albedo[cond2]
a_new, x_new, y = binned_statistic_dd(sample=np.vstack([cf_res[cond2]]).T, values=new_deviation,
                              statistic=np.nanstd, bins=[np.arange(0, 1.06, 0.05)])

a_new1, x_new1, y = binned_statistic_dd(sample=np.vstack([cf_final]).T, values=x_final,
                                      statistic=np.nanstd, bins=[np.arange(0, 1.06, 0.05)])



a25, x2, y = binned_statistic_dd(sample=np.vstack([h_index[cond2]]).T, values=-pred_albedo[cond2]+albedo[cond2],
                                statistic=lambda a1: np.nanpercentile(a1,25), bins=[np.arange(-0.03, 0.3, 0.02)])
a75, x2, y = binned_statistic_dd(sample=np.vstack([h_index[cond2]]).T, values=-pred_albedo[cond2]+albedo[cond2],
                                 statistic=lambda a1: np.nanpercentile(a1,75), bins=[np.arange(-0.03, 0.3, 0.02)])


ax[0].plot(x[0][1:]*100,a,'k-D',label=r'$\sigma_\alpha$',markersize=5)

ax[0].plot(x_new[0][1:]*100,a_new,'k-x',label=r'$\sigma_{\bar{H_\sigma}}$',markersize=5)
ax[0].plot(x_new1[0][1:]*100,a_new1,'r-o',label=r'$\sigma_{MCC}$',markersize=5)
ax[1].plot(x2[0][1:][::-1],a2,'k-D',label='Median Albedo Deviation')
ax[1].plot(x2[0][1:][::-1],a25,'k--')
ax[1].plot(x2[0][1:][::-1],a75,'k--')
ax[1].set_xticks([0.00,0.05,0.10,0.15,0.20,0.25,0.30])
ax[1].set_xticklabels(np.array([0.00,0.05,0.10,0.15,0.20,0.25,0.30])[::-1])
ax[0].set_yticks([0.00,0.02,0.04,0.06,0.08])
ax[0].set_yticklabels([0.00,0.02,0.04,0.06,0.08])
ax[1].set_ylim(-0.05,0.25)

ax[1].set_ylabel(r'Albedo  ($\bar{\alpha})$')
ax[1].set_xlabel(r'$\bar{H_\sigma}$')
ax[1].set_title('(b)',fontsize=10)
ax[1].legend()
ax[0].legend()
ax[0].set_ylim(0,0.085)
ax[0].set_title('(a)',fontsize=10)
ax[0].set_ylabel(r'$\sigma_\alpha$')
ax[0].set_xlabel('Cloud Fraction (%)')

fig.show()

fig.savefig('Residual_adjustment_final.pdf',dpi=300)

"""Then show the result after accounting for this variability"""




"""Detrended variability as a function of heterogeneity and cloud fraction"""




"""Fit the median Curve of Albedo"""

"""Compute and report the deviation in albedo for a fixed cloud fractino"""
"""What is the standard deviation for a fixed cloud fraction"""
"""What are the associations with Heterogeneity and other cloud types"""




"""Illustrate that differences between the heterogeneities are thre reason why the albedos are different for the same fixed cloud fraction"""
"""Do your little plot"""

"""Then do a plot of the role of different cloud morphologies"""


"""Do a plot where you subtract the dependancies, and the last plot should be 
a figure that has the final mae outputs"""




def f(x,a,b,c,d,e,f):
    return c*(1-x)+b*x**3+a*x**2+d*x**4+e*x**5+f*x**6
from scipy.optimize import curve_fit



#note only observations with SZA less than 40 have been fitted.
a, x, y = binned_statistic_dd(sample=np.vstack([cf_res[cond2]]).T, values=albedo[cond2][::],
                              statistic='mean', bins=[np.arange(0, 1, 0.003)])
bins_cf=x[0]
mask_for_fitting=~np.isnan(a)
params=curve_fit(f,x[0][1:][mask_for_fitting],a[mask_for_fitting])

# mask_for_fitting=(~np.isnan(cf_res[cond2]))&(~np.isnan(albedo[cond2]))
# params=curve_fit(f,cf_res[cond2][mask_for_fitting], albedo[cond2][mask_for_fitting])

fitted_data=f(x[0][1:],*params[0])
cf1=x[0][1:]
py.figure()
py.plot(cf1,fitted_data,'r--',label='fitted_data_with_Quartic')
py.plot(cf1,a,'b--',label='Mean cf')
py.legend()
py.xlabel('Cloud fraction')
py.ylabel('Albedo')

"""now lets examine the residuals"""
fitted_albedo_actual=f(cf_res[cond2],*params[0])
residual=fitted_albedo_actual-albedo[cond2]
"""now lets exxamine the relationship with h_sigma"""
cond3=(pred[cond2]==0)&(cf_res[cond2]>0.05)
a_s, x_s, y = binned_statistic_dd(sample=np.vstack([h_i[cond2]]).T, values=residual,
                              statistic='median', bins=[np.arange(0, 0.2, 0.003)])

# a, x, y = binned_statistic_dd(sample=np.vstack([cf_res[cond2][cond3],residual[cond3]]).T, values=residual[cond3],
#                               statistic='count', bins=[np.arange(0, 1, 0.005),np.arange(-0.2, 0.2, 0.01)])
#
# a_s, x_s, y = binned_statistic_dd(sample=np.vstack([cf_res[cond2][cond3]]).T, values=residual[cond3],
#                               statistic='std', bins=[np.arange(0, 1, 0.005)])

a, x, y = binned_statistic_dd(sample=np.vstack([h_i[cond2][cond3],residual[cond3]]).T, values=residual,
                              statistic='count', bins=[np.arange(0, 0.2, 0.003),np.arange(-0.2, 0.2, 0.01)])
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
fig,ax=plt.subplots(2,1,sharey=True,sharex=True)
#a[a<=2]=np.nan
x,y=np.meshgrid(*x)
#py.imshow(a)
c = ax[1].pcolormesh(x.T,y.T,a,
               norm=LogNorm(vmin=2, vmax=a.max()), cmap='jet',alpha=0.5)
ax[0].plot(x_s[0][:-1],a_s)
ax[1].set_xlabel(r'$\bar{H_{\sigma}}$')
ax[0].set_ylabel('Standard Deviation in $H_\sigma$')
ax[1].set_ylabel('Residual Albedo')
#note the above code

def f(x,a,b,c,d,e,f):
    return c*(1-x)+b*x**3+a*x**2+d*x**4+e*x**5+f*x**6
mask_for_fitting=~np.isnan(a_s)
params=curve_fit(f,x_s[0][1:][mask_for_fitting],a_s[mask_for_fitting])
"""here is now the median curve which we will fit again"""

# a, x, y = binned_statistic_dd(sample=np.vstack([h_i[cond2]]).T, values=residual,
fitted_residuals=f(h_i[cond2][cond3],*params[0])

f_2=fitted_residuals=f(x_s[0][1:][mask_for_fitting],*params[0])
py.figure()
py.plot(x_s[0][1:][mask_for_fitting],f_2,label='fitted_residuals')

final_residuals=residual[cond3]-fitted_residuals

a, x, y = binned_statistic_dd(sample=np.vstack([cf_res[cond2][cond3],final_residuals]).T, values=final_residuals,
                              statistic='count', bins=[np.arange(0, 1, 0.005),np.arange(-0.2, 0.2, 0.01)])
a_s, x_s, y = binned_statistic_dd(sample=np.vstack([cf_res[cond2][cond3]]).T, values=final_residuals,
                              statistic=np.nanstd, bins=[np.arange(0, 1, 0.005)])


fig,ax=plt.subplots(1,2)
x,y=np.meshgrid(*x)
c = ax[1].pcolormesh(x.T,y.T,a,
               norm=LogNorm(vmin=2, vmax=a.max()), cmap='jet',alpha=0.5)
c=ax[0].plot(x_s[0][1:],a_s)
#but if we further consider cloud type, we can resolve more variability for an individual cloud class

#then plot the residual variability as a function of f
a, x, y = binned_statistic_dd(sample=np.vstack([cf_res[cond2],final_residuals]).T, values=final_residuals,
                              statistic='count', bins=[np.arange(0, 1, 0.005),np.arange(-0.2,0.2,0.01)])
fig,ax=plt.subplots(1,2)
#a[a<=2]=np.nan
a2, x, y = binned_statistic_dd(sample=np.vstack([cf_res[cond2],final_residuals]).T, values=final_residual,
                              statistic='count', bins=[np.arange(0, 1, 0.005),np.arange(-0.2,0.2,0.01)])

x,y=np.meshgrid(*x)
#py.imshow(a)
c = ax[1].pcolormesh(x.T,y.T,a,
               norm=LogNorm(vmin=2, vmax=a.max()), cmap='jet',alpha=0.5)
c = ax[0].pcolormesh(x.T,y.T,a2,
               norm=LogNorm(vmin=2, vmax=a.max()), cmap='jet',alpha=0.5)

"""This figure should explain the unexplained variabilty. """

"""Now here we can do a final evaluations, where, we say, what is the importance of each factor and the order of 
of the factors. """

#these are sort of the overall evaluations of the topics
#think a little bit about the structyre and stuff. In the overall methodology, when you are explaining cloud morphol\
#ology


# mask_for_fitting=(~np.isnan(cf_res[cond2]))&(~np.isnan(albedo[cond2]))
# params=curve_fit(f,cf_res[cond2][mask_for_fitting], albedo[cond2][mask_for_fitting])

fitted_data=f(x[0][1:],*params[0])

"""We should also consider other avenues, like can we get rid of more variance, when we further subtract
or take cloud category into account. """


"""joint distribution of h and albedo for all cf"""
cond2=(sza<40.0)&(cf_res>0.05)&(height<3500.0)#&(pred==1)
a, x, y = binned_statistic_dd(sample=np.vstack([h_i[cond2],albedo[cond2]]).T, values=albedo[cond2][::],
statistic='count', bins=[np.arange(0, 0.3, 0.004),np.arange(0.05, 0.6, 0.01)])
cond2=(sza<60.0)&(cf_res>0.05)&(height<3500.0)
a, x, y = binned_statistic_dd(sample=np.vstack([cf_res[cond2],h_i[cond2]]).T, values=albedo[cond2][::],
                              statistic='mean', bins=[np.arange(0, 1, 0.01),np.arange(0.01, 0.2, 0.005)])
a, x, y = binned_statistic_dd(sample=np.vstack([cf_res[cond2],albedo[cond2]]).T, values=albedo[cond2][::],
statistic='count', bins=[np.arange(0, 1.0, 0.005),np.arange(0.05, 0.6, 0.01)])
a1, x1, y = binned_statistic_dd(sample=np.vstack([cf_res[cond2]]).T, values=albedo[cond2][::],
statistic=np.nanmedian, bins=[np.arange(0, 1.0, 0.005)])

"""Average albedo first"""
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
fig,ax=plt.subplots(1,2)
#a[a<=2]=np.nan
x,y=np.meshgrid(*x)
#py.imshow(a)
c = ax[1].pcolormesh(x.T,y.T,a,
               norm=LogNorm(vmin=2, vmax=a.max()), cmap='jet',alpha=0.5)
#c2=ax[0].twinx()
a2, x1, y = binned_statistic_dd(sample=np.vstack([cf_res[cond2]]).T, values=albedo[cond2][::],
statistic=np.nanstd, bins=[np.arange(0, 1.0, 0.005)])
ax[1].plot(x1[0][:-1],a1,'k-')
ax[1].plot(x1[0][:-1],a1+a2,'k--')
ax[1].plot(x1[0][:-1],a1-a2,'k--')
ax[0].plot(x1[0][:-1],a2,'b--')
ax3=fig.add_axes([0.95,0.15,0.01,0.7])
fig.colorbar(c,cax=ax3)

#edit all the axis labels and stuff like that


"""On the right hand side we will have the median curve"""


# py.pcolormesh(x.T,y.T,a,cmap='jet')
"""write a code that applies a non-linear correction to account for the effects of cloud heterogeneity"""

from scipy.optimize import curve_fit
def f(x,a,c,d,e,f):
    return (1-x)*a+x**2*c#+d*x**3


"""Albedo with Standard Deviation Plot"""

"""Perhaps compute the median plots for each and the standard deviation for each also. """
from scipy.stats import binned_statistic_dd
cond2=(sza<35.0)&(cf_res>0.05)&(height<3500.0)&(pred==0)#&(sza<65)
a, x, y = binned_statistic_dd(sample=np.vstack([albedo[cond2],cf_res[cond2]]).T, values=albedo[cond2][::],
                              statistic='count', bins=[np.arange(0, 0.6, 0.01),np.arange(0.05, 1, 0.005)])
x,y=np.meshgrid(*x)
a2, x1, y1 = binned_statistic_dd(sample=np.vstack([cf_res[cond2]]).T, values=albedo[cond2][::],
                              statistic=np.nanstd, bins=[np.arange(0.05, 1, 0.01)])
from scipy.optimize import curve_fit
#a2=curve_fit(f,cf_res[cond2],albedo[cond2])
#new_albedo=f(cf_res[cond2],*a2[0])
fig,ax=py.subplots(2)
a[a<3.0]=np.nan
#py.imshow(a)
ax[0].pcolormesh(y.T,x.T,a,cmap='jet')
ax[1].plot(x1[0][:-1],a2,'rx')

cond2=(sza<35.0)&(cf_res>0.05)&(height<3500.0)
a2, x1, y1 = binned_statistic_dd(sample=np.vstack([cf_res[cond2]]).T, values=albedo[cond2][::],
                              statistic=np.nanstd, bins=[np.arange(0.05, 1, 0.01)])
ax[1].plot(x1[0][:-1],a2,'bx')
"""Albedo and H _simga"""
py.figure()
cond2=(sza>45.0)&(cf_res>0.95)&(height<3500.0)&(sza<55)
a, x, y = binned_statistic_dd(sample=np.vstack([h_i[cond2],albedo[cond2]]).T, values=albedo[cond2][::],
                              statistic='count', bins=[np.arange(0.0,0.3,0.005),np.arange(0, 0.8, 0.01)])
x,y=np.meshgrid(*x)
py.pcolormesh(x.T,y.T,a,cmap='jet')

cond2=(sza<50.0)&(cf_res>0.05)&(height<3500.0)


a, x, y = binned_statistic_dd(sample=np.vstack([cf_res[cond2]]).T, values=albedo[cond2][::],
                              statistic=np.nanstd, bins=[np.arange(0.05, 1, 0.008)])
a1, x, y = binned_statistic_dd(sample=np.vstack([cf_res[cond2]]).T, values=albedo[cond2][::],
                              statistic='mean', bins=[np.arange(0.05, 1, 0.008)])
# py.plot(x[0][:-1],a1-a,'k--')
# py.plot(x[0][:-1],a1+a,'k--')
cond=(cf_res>0.95)&(height<3500.0)&(sza<50.0)
pred4=len(pred[cond])
pred1=pred[cond].tolist().count(0)*1.0/pred4
pred2=pred[cond].tolist().count(1)*1.0/pred4
pred3=pred[cond].tolist().count(2)*1.0/pred4
pred5=pred[cond].tolist().count(3)*1.0/pred4
pred4=len(pred[(cf_res>0.85)&(height<3500.0)])
print pred1,pred2,pred3,pred5

"""Construct a figure that outlines the RFO as a function of cloud fraction or something?"""


"""unresolved deviations. """
"""The standard deviation sort of highlights the unresolved variabilit."""
py.figure()
#py.plot(x[0][:-1],a1,'k-')
py.plot(x[0][:-1],a,'k--')
#py.plot(x[0][:-1],a1+a,'k--')
cond2=(sza<50.0)&(cf_res>0.05)&(height<3500.0)&((pred==2)|(pred==2))
a1, x, y = binned_statistic_dd(sample=np.vstack([cf_res[cond2]]).T, values=albedo[cond2][::],
                              statistic='mean', bins=[np.arange(0.05, 1, 0.008)])
a, x, y = binned_statistic_dd(sample=np.vstack([cf_res[cond2]]).T, values=albedo[cond2][::],
                              statistic=np.nanstd, bins=[np.arange(0.05, 1, 0.008)])

#py.plot(x[0][:-1],a1,'r-')
py.plot(x[0][:-1],a,'r--')
cond2=(sza<50.0)&(cf_res>0.05)&(height<3500.0)&((pred==3)|(pred==3))
a1, x, y = binned_statistic_dd(sample=np.vstack([cf_res[cond2]]).T, values=albedo[cond2][::],
                              statistic='mean', bins=[np.arange(0.05, 1, 0.008)])
a, x, y = binned_statistic_dd(sample=np.vstack([cf_res[cond2]]).T, values=albedo[cond2][::],
                              statistic=np.nanstd, bins=[np.arange(0.05, 1, 0.008)])

#py.plot(x[0][:-1],a1,'r-')
py.plot(x[0][:-1],a,'g--')
cond2=(sza<50.0)&(cf_res>0.5)&(height<3500.0)&((pred==1)|(pred==1))
a1, x, y = binned_statistic_dd(sample=np.vstack([cf_res[cond2]]).T, values=albedo[cond2][::],
                              statistic='mean', bins=[np.arange(0.05, 1, 0.008)])
a, x, y = binned_statistic_dd(sample=np.vstack([cf_res[cond2]]).T, values=albedo[cond2][::],
                              statistic=np.nanstd, bins=[np.arange(0.05, 1, 0.008)])
py.plot(x[0][:-1],a,'b--')

cond2=(sza<35.0)&(cf_res>0.7)&(height<3500.0)&((pred==0)|(pred==0))
a1, x, y = binned_statistic_dd(sample=np.vstack([cf_res[cond2]]).T, values=albedo[cond2][::],
                              statistic='mean', bins=[np.arange(0.05, 1, 0.008)])
a, x, y = binned_statistic_dd(sample=np.vstack([cf_res[cond2]]).T, values=albedo[cond2][::],
                              statistic=np.nanstd, bins=[np.arange(0.05, 1, 0.008)])

py.plot(x[0][:-1],a,'y--')
cond2=(sza<50.0)&(cf_res>0.5)&(height<3500.0)&((pred==0)|(pred==1))
a1, x, y = binned_statistic_dd(sample=np.vstack([cf_res[cond2]]).T, values=albedo[cond2][::],
                              statistic='mean', bins=[np.arange(0.05, 1, 0.008)])
a, x, y = binned_statistic_dd(sample=np.vstack([cf_res[cond2]]).T, values=albedo[cond2][::],
                              statistic=np.nanstd, bins=[np.arange(0.05, 1, 0.008)])

#py.plot(x[0][:-1],a1,'r-')
py.plot(x[0][:-1],a,'b--')
#py.plot(x[0][:-1],a1+a,'r--')
"""variance in albedo increases when albedo increases"""
"""low albedo, low cloud fraction has a large portion of unexplained variability."""
"""Subtract the residual variability. """


"""create the plot, of sigma vs cloud fraction and mean for each cloud type."""


"""What happens when we group the clouds into their four regimes, how much does this reduce the variabilty"""
py.plot(x[0][:-1],a1-a,'k--')
py.plot(x[0][:-1],a1+a,'k--')
h=data1.h_index[:,-1]
cond=(pred==3)&(height<3500.0)&(cf_res>0.5)&(albedo>0.0)&(h>0.0)&(cf_res<0.6)&(sza>40.0)&(sza<50.0)
py.figure()
py.plot(h[cond][::5],albedo[cond][::5],'ro')
cond=(pred==3)&(height<3500.0)&(cf_res>0.2)&(albedo>0.0)&(h>0.0)&(cf_res<0.5)&(sza>40.0)&(sza<50.0)
py.plot(h[cond][::5],albedo[cond][::5],'bo')
np.corrcoef(h[cond],albedo[cond])


"""Some of this variability can be reduced by incorparting knowledge of cloud heterogeneity. """









"""Building a predictive model based on the input albedo"""

from keras.models import Sequential
from keras.layers import Dense
model2=Sequential()
model2.add(Dense(32,input_shape=(3,)))
model2.add(Dense(16,activation='relu'))
model2.add(Dense(1,activation='linear'))
model2.compile(loss='mse',optimizer='adam',metrics=['mae','mse'])
from sklearn.model_selection import train_test_split

data1 = Albedo_data(f_name)
cf_res=data1.res_corr_cf[:]
cf_thres=data1.cf_threshold[:,2]/1000.0
pred=data1.predictions[:,-1]
height=data1.mean_height_alb[:]
albedo=data1.mean_r_albedo[:]
h_i=data1.hom_counts[:]*1.0/8100.0
sza=data1.sza[:]
h_i=data1.h_index[:,-1]

cf_res=data1.res_corr_cf[:]

pred=data1.predictions[:,-1]
albedo=data1.mean_r_albedo[:]
h_i=data1.h_index[:,-1]
import pylab as py
import numpy as np
from scipy.stats import binned_statistic_dd

#restrictions on cloud fraction
cond2=(sza<50.0)&(cf_res>0.2)&(height<3500.0)&(albedo>0.1)&(h_i>0.0000)&(pred>-1)

mean2=np.nanmean(albedo[cond2])
z=np.nanmean((albedo[cond2]-mean2)**2)


feature_vector=np.vstack([cf_res,pred,h_i])[:,cond2].T
from scipy.stats import linregress
a=linregress(cf_res[cond2],albedo[cond2])

pred_alb=a.slope*cf_res[cond2]+a.intercept

z=np.mean(abs(pred_alb-albedo[cond2])**2)
#rmse==0.06






bins=np.arange(0,1,0.05)
def draw_samples(albedo,bins,cond2):
    albedo_bins=np.digitize(albedo[cond2],bins=bins)
    z=np.bincount(albedo_bins)
    train_ind=[]
    test_ind=[]
    thres=1000
    for i in range(len(bins)):
        c=np.where(albedo_bins==i)[0]
        train_ind+=c[:thres].tolist()
        test_ind+=c[thres:].tolist()
    return train_ind,test_ind
train_ind,test_ind=draw_samples(albedo,bins,cond2)
model=keras.models.load_model(r'C:\Users\neele\Desktop\PDF_Research_Folder\HeartLab\MISR_Paper\best_model_albedo_prediction_val_simple.h5')
#x_train,x_test,y_train,y_test=train_test_split(feature_vector,albedo[cond2],train_size=0.8)

x_train=feature_vector[train_ind]
x_test=feature_vector[test_ind]
y_train=albedo[cond2][train_ind]
y_test=albedo[cond2][test_ind]
from sklearn.preprocessing import StandardScaler
scale=StandardScaler()
x_train_s=scale.fit_transform(x_train)
model.fit(x_train_s,y_train*100,epochs=300,batch_size=5000,validation_data=(scale.transform(x_test),y_test*100),verbose=2)


y_pred=model.predict(scale.transform(x_test))

fig=py.figure()
py.plot(y_pred[::10],y_test[::10]*100,'rx')
from scipy.stats import linregress
a=linregress(y_pred.ravel(),y_test*100)




from keras.layers import Input,concatenate,Dense
from keras.models import Model
import keras
#restrictions on cloud fraction
cond2=(sza<50.0)&(cf_res>-0.01)&(height<3500.0)&(albedo>0.0000)&(h_i>0.0000)&(pred>-1)

feature= np.vstack([cf_res,h_i])[:,cond2].T
pred_cate = keras.utils.to_categorical(pred[cond2])
y= albedo[cond2]

float_input = Input(shape=(2, ))
one_hot_input = Input(shape=(4,) )

first_dense = Dense(32,activation='relu')(float_input)
second_dense = Dense(32,activation='relu')(one_hot_input)

merge_one = concatenate([first_dense, second_dense])
dense_inner = Dense(10,activation='relu')(merge_one)
dense_output = Dense(1,activation='linear')(dense_inner)


model_en = Model(inputs=[float_input, one_hot_input], outputs=dense_output)


model_en.compile(loss='mean_squared_error',
              optimizer='Nadam',
              metrics=['mae'])

model_en.summary()

model_en.fit([feature,pred_cate], y*100, epochs=500,batch_size=1000,verbose=2,validation_split=0.1)
model_en.evaluate([feature[test_ind],pred_cate[test_ind]], y[test_ind]*100)

checkpoint = ModelCheckpoint(filepath, monitor='val_iou_score', verbose=1, save_best_only=True, mode='max')

model1.summary()
from keras.callbacks import ModelCheckpoint
#history= AdditionalValidationSets(model1.input_shape,[model_object.test_gen])
model_object.train_model(model1,callbacks=[checkpoint])
# model_object=Custom_Resnet_Scaling_Unet(base_shape=(224,224
#training_mae of 2.59 (12.42)
#validation mae of 2.38 (10.6)
from matplotlib import ticker, cm
predicted_albedo=model_en.predict([feature[test_ind],pred_cate[test_ind]])/100.0
a, x1, y1 = binned_statistic_dd(sample=np.vstack([y[test_ind],predicted_albedo.ravel()]).T, values=predicted_albedo.ravel(),
                              statistic='count', bins=[np.arange(0, 0.7, 0.005),np.arange(0, 0.7, 0.005)])
X,Y=np.meshgrid(*x1)

a[a<3]=np.nan

from scipy.stats import linregress
y_=linregress(y[test_ind],predicted_albedo[:,0])
fig,ax=plt.subplots()
range1=np.arange(0,0.7,0.01)
im=ax.pcolor(X,Y,a,cmap='jet',norm=matplotlib.colors.LogNorm(),alpha=0.6)
cbar=fig.colorbar(im)
cbar.set_label('Counts',fontsize=10)

#a#x.plot(y[test_ind][::50],predicted_albedo[::50]/100.0,'ko')
ax.plot(range1,range1*y_.slope+y_.intercept,color='k',label='$r^2=%.2f, \hspace{0.1} RMSE=0.032 $' %(y_.rvalue**2))
#ax.plot(range1,range1,'k--',label='y=x')


ax.set_xlabel('True Albedo',fontsize=10)
ax.set_ylabel('Predicted Albedo',fontsize=10)
ax.tick_params(labelsize=10)
#ax.plot(range1,range1*y_.slope+y_.intercept,color='r',label='y=%.2f x +%.2f, $\hspace{0.1} r^2=%.2f, \hspace{0.1} RMSE=0.032 $' %(y_.slope,y_.intercept,y_.rvalue**2))
ax.legend()
fig.savefig('predicted_2.pdf',dpi=100)