#cond=(pred==0)&(height<3500.0)&(cf_res>0.5)&(albedo>0.0)&(h>0.0)
#from class_misr import *
#no need to import modules
#have change the directory for the new computer

f_name = r'C:\Users\Neelesh\Desktop\needs_to_be_gridded_updated_interpolated_f.h5'

f_name=r'needs_to_be_gridded_updated_interpolated_f-DESKTOP-7GJLSIB.h5'
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


"""lets do a quadratic fit"""
def f(x,a,b,c,d,e,f):
    return c*(1-x)+b*x**3+a*x**2+d*x**4+e*x**5+f*x**6
from scipy.optimize import curve_fit


#now lets compute the median albedo curve
cond2=(sza<50.0)&(cf_res>0.05)&(height<3500.0)
#note only observations with SZA less than 40 have been fitted.
a, x, y = binned_statistic_dd(sample=np.vstack([cf_res[cond2]]).T, values=albedo[cond2][::],
                              statistic='mean', bins=[np.arange(0, 1, 0.003)])
bins_cf=x[0]
mask_for_fitting=~np.isnan(a)
params=curve_fit(f,x[0][1:][mask_for_fitting],a[mask_for_fitting])
fitted_data=f(x[0][1:],*params[0])
cf1=x[0][1:]
py.figure()
py.plot(cf1,fitted_data,'r--',label='fitted_data')
py.plot(cf1,a,'b--')

"""now lets examine the residuals"""
fitted_albedo_actual=f(cf_res[cond2],*params[0])
residual=fitted_albedo_actual-albedo[cond2]
"""now lets exxamine the relationship with h_sigma"""
cond3=pred[cond2]==3
a, x, y = binned_statistic_dd(sample=np.vstack([h_i[cond2][cond3],residual[cond3]]).T, values=residual,
                              statistic='count', bins=[np.arange(0, 0.2, 0.005),np.arange(-0.2, 0.2, 0.01)])
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
fig,ax=plt.subplots(1,2)
#a[a<=2]=np.nan
x,y=np.meshgrid(*x)
#py.imshow(a)
c = ax[1].pcolormesh(x.T,y.T,a,
               norm=LogNorm(vmin=2, vmax=a.max()), cmap='jet',alpha=0.5)

#note the above code



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
