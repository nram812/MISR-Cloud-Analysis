
class Albedo_data:

	def __init__(self, h5py_file):
		import h5py as f
		h5py_file=f.File(h5py_file,'r')
		self.cf_threshold = h5py_file['Cloud_Fraction_thres']
		self.feature_vector = h5py_file['Feature_Vector_24_Components_White_No_Scaling']
		self.lat = h5py_file['Lat']
		self.lon = h5py_file['Lon']
		self.probability = h5py_file['Model_Uncertainty_12_Features_Whiten']
		self.predictions = h5py_file['Model_Pred_24_Features_Whiten']
		self.scaling_params = h5py_file['Scaling_Law_Coefs']
		self.scaling_uncert = h5py_file['Scaling_Law_Var']
		self.h_index = h5py_file['Heterogenity_thres']
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
		self.weighted_height_cf = h5py_file['weighted_mean_1_AverageCloudHeight_cf']
		self.old_prediction = h5py_file['model_2_pred']
		self.mean_max_region_height = h5py_file['mean_MaxRegionalHeightFineResolution_ZeroWind_cf']
		self.sza =h5py_file['sza_albedo']
		self.var_cloud_height_cf = h5py_file['var_AverageCloudHeightAboveSurface_cf']
		self.cloud_fraction_albedo = h5py_file['cloud_fraction_albedo']
		self.var_albedo =h5py_file['variance_l_albedo_albedo']
		self.lat_tc=h5py_file['lat_tc_albedo']
		self.lon_tc=h5py_file['lon_tc_albedo']
		self.year=h5py_file['year']
		self.month=h5py_file['month']
		self.day=h5py_file['day']
		self.file=h5py_file
		self.sst=h5py_file['sst_interpolated']
		self.lts=h5py_file['lts_interpolated']
		self.u_wind=h5py_file['u10_interpolated']
		sza=h5py_file['sza_albedo'][:]
		albedo=h5py_file['average_r_albedo_albedo'][:]
		cf= h5py_file['Cloud_Fraction_thres'][:,1]/1000.0
		height= h5py_file['mean_height_albedo'][:]
		pred=h5py_file['Model_Pred_24_Features_Whiten'][:,-1]


	def close(self):
		self.file.close()
		return None
