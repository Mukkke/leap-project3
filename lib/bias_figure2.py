import xarray as xr
import numpy as np

def concat_datasets(mems_dict,recon_output_dir, init_date , fin_date):
    
    datasets_ens = []
    for ens, mem_list in mems_dict.items():
        datasets_member = []
        for n_member, member in enumerate(mem_list):
            # truth_path = f'gs://leap-persistent/abbysh/pco2_all_members_1982-2023/00_regridded_members/{ens}/{member}/{ens}.{member.split("_")[-1]}.Omon.zarr'

            # data_dir = f"{MLinputs_path}/{ens}/{member}"
            # fname = f"MLinput_{ens}_{member.split('_')[-1]}_mon_1x1_{init_date}_{fin_date}.pkl"
            # file_path = f"{data_dir}/{fname}"
            
            recon_dir = f"{recon_output_dir}/{ens}/{member}"    
            # recon_path = f"{recon_dir}/recon_pC02residual_{ens}_{member}_mon_1x1_{init_date}_{fin_date}.zarr"
            recon_path = f"{recon_dir}/recon_pCO2_{ens}_{member}_mon_1x1_{init_date}_{fin_date}.zarr"
            # recon_path = "gs://leap-persistent/Mukkke/pco2_residual/nmse/post02_xgb/reconstructions/CESM2/member_r10i1p1f1/recon_pC02residual_CESM2_member_r10i1p1f1_mon_1x1_200401_202312.zarr"
            kwargs = dict(chunks={'time':-1})
            # truth = xr.open_mfdataset(truth_path, engine='zarr',**kwargs).spco2 ## this is the testbed truth, RAW testbed pco2
            # recon = xr.open_mfdataset(recon_path, engine='zarr',**kwargs).pCO2_recon_unseen ## this is the reconstructed pco2 AFTER pco2-T is added back

            # print(xr.open_zarr(truth_path, consolidated=True))

            truth = xr.open_zarr(recon_path, consolidated=True)["pCO2_truth"]
            recon = xr.open_zarr(recon_path, consolidated=True)["pCO2_recon_unseen"]
            
            # assign_dimension coordinates
            truth = truth.assign_coords(status='truth')
            recon = recon.assign_coords(status='reconstructed')

            # make sure they have the same time coordinates
            # truth = truth.assign_coords(time=recon['time'].data)
            common_time = np.intersect1d(truth['time'], recon['time'])
            truth = truth.sel(time=common_time)
            recon = recon.sel(time=common_time)

            # print(truth.values)
            # print(recon.sel(status="").count())
            
            member_ds = xr.concat([truth, recon], dim='status')
            
            # add member_dimension and coordinate
            member_ds = member_ds.expand_dims({"member": [member]})
            
            # print(member_ds)
            
            datasets_member.append(member_ds)
        
        ds_ens = xr.concat(datasets_member, dim="member")
        ds_ens = ds_ens.expand_dims({"ens": [ens]})       
        
        datasets_ens.append(ds_ens)
        
    ds_concat = xr.concat(datasets_ens, dim='ens') 
    return ds_concat
    

class XarrayEvaluator:
    """
    A class to evaluate reconstructed climate data against truth using xarray.
    
    Supports calculating bias, RMSE, correlation, and more.
    
    Parameters:
    -----------
    ds : xarray.DataArray
        An xarray DataArray with a 'status' coordinate containing 'truth' and 'reconstructed'.
    
    Methods:
    --------
    calculate_bias() -> xarray.DataArray
        Computes bias (Reconstructed - Truth).
    
    calculate_rmse() -> xarray.DataArray
        Computes Root Mean Squared Error (RMSE).
    
    calculate_correlation() -> xarray.DataArray
        Computes correlation along the 'time' dimension.
    
    compute_all_metrics() -> xarray.Dataset
        Returns a dataset containing all computed evaluation metrics.
    """

    def __init__(self, ds):
        """
        Initialize with an xarray DataArray that has a 'status' coordinate.
        """
        self.ds_truth = ds.sel(status='truth')  # Extract truth data
        self.ds_recon = ds.sel(status='reconstructed')  # Extract reconstructed data

    def calculate_bias(self):
        """ Compute Bias: Reconstructed - Truth """
     
        return self.ds_recon - self.ds_truth

    def calculate_rmse(self):
        
        """ Compute Root Mean Squared Error (RMSE) along time dimension """
        return np.sqrt(((self.ds_recon - self.ds_truth) ** 2).mean(dim="time"))

    def calculate_correlation(self):
        """ Compute Pearson Correlation along the time dimension """
        return xr.corr(self.ds_recon, self.ds_truth, dim="time")

    def compute_all_metrics(self):
        """ Compute all evaluation metrics and return an xarray.Dataset """
        # print(self.calculate_bias())
        ds_eval = xr.Dataset({
            "bias": self.calculate_bias(),
            "rmse": self.calculate_rmse(),
            "correlation": self.calculate_correlation()
        })
        return ds_eval

def calc_bias(ens_array,mean_vars=['time', 'status', 'xlon', 'ylat'],ylat_slice=slice(-90,90),time_slice=slice('2022','2022')):
    
    ens_bias = ens_array.diff('status').sel(ens = ens_array.ensemble.values[0],ylat=ylat_slice,time=time_slice).mean(mean_vars)
    bias_ens_df = ens_bias.to_dataframe()
    bias_ens_df_reset = bias_ens_df.reset_index()
    bias_ens_set = bias_ens_df_reset.set_index(['ens','member'])

    return bias_ens_df