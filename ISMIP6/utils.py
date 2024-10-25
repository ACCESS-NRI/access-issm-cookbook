import datetime
import pandas as pd
import xarray as xr
import numpy as np
import os
import pathlib

# Constants
ICE_DENSITY = 0.917 # in Gt / km ^3
WATER_DENSITY = 1.027  # Gt / km ^3
OCEAN_SURFACE_AREA = 361.8 * 10**6  # km²

class AllModels:
    """
    For loading ISMIP6 model outputs. 
    - Searches for desired models, experiments, and variables. 
    - Loads data as xarray dataset, converting to standardised time coordinates.
    - Combines models together into one xarray dataset.
    """
    def __init__(self, path):
        self.path = path
        self.pathdict = self.pathto_dict(path)
        inst_list = [pathlib.PurePath(f).name for f in os.scandir(self.path) if f.is_dir()]
        self.modeldict = {}
        self.modellist = []
        for inst in inst_list:
            mdl_list = [pathlib.PurePath(f).name for f in os.scandir(self.path+"/"+inst) if f.is_dir()]
            self.modeldict[inst] = mdl_list
            for mdl in mdl_list:
                self.modellist.append(inst+'/'+mdl)

    def pathto_dict(self, path, level=0):
        """
        Creates a nested dictionary containing the experiments within each ISMIP6 model.
    
        Parameters
        ----------
        path: str
            The path to directory containing ISMIP6 model output data.
        level: int
            The level of recursion
    
        Returns
        -------
        tree: dict
            Nested dictionary.
        """
        for root, dirs, files in os.walk(path):
            if level == 3:
                tree = {"name": pathlib.PurePath(root).name,"path": root, "type":"experiment", "children":[]}
            elif level == 2:
                tree = {"name": pathlib.PurePath(root).name,"path": root, "type":"model", "children":[]}
            elif level == 1:
                tree = {"name": pathlib.PurePath(root).name,"path": root, "type":"organisation", "children":[]}
            else:
                tree = {"name": pathlib.PurePath(root).name,"path": root, "type":"folder", "children":[]}
            tree["children"].extend([self.pathto_dict(os.path.join(root, d), level=level+1) for d in dirs])
            tree["children"].extend([{"name": f, "path":os.path.join(root, f), "type":"file"} for f in files])
            return tree

    def find_dataset(self, exp, model, org):
        """
        Creates a nested dictionary containing the experiments within each ISMIP6 model.
    
        Parameters
        ----------
        exp: list of str or str
            List of experiments to look for.
        model: str
            model to look for.
        org: str
            institution to look for.
    
        Returns
        -------
        found_dataset: list of dict
            
        """
        mdat = next((sub for sub in self.pathdict['children'] if (sub['name'] == org) & (sub['type']=='organisation')), None)
        if mdat is not None:
            expdat = next((sub for sub in mdat['children'] if (sub['name'] == model) & (sub['type']=='model')), None)
        else:
            raise ValueError('Organisation not found.')
        if expdat is not None:
            dat = []
            if isinstance(exp, list):
                for e in exp:
                    dat_i = next((sub for sub in expdat['children'] if (sub['name'] == e) & (sub['type']=='experiment')), None)
                    if e.startswith('ctrl_proj') & (dat_i is None):
                        dat_i = next((sub for sub in expdat['children'] if ((sub['name'] == 'ctrl_proj') & (sub['type']=='experiment'))), None)
                        dat_i['name'] = e
                        dat.append(dat_i)
                    elif e.startswith('hist') & (dat_i is None):
                        dat_i = next((sub for sub in expdat['children'] if ((sub['name'] == 'hist') & (sub['type']=='experiment'))), None)
                        dat_i['name'] = e
                        dat.append(dat_i)
                    else:
                        dat.append(dat_i)
            else:
                dat = [next((sub for sub in expdat['children'] if (sub['name'] == exp) & (sub['type']=='experiment')), None)]
        else:
            raise ValueError('Model not found.')
        if None not in dat:
            self.found_dataset = dat #[datum['children'] for datum in dat]
        else:
            #print('1 or more experiment not found')
            self.found_dataset = [x for x in dat if x is not None]
            
        return self.found_dataset

    def get_dataset(self, exp, model, org, variables):
        """
        For a given model, load the data and convert coordinates to a standardised form so that it can be combined with and compared to other models.
    
        Parameters
        ----------
        exp: list of str or str
            List of experiments to look for.
        model: str
            model to look for.
        org: str
            institution to look for.
        variables: list of str
            names of variables to load
    
        Returns
        -------
        ds_model: :py:class:`~xarray.DataSet`
            
        """
        self.found_dataset = self.find_dataset(exp, model, org)
        
        if len(self.found_dataset) == 1:
            filename = [var['path'] for var in self.found_dataset[0]['children'] if var['name'].startswith(tuple(variables))]
            ds_exp = xr.open_mfdataset(filename, decode_times=False)
            return self.reassign_coords(ds_exp, org) 
        
        else:
            ds_exps = []
            exp_list = [exp['name'] for exp in self.found_dataset]
            
            for i in range(len(self.found_dataset)):
                variables_org = [var+'_AIS_'+org for var in variables]
                filename = [var['path'] for var in self.found_dataset[i]['children'] if var['name'].startswith(tuple(variables_org))]
                ds_exp = xr.open_mfdataset(filename, decode_times=False)
                ds_exps.append(self.reassign_coords(ds_exp, org)) # some experiment have inconsistent time coords within a given model
            ds_model = xr.concat(ds_exps, pd.Index(exp_list, name='experiment'))
    
        return ds_model

    def get_combined_models(self, org_list, exp, variables):
        """
        Get all standardised datasets for the supplied list of institutions and combine into one dataset
    
        Parameters
        ----------
        org_list: list of str
            List of institutions (and their corresponding models) to include.
        exp: list of str or str
            experiment to include
        variables: list of str
            names of variables to load
    
        Returns
        -------
        ism_dataset: :py:class:`~xarray.DataSet` or list of DataSets of exp starts with 'hist'
            
        """
        ds_models = []
        model_list = []
        hist = False
        if not isinstance(exp, list):
            if exp.startswith('hist'):
                hist = True

        for org in org_list:
            for model in self.modeldict[org]:
                print(org, model)
                if hist:
                    dataset = self.get_dataset_hist(exp, model, org, variables)
                    model_list.append(org+"/"+model)
                    ds_models.append(dataset)
                else:
                    dataset = self.get_dataset(exp, model, org, variables)
                    model_list.append(org+"/"+model)
                    ds_models.append(dataset)
        
        self.modellist = model_list
        if hist:
            return ds_models
        else:
            ism_dataset = xr.concat(ds_models, pd.Index(self.modellist, name='model'))
            return ism_dataset

    def get_dataset_hist(self, exp, model, org, variables):
        self.found_dataset = self.find_dataset(exp, model, org)

        GRID_COUNT = 761 ## this is the same for every model!
        GRID_EXTENT = 3040 # km
        x_km, y_km = np.linspace(-GRID_EXTENT, GRID_EXTENT, num=GRID_COUNT), np.linspace(-GRID_EXTENT, GRID_EXTENT, num=GRID_COUNT)
        #LON, LAT = np.load('ismip6_lonlat_coords.npy')
    
        variables_org = [var+'_AIS_'+org for var in variables]
        filename = [var['path'] for var in self.found_dataset[0]['children'] if var['name'].startswith(tuple(variables_org))]
        ds_exp = xr.open_mfdataset(filename, decode_times=False)
        num_tt = len(ds_exp.time)
        days_per_tt = np.median(np.diff(ds_exp.time))
        days_since = ds_exp.time - min(ds_exp.time)
        t_year = np.arange(num_tt)*days_per_tt/365
        t_year = 2015-t_year[::-1]
        ds_exp = ds_exp.assign_coords(x=("x", x_km),y=("y", y_km),time=("time", t_year))
    
        if days_per_tt < 360:
            if num_tt % 2 != 0:
                ds_exp = ds_exp.isel(time=slice(1,num_tt))
                t_year = t_year[1:]
            #remove 0.5 data points and average 
            a = ds_exp.sel(time=t_year[1::2])
            b = ds_exp.sel(time=t_year[::2])
            a = a.assign_coords(time=("time", t_year[::2]))
            b = b.assign_coords(time=("time", t_year[::2]))
            ab = xr.concat([a,b], pd.Index(['start','mid'], name='halfyear'))
            ds_exp = ab.mean(dim='halfyear')
        
        return ds_exp
        
    def reassign_coords(self, ds_exp, org):
        """
        Make a transformation to dataset to standardise the time coordinates based on the model's institution.
    
        Parameters
        ----------
        ds_exp: :py:class:`~xarray.DataSet`
            Input dataset containing model output
        org: str
            Name of institution
    
        Returns
        -------
        ds_exp: :py:class:`~xarray.DataSet` 
            Dataset after coordinate transformation 
            
        """
        
        GRID_COUNT = 761 ## this is the same for every model!
        GRID_EXTENT = 3040 # km
        x_km, y_km = np.linspace(-GRID_EXTENT, GRID_EXTENT, num=GRID_COUNT), np.linspace(-GRID_EXTENT, GRID_EXTENT, num=GRID_COUNT)
        #LON, LAT = np.load('ismip6_lonlat_coords.npy')
        
        #if org in ['UTAS','UCIJPL','IMAU','ULB','JPL1','DOE']: # 86 times, yearly data since 2015-1-1
        if len(ds_exp.time)==86:
            #IMAU STARTS AT 2015-07-01, ulb starts at 2015-11-05
            num_tt = 86
            t_year = np.arange(num_tt)
            ds_exp = ds_exp.assign_coords(x=("x", x_km),y=("y", y_km),time=("time", t_year))
        
        #elif org in ['VUW','NCAR','VUB']: #87 time points, yearly data since 2015-1-1, drop the extra year
        if len(ds_exp.time)==87:
            num_tt = 87
            t_year = np.arange(num_tt)
            ds_exp = ds_exp.assign_coords(x=("x", x_km),y=("y", y_km),time=("time", t_year))
            ds_exp = ds_exp.isel(time=slice(0,86))
    
        #elif org in ['ILTS_PIK','AWI']: #172 time points starting with 2015-07-01, 6-month data
        if len(ds_exp.time)==172:
            num_tt = 172
            t_year = np.arange(num_tt)/2
            ds_exp = ds_exp.assign_coords(x=("x", x_km),y=("y", y_km),time=("time", t_year))
    
            #remove 0.5 data points and average 
            a = ds_exp.sel(time=t_year[1::2])
            b = ds_exp.sel(time=t_year[::2])
            a = a.assign_coords(time=("time", t_year[::2]))
            b = b.assign_coords(time=("time", t_year[::2]))
            ab = xr.concat([a,b], pd.Index(['start','mid'], name='halfyear'))
            ds_exp = ab.mean(dim='halfyear')
            
        #elif org=='PIK': # 202 time points starting with 2015-07-01, 6-month data, extra 30 points
        if len(ds_exp.time)>=200:
            num_tt = len(ds_exp.time) # 202 in pism1 and 200 in pism2
            t_year = np.arange(num_tt)/2
            
            #drop last 30 points
            ds_exp = ds_exp.assign_coords(x=("x", x_km),y=("y", y_km),time=("time", t_year))
            ds_exp = ds_exp.isel(time=slice(0,172))
            t_year = np.arange(172)/2
    
            #remove 0.5 data points and average
            a = ds_exp.sel(time=t_year[1::2])
            b = ds_exp.sel(time=t_year[::2])
            a = a.assign_coords(time=("time", t_year[::2]))
            b = b.assign_coords(time=("time", t_year[::2]))
            ab = xr.concat([a,b], pd.Index(['start','mid'], name='halfyear'))
            ds_exp = ab.mean(dim='halfyear')
            
        #elif org=='LSCE': #173 time points, 6-month data starting with 2015-01-01, extra point
        if len(ds_exp.time)==173:
            num_tt = 173
            t_year = np.arange(num_tt)/2
    
            #drop last point
            ds_exp = ds_exp.assign_coords(x=("x", x_km),y=("y", y_km),time=("time", t_year))
            ds_exp = ds_exp.isel(time=slice(0,172))
            t_year = t_year[:-1]
    
            #remove 0.5 data points and average
            a = ds_exp.sel(time=t_year[1::2])
            b = ds_exp.sel(time=t_year[::2])
            a = a.assign_coords(time=("time", t_year[::2]))
            b = b.assign_coords(time=("time", t_year[::2]))
            ab = xr.concat([a,b], pd.Index(['start','mid'], name='halfyear'))
            ds_exp = ab.mean(dim='halfyear')

        LON, LAT = np.zeros((761,761)), np.zeros((761,761)) ## we won't be using lon, lat for now
        ds_exp = ds_exp.assign_coords(lat=(("y", "x"), LAT), lon=(("y", "x"), LON))
        return ds_exp

def get_ice_mass_area(lithk, fraction=None):
    """
    Integrate ice thickness over Antarctic ice sheet to get total mass (and also return total area).

    Parameters
    ----------
    lithk: :py:class:`~xarray.DataArray`
        Ice thickness [m] with coords x [m], y[m], time [year]
    fraction:  :py:class:`~xarray.DataSet`(optional)
        Weights to apply to each grid point when integrating.

    Returns
    -------
    ice_mass: :py:class:`~xarray.DataArray` 
        Total ice mass in units of 1e7 Gt
    integrated_area: 
        Total (unmasked) area in units of 1e7 km^2
    """
    
    thickness =  lithk * 1e-3 # convert to km units
    
    ones_array = xr.DataArray(np.ones_like(thickness), coords=thickness.coords, dims=thickness.dims)
    masked_ones = ones_array.where(~np.isnan(thickness))
    
    # Mask the DataArray to replace NaNs with 0
    masked_data = thickness.where(~np.isnan(thickness), other=0)
    if fraction is not None:
        fraction = fraction.where(~np.isnan(fraction), other=0)
        masked_data *= fraction
        masked_ones *= fraction
    
    # Integrate over the x and y coordinates
    dx = np.gradient(thickness.coords['x']) # in km 
    dy = np.gradient(thickness.coords['y']) # in km 
    
    # Create an area array if the grid spacing varies
    integrated_area = (masked_ones * dx[:, np.newaxis] * dy).sum(dim=['x', 'y'])
    integrated_area = integrated_area.compute()/1e7 # in 1e7 km ^2
    
    volume = (masked_data * dx[:, np.newaxis] * dy).sum(dim=['x', 'y'])
    volume = volume.compute() # in km ^3
    
    ice_mass = volume*ICE_DENSITY/1e7 # convert to Gt 1e7 units
    
    return ice_mass, integrated_area


def sea_level_rise(grounded_ice_mass_gt):
    """
    Calculate projected sea level rice from the melting of a certain mass of Antarctic ice.

    Parameters
    ----------
    grounded_ice_mass_gt: :py:class:`~xarray.DataArray`
        Mass of ice in Gt.

    Returns
    -------
    sea_level_rise_mm: :py:class:`~xarray.DataArray` 
        Projected sea level rise in mm.
    """

    # Convert grounded ice mass to water equivalent volume (in km³)
    water_volume_km3 = grounded_ice_mass_gt * (ICE_DENSITY / WATER_DENSITY)

    # Calculate the sea level rise in mm
    sea_level_rise_mm = (water_volume_km3 / OCEAN_SURFACE_AREA) * 10**6  # mm

    return sea_level_rise_mm

def get_smb(acabf, fraction=None):
    """
    Integrate surface mass balance over Antarctic ice sheet.

    Parameters
    ----------
    acabf: :py:class:`~xarray.DataArray`
        Surface mass balance flux [km/m^2/s] with coords x [m], y[m], time [year]
    fraction:  :py:class:`~xarray.DataSet`(optional)
        Weights to apply to each grid point when integrating.

    Returns
    -------
    integrated_smb: :py:class:`~xarray.DataArray` 
        Total surface mass balance in units Gt/year
    """
    
    smb =  acabf * 31536000 * 1000000 / 1e12 # convert from kg/s/m^2 to Gt/yr/km^2 units
    
    ones_array = xr.DataArray(np.ones_like(smb), coords=smb.coords, dims=smb.dims)
    masked_ones = ones_array.where(~np.isnan(smb))
    
    # Mask the DataArray to replace NaNs with 0
    masked_data = smb.where(~np.isnan(smb), other=0)
    if fraction is not None:
        fraction = fraction.where(~np.isnan(fraction), other=0)
        masked_data *= fraction
        masked_ones *= fraction
    
    # Integrate over the x and y coordinates
    dx = np.gradient(smb.coords['x']) # in km 
    dy = np.gradient(smb.coords['y']) # in km 
    
    # Create an area array if the grid spacing varies
    integrated_area = (masked_ones * dx[:, np.newaxis] * dy).sum(dim=['x', 'y'])
    integrated_area = integrated_area.compute()
    
    integrated_smb = (masked_data * dx[:, np.newaxis] * dy).sum(dim=['x', 'y'])
    integrated_smb = integrated_smb.compute() 
        
    return integrated_smb