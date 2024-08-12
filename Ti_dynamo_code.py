import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.interpolate import interp1d
from math import pi
import random
#import warnings
#warnings.filterwarnings('ignore')

class Diapir:
    def __init__(self, maxD, sdD, fixed=False):
        """
        Initialize the Diapir object with specified parameters.

        Args:
            maxD (int): The maximum number of diapirs that can form.
            sdD (float): The standard deviation for the diapir size distribution.
            fixed (bool): Whether to use a fixed random seed for reproducibility.
        """
        self.maxD = maxD
        self.sdD = sdD
        self.fixed = fixed

    def diapir_size(self, mu=40):
        """
        Generate the cumulative distribution function (CDF) for diapir sizes based on a normal distribution.

        Args:
            mu (float, optional): The mean of the normal distribution for diapir sizes. Default is 40 km.

        Returns:
            tuple: Arrays representing the x-values (diapir sizes) and the normalized CDF values.
        """
        x = np.linspace(0, 40, 1000)
        pdf_values = norm.pdf(x, loc=mu, scale=self.sdD)
        pdf_values_reversed = 1 - pdf_values / max(pdf_values)
        cdf_values = np.cumsum(pdf_values_reversed)
        cdf_values_norm = cdf_values - min(cdf_values)
        cdf_values_norm = cdf_values_norm / max(cdf_values_norm)

        self.x = x
        self.cdf_values_norm = cdf_values_norm

        return x, cdf_values_norm

    def diapir_generate(self):
        """
        Generate a random set of diapirs based on the maximum number of diapirs and their size distribution.

        Returns:
            tuple: Arrays of sampled diapir radii, the maximum diapir radius and the number of diapirs generated.
        """
        if self.fixed:
            rnd = np.random.RandomState(42)
            num_samples = rnd.randint(0, self.maxD)
            num_samples = rnd.choice([0, num_samples])
        else:
            num_samples = np.random.randint(0, self.maxD)
            num_samples = np.random.choice([0, num_samples])

        try:
            _ = self.cdf_values_norm
        except AttributeError:
            self.diapir_size()

        interp_pdf = interp1d(self.cdf_values_norm, self.x)

        if self.fixed:
            rnd = np.random.RandomState(42)
            sampled_x = rnd.uniform(0, 1, num_samples)
        else:
            sampled_x = np.random.uniform(0, 1, num_samples)

        self.sampled_diapir_radius = interp_pdf(sampled_x)

        if num_samples == 0:
            self.sampled_diapir_radius = np.array([0])

        self.max_diapir_radius = np.max(self.sampled_diapir_radius)
        self.num_samples = num_samples

        return self.sampled_diapir_radius,  self.max_diapir_radius, self.num_samples

    def diapir_dimensions(self):
        """
        Calculate the dimensions of diapirs including their total volume, equivalent radius, and height at the core-mantle boundary (CMB).

        Returns:
            tuple: Total volume of diapirs, equivalent radius, and height of diapirs at the core-mantle boundary (CMB).
        """
        core_radius = 350  # Core radius in kilometers
        core_volume = (4 / 3) * pi * core_radius ** 3

        try:
            _ = self.sampled_diapir_radius
        except AttributeError:
            self.diapir_generate()

        sampled_diapir_volume = (4 / 3) * pi * self.sampled_diapir_radius ** 3

        self.sum_diapir_volume = np.sum(sampled_diapir_volume)

        self.total_diapir_radius = ((3 / (4 * pi)) * self.sum_diapir_volume) ** (1 / 3)

        big_volume = self.sum_diapir_volume + core_volume
        big_radius = ((3 / (4 * pi)) * big_volume) ** (1 / 3)
        self.sum_height = big_radius - core_radius

        return self.sum_diapir_volume, self.total_diapir_radius, self.sum_height



def melting_time(diapir, method, tOff, tOn=None, constant=None, fixed=False):
    """
    Calculate the melting time of a diapir based on the specified method and parameters.

    Args:
        diapir (Diapir): An instance of the Diapir class, which contains information about diapir dimensions and volumes.
        method (str): The method to use for calculating melting time. Options are 'height', 'volume', and 'independent'.
        tOff (list or np.ndarray): The length of random dynamo off periods to choose from, used if no diapirs are present.
        tOn (float, optional): The time for which the dynamo is on, required if method is 'independent'.
        constant (float, optional): The constant used for calculation depending on the method ('height' or 'volume').
        fixed (bool, optional): Whether to use a fixed random seed for reproducibility.

    Returns:
        float: The calculated melting time for the diapir.

    Raises:
        ValueError: If required parameters are not provided for the selected method.
    """
    if method == 'height':
        if constant is not None:
            t_melt = constant * diapir.sum_height
        else:
            raise ValueError(f'You picked {method} method but did not set constant parameter')

    elif method == 'volume':
        # Ensure diapir has generated its sizes
        if diapir.max_diapir_radius is None:
            diapir.diapir_generate()  # Ensure the maximum diapir radius is calculated

        max_vol = (4 / 3) * pi * diapir.max_diapir_radius ** 3  # Volume in km^3
        if constant is not None:
            t_melt = constant * max_vol
        else:
            raise ValueError(f'You picked {method} method but did not set constant parameter')

    elif method == 'independent':
        if tOn is not None:
            if fixed:
                rnd = np.random.RandomState(42)
                t_melt = rnd.randint(1, 11) * tOn  # Random integer between 1 and 10
            else:
                t_melt = np.random.randint(1, 11) * tOn  # Random integer between 1 and 10
        else:
            raise ValueError(f'You picked {method} method but did not set tOn parameter')

    else:
        raise ValueError(f'Unknown method: {method}')

    # Check if no diapirs are present and select an off period if needed
    if diapir.num_samples == 0:
        if fixed:
            rnd = np.random.RandomState(42)
            t_melt = rnd.choice(tOff)  # Length of random dynamo off period
        else:
            t_melt = np.random.choice(tOff)  # Length of random dynamo off period

    # Ensure t_melt is not zero
    if t_melt == 0:
        t_melt = 1

    return t_melt





class FieldCalc:
    def __init__(self, t_melt, diapir):
        """
        Initialize the FieldCalc class.

        Args:
            t_melt (float): The melting time of the diapir in thousands of years.
            diapir (Diapir): An instance of the Diapir class, which contains information about diapir dimensions and volumes.
        """
        self.t_melt = t_melt
        self.diapir = diapir

    def heat_flux_calc(self):
        """
        Calculate the heat flux across the core-mantle boundary (CMB) based on the diapir properties and melting time.

        Returns:
            float: The calculated heat flux across the CMB in W/m^2.
        """
        core_radius = 350  # Core radius in km
        diapir_density = 4000  # Density of each diapir in kg/m^3
        latent = 300e3  # Latent heat of fusion for the mantle in J/kg

        # Convert t_melt from thousands of years to seconds
        t_melt_seconds = self.t_melt * 1000 * 365 * 24 * 3600

        # Calculate the heat flux across the CMB
        Q_cmb = ((self.diapir.total_diapir_radius * 1e3) ** 3 * diapir_density * latent) / (
            3 * (core_radius * 1e3) ** 2 * t_melt_seconds
        )  # Diapir and core radii converted to meters

        if self.diapir.num_samples == 0:
            Q_cmb = 0  # Introduces random off periods

        self.Q_cmb = Q_cmb

        return Q_cmb

    def field_calc(self):
        """
        Calculate the magnetic field strength based on the calculated heat flux.

        Returns:
            float: The calculated magnetic field strength in microteslas (µT).
        """
        field_constant = 2e-5
        fdip = 1 / 7  # The fraction of the magnetic field that is dipolar

        # Calculate the surface magnetic field
        try:
            _ = self.Q_cmb
        except AttributeError:
            self.heat_flux_calc()
        B_surface = field_constant * fdip * self.Q_cmb ** (1 / 3)
        B_microT = B_surface * 1e6
        if self.diapir.num_samples == 0:
            B_microT = 0  # Introduces random off periods

        self.B_microT = B_microT

        return B_microT

    
class DynamoGenerate:
    def __init__(self, maxD, sdD, tOff, method, constant=None, tOn=None, fixed=False):
        """
        Initialize the DynamoGenerate class for simulating and analyzing dynamo activity.

        Args:
            maxD (int): Maximum number of diapirs that can be generated.
            sdD (float): Standard deviation for diapir size distribution.
            tOff (list or array-like): List of possible off periods for the dynamo.
            method (str): Method used for calculating melting time ('height', 'volume', or 'independent').
            constant (float, optional): Constant used in melting time calculation if method is 'height' or 'volume'.
            tOn (float, optional): Time period for the 'independent' method, if applicable.
            fixed (bool, optional): If True, use a fixed random seed for reproducibility.
        """
        self.maxD = maxD
        self.sdD = sdD
        self.tOff = tOff
        self.method = method
        self.constant = constant
        self.tOn = tOn
        self.fixed = fixed

    def dynamo(self):
        """
        Run a single simulation of dynamo activity.

        Returns:
            tuple: A tuple containing:
                - Diapir: An instance of the Diapir class with generated dimensions.
                - float: Melting time of the diapir.
                - FieldCalc: An instance of the FieldCalc class with calculated magnetic field.
        """
        # Generate diapirs and calculate their dimensions
        d = Diapir(self.maxD, self.sdD)
        d.diapir_dimensions()

        # Calculate melting time of diapirs
        melt_t = melting_time(d, self.method, self.tOff, self.tOn, self.constant, self.fixed)

        # Calculate CMB heat flux and generated surface field
        field = FieldCalc(melt_t, d)
        field.field_calc()

        return d, melt_t, field

    def dynamo_history(self):
        """
        Create a synthetic dynamo history until the Ti cumulates are exhausted.

        Returns:
            tuple: A tuple containing:
                - float: Percentage of time the dynamo was active.
                - float: Total time in thousands of years.
                - float: Average magnetic field strength in microteslas.
                - float: Maximum magnetic field strength in microteslas.
                - float: Minimum magnetic field strength in microteslas.
                - float: Average melting time in thousands of years.
                - float: Average off period time in thousands of years.
                - float: Average volume of diapirs in km^3.
        """
        diapir_list = []
        melt_t_list = []
        field_list = []
        tOff_list = []
        vol_list = []

        total_time = 0
        total_time_on = 0
        total_diapir_volume = 0  # Keeps track of how much Ti cumulate material we have left

        for x in np.arange(1, 100000):
            d, melt_t, field = self.dynamo()

            if d.num_samples > 0:
                total_time_on += melt_t
                melt_t_list.append(melt_t)
                field_list.append(field.B_microT)

            if d.num_samples == 0:
                tOff_list.append(melt_t)

            vol_list.append(d.sum_diapir_volume)
            diapir_list.append(d)

            total_time += melt_t
            total_diapir_volume += d.sum_diapir_volume

            if total_diapir_volume >= 520e6:  # Stops the loop when Ti cumulates are exhausted
                break

        percent_on = total_time_on / total_time * 100
        total_on = total_time
        av_B = np.mean(field_list)
        max_B = np.max(field_list)
        min_B = np.min(field_list)
        av_melt = np.mean(melt_t_list)
        tOff_av = np.mean(tOff_list)
        av_vol = np.mean(vol_list)

        return percent_on, total_on, av_B, max_B, min_B, av_melt, tOff_av, av_vol

    def plot_int_vs_time(self, save=None):
        """
        Create plots of paleointensity vs. time and additional information.

        Args:
            save (str, optional): If specified, save the plot to a file with this name.
        """
        self.save = save
        core_radius = 350  # Core radius in km
        core_volume = (4 / 3) * pi * core_radius ** 3

        # Create empty lists to store values
        B_microT_list = []
        time_list = []
        height_list = []
        volume_list = []

        total_time = 0
        total_diapir_volume = 0
        total_height = 0  # Keeps track of the thickness of the Ti cumulate layer

        for x in np.arange(1, 1000):
            d, melt_t, field = self.dynamo()

            # Append the values to the lists
            B_microT_list.append(field.B_microT)
            time_list.append(total_time)
            volume_list.append(total_diapir_volume)
            height_list.append(total_height)

            total_time += melt_t
            total_diapir_volume += d.sum_diapir_volume

            big_volume = total_diapir_volume + core_volume
            big_radius = ((3 / (4 * pi)) * big_volume) ** (1 / 3)
            total_height = big_radius - core_radius

            # Append the updated values to the lists
            B_microT_list.append(field.B_microT)
            time_list.append(total_time)
            volume_list.append(total_diapir_volume)
            height_list.append(total_height)

            if total_diapir_volume >= 520e6:  # Stops the loop when Ti cumulates are exhausted
                break

        # Create arrays for plotting
        B_microT_array = np.array(B_microT_list)
        time_array = np.array(time_list)
        height_array = np.array(height_list)
        volume_array = np.array(volume_list)

        # Convert time to Myr
        time_array = time_array / 1e3

        # Create plots
        fig, (ax1, ax3) = plt.subplots(ncols=2, figsize=(20, 6))

        # First subplot: Intensity vs. Time and Height vs. Time
        ax1.plot(time_array, B_microT_array, color='black', marker='o', linestyle='-')
        ax1.set_xlabel("Time (Myr)")
        ax1.set_ylabel("Intensity ($\mu$T)", color='black')
        ax1.tick_params(axis='y', labelcolor='black')
        ax1.set_title("Lunar Paleomagnetic Record")
        ax1.grid()

        ax2 = ax1.twinx()
        ax2.plot(time_array, height_array, color='blue', marker='s', linestyle='--')
        ax2.set_ylabel("Height (km)", color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')

        # Second subplot: Intensity vs. Time and Volume vs. Time
        ax3.plot(time_array, B_microT_array, color='black', marker='o', linestyle='-')
        ax3.set_xlabel("Time (Myr)")
        ax3.set_ylabel("Intensity ($\mu$T)", color='black')
        ax3.tick_params(axis='y', labelcolor='black')
        ax3.set_title("Lunar Paleomagnetic Record (Volume)")
        ax3.grid()

        ax4 = ax3.twinx()
        ax4.plot(time_array, volume_array, color='red', marker='s', linestyle='--')
        ax4.set_ylabel("Volume (km$^3$)", color='red')
        ax4.tick_params(axis='y', labelcolor='red')

        plt.tight_layout()

        if self.save is not None:
            plt.savefig(self.save + '.svg')

        plt.show()

    def loop_dynamo(self, method_value, maxD_values, tOff_values, sdD_values, tOn_values=None, constant_values=None, save_output=False, outfile_name=None):
        """
        Run simulations over a range of parameter values and collect the results.

        Args:
            method_value (str): Method used for calculating melting time ('height', 'volume', or 'independent').
            maxD_values (list): List of values for maximum diapirs.
            tOff_values (list): List of values for off periods.
            sdD_values (list): List of values for standard deviation of diapir size distribution.
            tOn_values (list, optional): List of values for 'tOn' if using 'independent' method.
            constant_values (list, optional): List of constant values if using 'height' or 'volume' method.
            save_output (bool, optional): If True, save the results to a file.
            outfile_name (str, optional): File name to save the results.

        Returns:
            np.ndarray: Array of simulation results for different parameter sets.
        """
        properties_list = []

        for self.maxD in maxD_values:
            for self.tOff in tOff_values:
                for self.sdD in sdD_values:
                    if method_value in ['height', 'volume']:
                        for self.constant in constant_values:
                            self.maxD = int(self.maxD)
                            self.tOff = int(self.tOff)

                            properties = self.dynamo_history()
                            properties_list.append(properties)

                    elif method_value == 'independent':
                        for self.tOn in tOn_values:
                            self.maxD = int(self.maxD)
                            self.tOff = int(self.tOff)

                            properties = self.dynamo_history()
                            properties_list.append(properties)

        self.properties_array = np.array(properties_list)

        self.dynamo_loop_properties()
        self.dynamo_loop_properties_dataframe()

        if save_output:
            if outfile_name is not None:
                self.properties_df.to_csv(outfile_name, index=False)
            else:
                raise ValueError('Please provide an outfile_name to save the dataframe')

        return self.properties_array

    def loop_dynamo_parameters(self, method_value, maxD_values, tOff_values, sdD_values, tOn_values=None, constant_values=None, save_output=False, outfile_name=None):
        """
        Create a DataFrame of parameter combinations used in simulations.

        Args:
            method_value (str): Method used for calculating melting time ('height', 'volume', or 'independent').
            maxD_values (list): List of values for maximum diapirs.
            tOff_values (list): List of values for off periods.
            sdD_values (list): List of values for standard deviation of diapir size distribution.
            tOn_values (list, optional): List of values for 'tOn' if using 'independent' method.
            constant_values (list, optional): List of constant values if using 'height' or 'volume' method.
            save_output (bool, optional): If True, save the DataFrame to a file.
            outfile_name (str, optional): File name to save the DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing parameter combinations.
        """
        maxD_list = []
        tOff_list = []
        sdD_list = []
        constant_list = []
        tOn_list = []

        for self.maxD in maxD_values:
            for self.tOff in tOff_values:
                for self.sdD in sdD_values:
                    if method_value in ['height', 'volume']:
                        for self.constant in constant_values:
                            self.maxD = int(self.maxD)
                            self.tOff = int(self.tOff)

                            maxD_list.append(self.maxD)
                            tOff_list.append(self.tOff)
                            sdD_list.append(self.sdD)
                            constant_list.append(self.constant)

                    elif method_value == 'independent':
                        for self.tOn in tOn_values:
                            self.maxD = int(self.maxD)
                            self.tOff = int(self.tOff)

                            maxD_list.append(self.maxD)
                            tOff_list.append(self.tOff)
                            sdD_list.append(self.sdD)
                            tOn_list.append(self.tOn)

        maxD_array = np.array(maxD_list)
        tOff_array = np.array(tOff_list)
        sdD_array = np.array(sdD_list)
        if method_value in ['height', 'volume']:
            constant_array = np.array(constant_list)
            self.parameters_df = pd.DataFrame({
                'maxD_in': maxD_array,
                'tOff_in': tOff_array,
                'sdD_in': sdD_array,
                'constant_in': constant_array
            })
        elif method_value == 'independent':
            tOn_array = np.array(tOn_list)
            self.parameters_df = pd.DataFrame({
                'maxD_in': maxD_array,
                'tOff_in': tOff_array,
                'sdD_in': sdD_array,
                'tOn_in': tOn_array
            })

        if save_output:
            if outfile_name is not None:
                self.parameters_df.to_csv(outfile_name, index=False)
            else:
                raise ValueError('Please provide an outfile_name to save the dataframe')

        return self.parameters_df

    def dynamo_loop_properties(self):
        """
        Extract and organize simulation results from the properties array.
        """
        try:
            _ = self.properties_array
        except AttributeError:
            raise ValueError('No properties array found. Please run loop_dynamo() first.')

        self.percent_on_array = self.properties_array[:, 0]
        self.total_on_array = self.properties_array[:, 1]
        self.av_B_array = self.properties_array[:, 2]
        self.max_B_array = self.properties_array[:, 3]
        self.min_B_array = self.properties_array[:, 4]
        self.av_melt_array = self.properties_array[:, 5]
        self.tOff_av_array = self.properties_array[:, 6]
        self.av_vol_array = self.properties_array[:, 7]

    def dynamo_loop_properties_dataframe(self):
        """
        Create a DataFrame from the simulation results.
        """
        try:
            _ = self.properties_array
        except AttributeError:
            raise ValueError('No properties array found. Please run loop_dynamo() first.')

        self.properties_df = pd.DataFrame(
            self.properties_array,
            columns=['percent_on', 'total_on', 'av_B', 'max_B', 'min_B', 'av_melt', 'tOff', 'av_vol']
        )

    