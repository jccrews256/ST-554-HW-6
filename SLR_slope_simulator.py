# Importing necessary modules
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng
from sklearn import linear_model

class SLR_slope_simulator:
    '''
    A class created to simulate the distribution of the slope estimate for a 
    simple linear regression with true intercept beta_0 (defaults to 0) and 
    true slope beta_1 (defaults to 1). 
    
    Key attributes beyond beta_0 and beta_1:
        - sigma: the true standard deviation of the random errors; defaults to 1
        - x: the 1D array of predictor values; defaults to three observations for
            each integer between 0 and 10 (inclusive)
        - n: the sample size, equal to the length of x
        - rng: the random number generator, set to default_rng(32), making results
            results reproducible
        - slopes: the list of simulated slopes for each model fit
      
    Methods:
        - generate_data: a method with no inputs that returns separate 1D arrays
            containing the pre-defined predictor values (x) and generated 
            response values (y) based on preset values for x, beta_0, beta_1,  
            and sigma
        - fit_slope: a method that takes in two 1D arrays, x and y, and returns
            the ordinary least squares slope estimate for the simple linear
            regression relating y (response) and x (predictor)
        - run_simulations: a method that takes in an integer, sims, that is the 
            number of simulated slope estimates that will be generated using 
            generate_data and fit_slope iteratively, thus creating a simulated
            sampling distribution of slopes captured in the slopes attribute; 
            note that the slopes attribute is reset before the new slopes are
            generated
        - plot_sampling_distribution: a method that returns a histogram of the
            simulated slope estimates, failing if run_simulations() has not
            been run yet.
        - find_prob: a method that returns an estimated one-sided or two-sided
            probability for the input value, with the probability type dependent
            on the the sided input; "above" for the probability above, "below"
            for the probability below, or "two-sided" for the two-sided probability
            - Note: the two-sided probability is calculated as double the proportion
                below when value is below the sample median and double the 
                proportion above when value is above the sample median
    '''
    
    # Initializing the class with initial values for each attribute
    def __init__(self, beta_0: float = 0, beta_1: float = 1, 
                    x: np.ndarray = None,
                    sigma: float = 1, seed: int = 32):
          self.beta_0 = beta_0
          self.beta_1 = beta_1
          self.sigma = sigma
          self.x = x if x is not None else np.array(list(np.linspace(start = 0, 
                        stop = 10, num = 11))*3)
          self.n = len(self.x) # Setting n to the length of x
          self.rng = default_rng(seed) # Initializing random number generator
          self.slopes = [] # Initially creating an empty list
    
    # Creating the generate_data method
    def generate_data(self):
        # Setting x to values specified upon initialization
        x = self.x
        
        # Generating y based on pre- or user-set true coefficients, x, and sigma
        y = self.beta_0 + self.beta_1*x + self.rng.normal(scale = self.sigma, 
            size = self.n)
            
        # Returning two arrays for x and y
        return x, y
      
    # Creating the fit_slope method
    def fit_slope(self, x: np.array = None, y: np.array = None):
        # Setting x to initial values if x not provided
        if x is None: 
            x = self.x
        
        # Generating y based on x and initial values of coefficients and sigma
        if y is None: 
            y = self.beta_0 + self.beta_1*x + self.rng.normal(scale = self.sigma, 
            size = len(x))
        
        # Creating the regression object
        reg = linear_model.LinearRegression()
        
        # Fitting model and extracting the least squares slope estimate
        fit = reg.fit(x.reshape(-1, 1), y)
        slope = fit.coef_[0]
        
        # Returning slope estimate
        return slope
      
    # Creating the run_simulations method
    def run_simulations(self, sims: int = 5000):
        # Clearly out any previous simulations
        self.slopes = []
      
        # Creating a for loop to generate and capture the slopes for each sim
        for _ in range(sims):
          # Generating response values given pre-set x values
          x, y = self.generate_data()
          
          # Capturing the slope coefficient for the SLR fit relating y and x
          self.slopes.append(self.fit_slope(x, y))
          
    # Creating the plot_sampling_distribution method
    def plot_sampling_distribution(self):
        # Confirming the slopes attribute is not empty
        if len(self.slopes) == 0:
            raise ValueError("The slopes attribute is empty. run_simulations() must be run before plot_sampling_distribution() is called.")

        # Rendering the histogram
        plt.hist(self.slopes)
        plt.xlabel("Slope Estimate")
        plt.ylabel("Frequency")
        plt.title("Simulated Sampling Distribution of Slope Estimate")
        plt.show()


    # Creating the find_prob method
    def find_prob(self, value: float, sided: str = "two-sided"):
        # Confirming the slopes attribute is not empty
        if len(self.slopes) == 0:
            raise ValueError("The slopes attribute is empty. run_simulations() must be run before find_prob is called.")

        # Converting slopes to an array for calculations
        slopes_array = np.array(self.slopes)

        # Returning the estimated probability based on sided
        if sided == "above": # probability above value
            prob = np.mean(slopes_array > value)
        elif sided == "below": # probability below value
            prob = np.mean(slopes_array < value)
        elif sided == "two-sided": #two-sided probability
            if value < np.median(slopes_array): # case where below median
                prob = 2*np.mean(slopes_array < value)
            else: # case where at or above median
                prob = 2*np.mean(slopes_array > value)
        else: # Raising error if anything else is passed to sided
            raise ValueError("sided must be one of 'above', 'below', or 'two-sided'")


        # Returning the probability
        return prob
            
# Running tests on class, methods, and attributes
def main():
    # Creating an instance of an SLR_slope_simulator object
    test_sim = SLR_slope_simulator(beta_0 = 12, beta_1 = 2, x = np.array(list(
        np.linspace(start = 0, stop = 10, num = 11))*3), sigma = 1, seed = 10)
    
    # Ensuring correct errors are thrown when run_simulations() hasn't been run
    # and moving to the next test
    try: 
        test_sim.plot_sampling_distribution()
    except ValueError as error:
        print(error)

    # Generating 10,000 slope estimates (nothing will print)
    test_sim.run_simulations(10000)

    # Generating histogram of slope estimates
    test_sim.plot_sampling_distribution()

    # Estimating probability of being more extreme than 2.1
    print(test_sim.find_prob(value = 2.1, sided = "two-sided"))

    # Extracting the slopes
    print(test_sim.slopes)

# Printing out the test results above if this file is run explicitly
if __name__ == "__main__":
    main()







