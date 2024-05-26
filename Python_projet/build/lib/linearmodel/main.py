from .dataframe import load, Matrix, unique
import linearmodel.statistics as stat
import linearmodel.visualization as vis

# Import of the data
path = "D:\projet_python\eCO2mix_RTE_Annuel-Definitif_2020.csv"
data = load(path, sep=";")
data = Matrix(data)

# Text
print("\nPart 1.1: Data exploration \n")
print("Variables are: ")
for element in data[0]:
    print(element)
print("\nFirst 5 rows are:")
for element in data[0:5,:]:
    print(element)
print("\nWe observe that, every two rows, "+
        "only forecast values are present so we can delete these rows")

# Data transformation
columns = data[0]
data = Matrix(data[1::2,:])
print("\nNow, first 5 rows are:")
for element in data[0:5,:]:
    print(element)
    
# Text
print(f"\nThe time step is now {data[1][1]} hour and there are {data.shape[0]} rows")
print("To make data easier to use and quicker for calculation, "+
      "we can group the data by day and we also need to convert str into int values")

# Data transformation
data = data.transform_int([i for i in range(2,16)])
day_list = unique(data[1:,0])
data = data.group_by_day(day_list)

# Text
print(f"\nWe finally have {data.shape[0]} rows for the 366 days and {data.shape[1]} columns")
print(f"\nThe columns are {columns[2:]}")
print("\nand first 5 rows are:")
for element in data[0:5,:]:
    print(element)

# Descriptive analysis
print("\nPart 1.2: Data visualization")
print(f"\nMinimum of the variables are:")
print(stat.min_data(data))
print(f"\nMaximum of the variables are:")
print(stat.max_data(data))
print(f"\nThe mean of the variables are:")
print([round(x,2) for x in stat.mean_data(data)])
print(f"\nThe variance of the variables are:")
print([round(x,2) for x in stat.var_data(data)])
vis.graphics_part1_2(stat.cor_data(data),
                    [_ for _ in range(data.shape[0])],
                    [[_ for _ in range(data.shape[0])] for i in range(9)],
                    data[:,0],
                    [data[:,i] for i in range(3,12)],
                    label1=None, label2=columns[5:14])
print("\nWe observe on the correlation matrix that a lot of variables are"+
      " positively or negatively correlated like the forecast at j and j-1 with the consumption")
print("We are mostly interested by the correlation between the CO2 emission"+
      " and the other variables\n")
print([round(x,2) for x in stat.cor_data(data)[-1]])
print("\nWe notice that the variables that are the most correlated to C02 emission are "+
      "4 and 5 that are Coal and Gaz so it makes sense")
print("\nOn the energy production graphic, we observe the different production throught the year")
print("We can notice that the 'pompage' is negative because it is used to manage "+
      "the intermittency of the different energy sources")

# Ordinary Least squares
print("\nPart 1.3: Ordinary Least Squares")
print("\nfor this part, I will use numpy for the class OrdinaryLeastSquares because "+
      "the Matrix class is too slow and not optimized to inverse matrices")
print("Now we suppose the linear model between the CO2 emission variable and other ones")
print("It means that y = Xb + eps where y is the CO2 emission, X is the data, b a parameter "+
      "and eps is gaussian centered")
print("All the variables will be used even if they are probably not all usefull or redundant "+
      "because the information contained in the consumption is the same as the prevision "+
      "at j and j-1")
print("We can start by looking at the graph of CO2 emission during the year, then compute the model")
y = data[:,-1]
X = data[:,:13]
vis.graphic([_ for _ in range(366)], y, title="CO2 emission in France in 2020 (g/kWh)")
model = stat.OrdinaryLeastSquares(X,y)
model.fit()
print("\nThe coefficients of the fitted model are:")
print(model.get_coeffs())
print("\nThere are multiple ways to calculate the determination coefficient")
print("They are supposed to give the same result but due to the different "+
      "normalization I used to calculate the correlation they are not exactly the same")
print(f"We obtain {round(model.determination_coefficient(),2)} and "+
      f"{round(model.determination_coefficient2(),2)} that are both really close to 1 so our "+
      "model explains well the CO2 emission")
model.residual_histogram()
print("\nWe can see on the residual histogram that it looks like to be gaussian distributed")
print(f"the variance of the residuals is {model.residual_variance()}")
print("\nFinally we obtain the confidence interval 95%, by using scipy.stat to find the quantile "+
      "n-p of the Student's distribution\n")
print(Matrix(model.confidance_interval()))
