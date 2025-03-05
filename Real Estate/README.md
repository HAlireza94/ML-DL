## House Prices Prediction for King County  
In here, I built a model to predict housing prices in King County, KC. Using XGBoost, I could get an accuray of almost 92%.
To get use of high-performance, it is better to run the script in HPCs as:  
python ./price.py > output.log 2>&1 &


# Quick Take Away from Histogram, Price Distribution
* We can see most houses are somehwere between 0 to $1 million
* Price distribution has a correlation with Squared Feet Living, which is not that much strong

## Skewness & Kurtosis
Skewness is a measure of the asymmetry of a distribution. A distribution is asymmetrical when its left and right side are not mirror images. we have a skewness of ~ 4, meaning it is right-hand wing, that means data needs to be normalized.

## Kurtosis
Kurtosis is a measure of whether the data are heavy-tailed or light-tailed relative to a normal distribution. The Kurtosis is around 34 which is highly leptokurtic, meaning we have many outliers or large values, perhaps it is just a luxury indicator :) Therefore, the key to a normal distribution could be using log!
