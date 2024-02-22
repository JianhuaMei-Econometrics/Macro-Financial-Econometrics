* a). Set the working directory to a sensible location
cd "D:\EMET8005\Week1"
* b). Load the data into Stata
use "CAschools.dta", clear

* c)Explore what variables are in the dataset, set more on and set more off
list 
describe
browse

* d)List observations 56-60 for specific variables
list dist_cod county stratio testscr in 56/60

* e)List observations for Kern county
list dist_cod county stratio testscr if county == "Kern"

* f) Create a histogram for average income
histogram avginc, bin(10) frequency
histogram avginc, bin(20) frequency

* g) Create a new variable for the logarithm of average income
gen ln_avginc = ln(avginc)

drop ln_avginc

* h) Create a histogram for the logarithm of average income
histogram ln_avginc, xtitle("Logarithm of Average Income") ytitle("Frequency") title("Logarithm of Average Income") color(blue) bin(10)

histogram ln_avginc, xtitle("Logarithm of Average Income") ytitle("Frequency") title("Logarithm of Average Income") color(blue) bin(20)

* i) Find out how many districts have more than 1000 computers
count if computer > 1000

* j) Check if stratio is equal to enrl.tot divided by teachers for all observations
gen check_stratio =  stratio - enrl_tot / teachers

assert(abs(check_stratio)<0.0001)

assert(abs(check_stratio)>0.0001)

* (k) Get summary statistics for avginc if avginc is lower than the median value
summarize avginc
summarize avginc if avginc < r(p50)

* (l) Get summary statistics for avginc in Kern county
summarize avginc if county == "Kern"

* m) creating a categorical variable for the student-teacher ratio:

gen cat = .
replace cat=1 if stratio<=17
replace cat=2 if 17<stratio & stratio<=20
replace cat=3 if 20<stratio

* check missing value
tabulate cat, missing

* n) finding the binomial probability:
display binomialp(19, 8, 0.53)
display 1 - binomial(19, 7, 0.53)

* o) c* matches any variable name that start with c 

summarize c*

* p) Get summary statistics for the variables in the dataset whose name ends with `pct'

sum *pct

* q) Make a copy of the variable stratio, ie create a new variable that is equal to stratio. Then replace the values of the variable with missings for observations where average income is less than 20.

gen newvar = stratio
replace newvar = . if avginc < 20
list newvar stratio if avginc < 20
summarize newvar stratio

* r) Drop all the new variable you created from the dataset.

drop newvar
 
*s) Summarize the variable testscr.

summarize testscr

* t) Use the mean command to compute the mean of testscr, the standard error of the mean, and a 95% confidence interval for the mean.

mean testscr


