
# Biterm Topic Model Stochastic Collapsed Variational Bayesian Hyper-parameter Inference 
Also included other algorithms BTM-CGS, SPTM-CGS .

## Requirement  
Pyton 3.6.1  
clang 3.9.1  
virtualenv	15.0.1

Others written in  requirment.txt , and set up later
## Usage  
the algorithms is  implemeted in C++ on libraries, which is included in `./pylibs/topicmodels/cpp_codes/cpp_libs/`  and select topic models algorithms.

### C++  
In `pylibs/topicmodels/cpp_codes/` , you can move this directory all, anywhere and call Makefile  
```
$ make
```

### Python  
you can use the algorithms for python, too.
To set up virtualenv, use Makefile contained in this directory.  
```
$ make
```   
  
set up enviroment variables  
```
$ source .envrc
```  
And `cd ./fit_models/models/ ` , you can select algorithmname.py
and then you run python program, cpp codes is called from cpp_libs directory
