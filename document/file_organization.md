# File organization of George
This gives an overview of how the code is organized.

<a id="Organizational_chart"></a>

|Folder | Nested files |
--------|-----------|
|[george](#george) |           |
|       |[gp.py](#gp.py)|
|       |[kernels.py](#kernels.py)|
|       |[kernels.pxd](#kernels.pxd)|
|       |[_kernels.pxd](#_kernels.pyx)|
|       |[basic.py](#basic.py) |
|       |[hodlr.pyx](#hodlr.pyx)
|       |[utils.py](#utils.py)
|-------|------------|
|[include](#include)| |
||[kernel.h](#kernels.h)|
||[george.h](#george.h)|
||[metric.h](#metric.h)|
||[solver.h](#solver.h)|
||[constants.h](#constants.h)|
|-------|------------|
|[hodlr](#hodlr)||
|-------|------------|




# george
<a id="george"></a>
Folder that holds the python wrapper modules.   
[Back to organization chart](#Organizational_chart)

## `basic.py`
<a id="basic.py"></a>
Code for basic matrix solver.   
[Back to organization chart](#Organizational_chart)

## `kernels.py`
<a id="kernels.py"></a>
Code for laying out the `Python` definitions of kernels.
* sets parameters that specifies the kernels for the `Cython` definitions.  

[Back to organization chart](#Organizational_chart)

## `kernels.pxd`
<a id="kernels.pxd"></a>
`Cython` wrapper for gluing the Python and C++ header code.  
contains code for passing the kernel parameters between `Python` and
`C++` code.  
[Back to organization chart](#Organizational_chart)

## `_kernels.pyx`
<a id="_kernels.pyx"></a>
Contains for loops for computing values for kernel covariance matrices 
element by element.   

[Back to organization chart](#Organizational_chart)

## `gp.py`
<a id="gp.py"></a>
Wrapper for all the `gp` methods, such as:   

* compute - for computing the constants for the lnlikelihood
* lnlikelihood
* predict  
etc.   

[Back to organization chart](#Organizational_chart)

## `hodlr.pyx`
<a id="hodlr.pyx"></a>
`Cython` wrapper for the `hodlr` matrix solver.  
[Back to organization chart](#Organizational_chart)

## `utils.py`
<a id="utils.py"></a>
Contains `Python` wrapper code to `SciPy` / `NumPy` libraries for 
* drawing samples of multivariate gaussians,
* building KDTree for sorting data.
* computing numerical gradient for optimizing lnlikelihood?

[Back to organization chart](#Organizational_chart)

# include
<a id="include"></a>   
`C++` header files that contain most of the `C++` implementation.   
[Back to organization chart](#Organizational_chart)

## `kernels.h`
<a id="kernels.h"></a>   
Contains the `C++` implementation of the different kernels.   
[Back to organization chart](#Organizational_chart)

## `george.h`
<a id ="george.h"></a>
Contains `include` statements for several other header files.   
[Back to organization chart](#Organizational_chart)

## `metric.h`
<a id="metric.h"></a>
Contains code for setting the `l_sq` or other kernel parameters that is 
multiplied to the distances.  

[Back to organization chart](#Organizational_chart)

## `solver.h`
<a id="solver.h"></a>
Contains the actual `C++` implementation of the `hodlr` solver that uses 
`hodlr` functions to solve matrix inversion etc.   

[Back to organization chart](#Organizational_chart)

## `constants.h`
<a id="constants.h"></a>
Sets integer code for reporting `C++` errors.  
[Back to organization chart](#Organizational_chart)


# hodlr 
<a id="hodlr"></a>
Folder with `C++` code that holds the sparse matrix solver.   
[Back to organization chart](#Organizational_chart)
