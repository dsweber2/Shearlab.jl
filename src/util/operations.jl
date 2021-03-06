# Submodule with some operations that will help to pad multidimensional arrays and Upsample in the Shearlab3D.m fashion

###################################################
# Padding array function to resize an array padding
# zeros
"""
...
padarray(array, newSize,gpu)
pads array with zeros to have a new size if the input array
is bigger than the targeted size, it will centered it at zero
and cut it off to fit
...
"""
function padarray(array::AbstractArray{T},newSize,gpu = false) where {T<:Number}
    # If you want to use gpu via ArrayFire array needs to be ArrayFire.AFArray{Float32,2}
    # Small patch if the array is one dimensional
    if size([size(array)...])[1]==1
        array = transpose(array)
    end
    padSizes = zeros(Integer,1,length(newSize))
    n = length(newSize)
    sizes = size(array)
    @fastmath @inbounds @simd for k = 1:n
        sizeDiff = newSize[k]-sizes[k];
        if mod(sizeDiff,2) == 0
            padSizes[k] = Int(sizeDiff/2);
        else
            padSizes[k] = Int(ceil.(sizeDiff/2))
        end
    end
    # Correct the matrix that is bigger than the targeted size
    idbig = [1:sizes[i] for i in 1:n]
    @fastmath @inbounds @simd for i in 1:length(newSize)
        if padSizes[i] < 0
            idbig[i] = (round(Int64,(size(array,i)-newSize[i])/2)+1):(round(Int64,(size(array,i)+newSize[i])/2))
            padSizes[i] = 0
        end
    end
    # We need to check if some of the padsizes are negative and cut the array from the center
    # Initialize the padded array with zeros
    if gpu
        paddedArray = AFArray(zeros(typeof(array[1]), newSize...))
    else
        paddedArray = zeros(typeof(array[1]),newSize...)
    end
    # lets create the indices array
    if gpu
        paddedArray[[idbig[1]+padSizes[1],idbig[2]+padSizes[2]]...] = array[idbig...]
    else
        view(paddedArray,[idbig[1].+padSizes[1],idbig[2].+padSizes[2]]...) .= view(array,idbig...)
    end
    return paddedArray
end #padarray

################################################################
# Function that flips from left to right an array in the second dimension
"""
...
fliplr(array,gpu) flips an array from left to right in the second dimension
...
"""
function fliplr(array::AbstractArray{T},gpu = false) where {T<:Number}
    return reverse(array,dims=2)
end #fliplr

##################################################################
# function that upsample an multidimensional array based on the same
# function at the matlab version
"""
..
upsample(array,dims,nZeros,gpu) upsample an array, in the dimensions dims
with nZeros
...
"""
function upsample(array::AbstractArray{T},dims::Integer,nZeros::Integer, gpu = false) where {T<:Number}
    sz = [size(array)...]
    szUpsampled = sz
    szUpsampled[dims] = (szUpsampled[dims]-1).*(nZeros+1)+1
    if gpu
        arrayUpsampled = AFArray(zeros(typeof(array[1]),szUpsampled...))
    else
        arrayUpsampled = zeros(typeof(array[1]),szUpsampled...);
    end
    # Generate the indices per dimension
    ids = [1:1:size(array,i) for i in 1:length(size(array))]
    @fastmath @inbounds @simd  for k in 1:length(dims)
        ids[dims[k]] = 1:(nZeros[k]+1):szUpsampled[dims[k]]
    end
    arrayUpsampled[ids...] = array
    return arrayUpsampled
end #upsample

#####################################################################
# Function that rounds a number to the nearest integer towards zero
"""
...
fix(x) rounds a number x to the nearest integer towards zero
...
"""
function fix(x::T) where {T<:Number}
    if x < 0
        fixed = ceil(x)
    else
        fixed = floor(x)
    end
    return convert(Int64,fixed)
end #fix

#######################################################################
# Function that shears an array in order k in the direction of axis
# based on the same function on the Matlab version
"""
...
dshear(array,k,axis,gpu) shears and array in order k in the direction of
axis
...
"""
function dshear(array::AbstractArray{T},k::Integer,axis::Integer, gpu = false) where {T<:Number}
    if gpu
        array = Array(array)
    end
    if k == 0
        sheared = array;
    else
        rows = size(array,1);
        cols = size(array,2);
        sheared = zeros(typeof(array[1]),size(array)...)
        if axis == 1
            for col = 1:cols
                sheared[:,col] = reshape(circshift(reshape(array[:,col],rows,1),[-k*(floor(cols/2)+1-col) 0]),(rows,));
            end
        else
            for row = 1:rows
                sheared[row,:] = reshape(circshift(reshape(array[row,:],1,cols),[0 -k*(floor(rows/2)+1-row)]),(cols,));
            end
        end
    end
    if gpu
        sheared = AFArray(sheared)
    end
    return sheared
end #dshear

######################################################################
## Type of filter configurations
struct Filterconfigs
    directionalFilter
    scalingFilter
    waveletFilter
    scalingFilter2
end #Filterconfigs

#######################################################################
# Function that check the sizes of the filters to know if it is possible
"""
...
checkfiltersizes(rows,cols,shearLevels,directionalFilter,scalingFilter,waveletFilter,scalingFilter2) function that check
wBand
the size of the filters and set new possible configurations
...
"""
function checkfiltersizes(rows,cols,shearLevels,directionalFilter,scalingFilter,waveletFilter,scalingFilter2)
    # println("input is rows=$(rows)")
    # println("cols=$(cols)")
    # println("shearLevels=$(shearLevels)")
    # println("directionalFilter=$(directionalFilter)")
    # println("scalingFilter=$(scalingFilter)")
    # println("waveletFilter=$(waveletFilter)")
    # println("scalingFilter2 = $(scalingFilter2)")
    # Lets initialize the FilterConfig array,
    filterSetup = []
    # Set all configurations

    # Configuration 1
    push!(filterSetup,Filterconfigs(directionalFilter,scalingFilter,
                                    waveletFilter,scalingFilter2));

    # Configuration 2
    # Check the default configuration
    scalingFilter = filt_gen("scaling_shearlet");
    directionalFilter = filt_gen("directional_shearlet");
    waveletFilter = mirror(scalingFilter);
    scalingFilter2 =  scalingFilter;
    push!(filterSetup,Filterconfigs(directionalFilter,scalingFilter,
                                    waveletFilter,scalingFilter2));

    # Configuration 3
    # Just change the directionalFilter
    directionalFilter = filt_gen("directional_shearlet2");
    push!(filterSetup,Filterconfigs(directionalFilter,scalingFilter,
                                    waveletFilter,scalingFilter2));

    # Configuration 4
    # The same as 3
    push!(filterSetup,Filterconfigs(directionalFilter,scalingFilter,
                                    waveletFilter,scalingFilter2));

    # Configuration 5
    # It changes the scalingFilter, the waveletFilter and the scalingFilter2
    scalingFilter = filt_gen("Coiflet1");
    waveletFilter = mirror(scalingFilter);
    scalingFilter2 =  scalingFilter;
    push!(filterSetup,Filterconfigs(directionalFilter,scalingFilter,
                                    waveletFilter,scalingFilter2));

    # Configuration 6
    # It changes the scalingFilter, the waveletFilter and the scalingFilter2
    scalingFilter = filt_gen(WT.db2)[2:5];
    waveletFilter = mirror(scalingFilter);
    scalingFilter2 =  scalingFilter;
    push!(filterSetup,Filterconfigs(directionalFilter,scalingFilter,
                                    waveletFilter,scalingFilter2));

    # Configuration 7
    # Just change the directionalFilter
    directionalFilter = filt_gen("directional_shearlet3");
    push!(filterSetup,Filterconfigs(directionalFilter,scalingFilter,
                                    waveletFilter,scalingFilter2));

    # Configuration 8
    # It changes the scalingFilter, the waveletFilter and the scalingFilter2
    scalingFilter = filt_gen(WT.haar)[2:3]
    waveletFilter = mirror(scalingFilter);
    scalingFilter2 =  scalingFilter;
    push!(filterSetup,Filterconfigs(directionalFilter,scalingFilter,
                                    waveletFilter,scalingFilter2));

    # Check the sizes of the filters in comparison with the rows and cols
    kk = 0;
    success1 = 0;
    # println("lets find out why this breaks")
    for k = 1:8
        # println("k=$(k)")
        ## check 1
        lwfilter = length(filterSetup[k].waveletFilter);
        lsfilter = length(filterSetup[k].scalingFilter);
        lcheck1 = lwfilter;
        for j = 1:(length(shearLevels)-1)
            lcheck1 = lsfilter + 2*lcheck1 - 2;
        end
        # println("lcheck1 = $(lcheck1)")
        if lcheck1 > cols || lcheck1 > rows
            continue;
        end

        ## check  2
        rowsdirfilter = size(filterSetup[k].directionalFilter,1);
        colsdirfilter = size(filterSetup[k].directionalFilter,2);
        lcheck2 = (rowsdirfilter-1)*2^(maximum(shearLevels)+1) + 1;

        lsfilter2 = length(filterSetup[k].scalingFilter2);
        lcheck2help = lsfilter2;
        for j = 1:maximum(shearLevels)
            lcheck2help = lsfilter2 + 2*lcheck2help - 2;
        end
        lcheck2 = lcheck2help + lcheck2 - 1;
        # println("lcheck2 = $(lcheck2)")
        if lcheck2 > cols || lcheck2 > rows || colsdirfilter > cols || colsdirfilter > rows
            continue;
        end
        success1 = 1;
	kk = k;
        break;
    end
    if success1 == 0
        @error "The specified Shearlet system was not available for data of size "* string(rows) *"x",string(cols)* ". Filters were not set (see operations.jl)."
    end
    if success1 == 1 && kk > 1
        @warn "The specified Shearlet system was not available for data of size "*string(rows)*"x"*string(cols)*". Filters were automatically set to configuration "*string(kk)*" (see operations.jl)."
    end
    filterSetup[kk]
end #checkfiltersizes


function describeConfig(kk::Int)
    if kk==1
        # the requested settings are available
        return "you shouldn't see this"
    elseif kk==2
        # Configuration 2
        return "the scaling shearlet as both scaling function and  and and directional shearlet"
    elseif kk==3
        # Configuration 3
        return "the scaling shearlet and the 2nd directional shearlet"
    elseif kk==4
        # Configuration 4
        return "the scaling shearlet and the 2nd directional shearlet"
    elseif kk==5
        # Configuration 5
        return "the Coiflet1 scaling function, the mirrored scaling shearlet as the wavelet, and the 2nd directional shearlet"
    elseif kk==6
        # Configuration 6
        return "the Daubechies2 scaling function, the mirrored version as the wavelet, and the 2nd directional shearlet"
    elseif kk==7
        # Configuration 7
        return "the Daubechies2 scaling function, the mirrored version as the wavelet, and the 3nd directional shearlet"
    elseif kk==8
        # Configuration 8
        return "the Haar scaling function, the mirrored version as the wavelet, and the 3nd directional shearlet"
    end
end


"""
    findSupport(shears::Array{<:Number, 3}; tolerance::Float64=1e-10)

The default tolerance keeps ~99% of the coefficients (and a significantly high fraction of the mass)
TODO: implement this in an efficient way (i.e. use the fact that these originate from sheared versions of the same functions)
"""
function findSupport(shears::Array{<:Number, 3}; tolerance::Float64=1e-10)
  nShears = size(shears,3)
  effSupport = Array{Tuple{Tuple{Int,Int}, Tuple{Int,Int}}}(undef, nShears)
  spaceShears = real.(fftshift(ifft(ifftshift(shears), (1, 2))))
  actuallySupported = abs.(spaceShears).>=tolerance
  doesThisRowHaveTolEls= maximum(actuallySupported,dims=1)
  doesThisColHaveTolEls = maximum(actuallySupported,dims=2)
  size(doesThisRowHaveTolEls)
  size(doesThisColHaveTolEls)
  for i=1:nShears
     effSupport[i] = ((findfirst(doesThisColHaveTolEls[:,1,i]), findlast(doesThisColHaveTolEls[:,1,i])), (findfirst(doesThisRowHaveTolEls[1,:,i]), findlast(doesThisRowHaveTolEls[1,:,i])))
  end
  return effSupport
end


"""
    padBy = 

getPadBy(supports; tolerance::Float64=1e-10)

Find the amount to pad by to guarantee that no shearlet's support wraps around
the boundary. Support here being the space window that captures `1-tolerance`
of the mass

"""
function getPadBy(shears; tolerance::Float64=1e-10)
    supports = findSupport(shears; tolerance = tolerance)
    padBy = (0, 0)

    for sysPad in supports
        padBy = (max(sysPad[1][2]-sysPad[1][1], padBy[1]), max(padBy[2],
                                                               sysPad[2][2]-sysPad[2][1]))
    end
    return padBy
end


"""
    pad(x, padBy)
create a padded version of X so that it has +/-padBy[1] extra zeros in the
first dimension and +/-padBy[2] in the second dimension. All other dimensions
are untouched.
"""
function pad(x, padBy)
    T = eltype(x)
    szx = size(x)
    corner = zeros(T, padBy...,  szx[3:end]...)
    firstRow = cat(corner,
                   zeros(T, padBy[1], szx[2:end]...),
                   corner, dims = 2)
    secondRow = cat(zeros(T, szx[1] , padBy[2], szx[3:end]...),
                    x,
                    zeros(T, szx[1] , padBy[2], szx[3:end]...),
                    dims=2)
    thirdRow = cat(corner,
                   zeros(T, padBy[1], szx[2:end]...),
                   corner,
                   dims = 2)
    return cat(firstRow, secondRow, thirdRow, dims = 1)
end
