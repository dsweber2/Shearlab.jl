# Submodule that constructs the system of Shearlets in 2D of size rows x cols


######################################################################
## ype of Filterconfigsuration to check the sizes
struct Filterswedgebandlow
  wedge::AbstractArray
  bandpass::AbstractArray
  lowpass::AbstractArray
end

#######################################################################
# Function that generates the whole Shearlet System filters (wedge, bandpass and lowpass) of size rows x cols
"""
  ...
  getwedgebandpasslowpassfilters2D(rows,cols,directionalFilter = filt_gen("directional_shearlet"), scalingFilter = filt_gen("scaling_shearlet"),gpu = false)
  generates the Wedge, Bandpass and LowspassFilter of size rows x cols
  ...
  """

function getwedgebandpasslowpassfilters2D(rows::Int, cols::Int, shearLevels,
                                          directionalFilter =
                                          filt_gen("directional_shearlet"),
                                          scalingFilter =
                                          filt_gen("scaling_shearlet"),
                                          waveletFilter =
                                          mirror(scalingFilter),
                                          scalingFilter2 = scalingFilter, gpu =
                                          false)

  FFTW.set_num_threads(Sys.CPU_THREADS)
  # Make shearLevels integer
  shearLevels = map(Int,shearLevels);
  # The number of scales
  NScales = length(shearLevels);

  # Initialize the bandpass and wedge filter
  # Bandpass filters partion the frequency plane into different scales
  if gpu
    bandpass = zeros(Float32,rows,cols,NScales);
    bandpass = AFArray(bandpass+im*zeros(bandpass));
  else
    bandpass = zeros(Float64,rows,cols,NScales);
    bandpass = bandpass+im*zeros(size(bandpass));
  end

  # Wedge filters partition the frequency plane into different directions
  wedge = Array{Any}(undef, 1, round(Int64,maximum((shearLevels) .+ 1)));

  # normalize the directional filter directional filter
  if gpu
    directionalFilter = AFArray(convert(Array{Float32},directionalFilter/sum(abs.(directionalFilter[:]))));
  else
    directionalFilter = directionalFilter/sum(abs.(directionalFilter[:]));
  end

  # Compute 1D high and lowpass filters at different scales

  # filterHigh{NScales} = g, and filterHigh{1} = g_J (analogous with filterLow)
  filterHigh = Array{Any}(undef, 1,NScales);
  filterLow = Array{Any}(undef, 1,NScales);
  filterLow2 = Array{Any}(undef, 1,round(Int64,maximum(shearLevels)+1));

  # Initialize wavelet highpass and lowpass filters
  if gpu
    filterHigh[size(filterHigh,2)] = AFArray(convert(Array{Float32},waveletFilter));
    filterLow[size(filterLow,2)] = AFArray(convert(Array{Float32},scalingFilter));
    filterLow2[size(filterLow2,2)] = AFArray(convert(Array{Float32},scalingFilter2));

    # Lets compute the filters in the other scales 
    for j = (size(filterHigh)[2]-1):-1:1
      filterLow[j] = conv2(filterLow[size(filterLow,2)],upsample(filterLow[j+1],1,1,1))
      filterHigh[j] = conv(filterLow[size(filterLow,2)],upsample(filterHigh[j+1],1,1,1))
    end

    # Lets compute the filters in the other scales
    for j=(size(filterLow2)[2]-1):-1:1
      filterLow2[j] = conv(filterLow2[size(filterLow2,2)],upsample(filterLow2[j+1],1,1,1))
    end
    # Construct the bandpassfilter
    # Need to convert first to complex array since
    for j = 1:size(filterHigh,2)
      bandpass[:,:,j] = -fftshift(fft(ifftshift(padarray(filterHigh[j],[rows,cols],1))));
    end
  else
    filterHigh[size(filterHigh,2)] = waveletFilter;
    filterLow[size(filterLow,2)] = scalingFilter;
    filterLow2[size(filterLow2,2)] = scalingFilter2;

    # Lets compute the filters in the other scales
    for j = (size(filterHigh)[2]-1):-1:1
      filterLow[j] = conv(filterLow[size(filterLow,2)],upsample(filterLow[j+1],1,1))
      filterHigh[j] = conv(filterLow[size(filterLow,2)],upsample(filterHigh[j+1],1,1))
    end

    # Lets compute the filters in the other scales
    for j=(size(filterLow2)[2]-1):-1:1
      filterLow2[j] = conv(filterLow2[size(filterLow2,2)],upsample(filterLow2[j+1],1,1))
    end

    # Construct the bandpassfilter
    # Need to convert first to complex array since
    for j = 1:size(filterHigh,2)
      bandpass[:,:,j] = fftshift(fft(ifftshift(padarray(filterHigh[j],[rows,cols]))));
    end
  end



  ## construct wedge filters for achieving directional selectivity.
  # as the entries in the shearLevels array describe the number of differently
  # sheared atoms on a certain scale, a different set of wedge
  # filters has to be constructed for each value in shearLevels.
  if gpu
    for shearLevel = unique(shearLevels)
      #preallocate a total of floor(2^(shearLevel+1)+1) wedge filters, where
      #floor(2^(shearLevel+1)+1) is the number of different directions of
      #shearlet atoms associated with the horizontal (resp. vertical)
      #frequency cones.
      wedge[shearLevel+1] = AFArray(zeros(Complex{Float32},rows,cols,floor(2^(shearLevel+1)+1)));
      #upsample directional filter in y-direction. by upsampling the directional
      #filter in the time domain, we construct repeating wedges in the
      #frequency domain ( compare abs.(fftshift(fft2(ifftshift(directionalFilterUpsampled)))) and
      #abs.(fftshift(fft2(ifftshift(directionalFilter)))) ).
      directionalFilterUpsampled = upsample(directionalFilter,1,2^(shearLevel+1)-1,1);

      #remove high frequencies along the y-direction in the frequency domain.
      #by convolving the upsampled directional filter with a lowpass filter in y-direction, we remove all
      #but the central wedge in the frequency domain.
      wedgeHelp = conv2(directionalFilterUpsampled, filterLow2[size(filterLow2,2)-shearLevel][:,:])
      wedgeHelp = padarray(wedgeHelp,[rows,cols],1);
      #please note that wedgeHelp now corresponds to
      #conv(p_j,h_(J-j*alpha_j/2)') in the language of the paper. to see
      #this, consider the definition of p_j on page 14, the definition of w_j
      #on the same page an the definition of the digital sheralet filter on
      #page 15. furthermore, the g_j part of the 2D wavelet filter w_j is
      #invariant to shearings, hence it suffices to apply the digital shear
      #operator to wedgeHelp.;

      ## application of the digital shear operator (compare equation (22))

      #upsample wedge filter in x-direction. this operation corresponds to
      #the upsampling in equation (21) on page 15.
      wedgeUpsampled = upsample(wedgeHelp,2,2^shearLevel-1,1);
      #convolve wedge filter with lowpass filter, again following equation
      #(21) on page 14.
      lowpassHelp = padarray(filterLow2[size(filterLow2,2)-max(shearLevel-1,0)],size(wedgeUpsampled),1);
      if shearLevel >= 1
        wedgeUpsampled = fftshift(ifft(ifftshift(fftshift(fft(ifftshift(lowpassHelp))).*fftshift(fft(ifftshift(wedgeUpsampled))))));
      end
      lowpassHelpFlip = fliplr(lowpassHelp,1);
      #traverse all directions of the upper part of the left horizontal
      #frequency cone
      for k = -2^shearLevel:2^shearLevel
        #resample wedgeUpsampled as given in equation (22) on page 15.
        wedgeUpsampledSheared = dshear(wedgeUpsampled,k,2,1);
        #convolve again with flipped lowpass filter, as required by equation (22) on
        #page 15.
        if shearLevel >= 1
          wedgeUpsampledSheared = fftshift(ifft(ifftshift(fftshift(fft(ifftshift(lowpassHelpFlip))).*fftshift(fft(ifftshift(wedgeUpsampledSheared))))));
        end
        #obtain downsampled and renormalized and sheared wedge filter in the
        #frequency domain, according to equation (22) on page 15
        wedge[shearLevel+1][:,:,fix(2^shearLevel)+1-k] = fftshift(fft(ifftshift(2^shearLevel*wedgeUpsampledSheared[:,1:2^shearLevel:(2^shearLevel*cols-1)])));
      end
    end
  else
    for shearLevel = unique(shearLevels)
      #preallocate a total of floor(2^(shearLevel+1)+1) wedge filters, where
      #floor(2^(shearLevel+1)+1) is the number of different directions of
      #shearlet atoms associated with the horizontal (resp. vertical)
      #frequency cones.
      wedge[shearLevel+1] = zeros(rows,cols,floor(2^(shearLevel+1)+1))+zeros(rows,cols,floor(2^(shearLevel+1)+1))*im;

      #upsample directional filter in y-direction. by upsampling the directional
      #filter in the time domain, we construct repeating wedges in the
      #frequency domain ( compare abs.(fftshift(fft2(ifftshift(directionalFilterUpsampled)))) and
      #abs.(fftshift(fft2(ifftshift(directionalFilter)))) ).
      directionalFilterUpsampled = upsample(directionalFilter,1,2^(shearLevel+1)-1);

      #remove high frequencies along the y-direction in the frequency domain.
      #by convolving the upsampled directional filter with a lowpass filter in y-direction, we remove all
      #but the central wedge in the frequency domain.

      wedgeHelp = conv(directionalFilterUpsampled, filterLow2[size(filterLow2,2)-shearLevel][:,:])
      wedgeHelp = padarray(wedgeHelp,[rows,cols]);

      #please note that wedgeHelp now corresponds to
      #conv(p_j,h_(J-j*alpha_j/2)') in the language of the paper. to see
      #this, consider the definition of p_j on page 14, the definition of w_j
      #on the same page an the definition of the digital sheralet filter on
      #page 15. furthermore, the g_j part of the 2D wavelet filter w_j is
      #invariant to shearings, hence it suffices to apply the digital shear
      #operator to wedgeHelp.;

      ## application of the digital shear operator (compare equation (22))

      #upsample wedge filter in x-direction. this operation corresponds to
      #the upsampling in equation (21) on page 15.
      wedgeUpsampled = upsample(wedgeHelp,2,2^shearLevel-1);

      #convolve wedge filter with lowpass filter, again following equation
      #(21) on page 14.
      lowpassHelp = padarray(filterLow2[size(filterLow2,2)-max(shearLevel-1,0)],size(wedgeUpsampled));
      if shearLevel >= 1
        wedgeUpsampled = fftshift(ifft(ifftshift(fftshift(fft(ifftshift(lowpassHelp))).*fftshift(fft(ifftshift(wedgeUpsampled))))));
      end
      lowpassHelpFlip = fliplr(lowpassHelp);
      #traverse all directions of the upper part of the left horizontal
      #frequency cone
      for k = -2^shearLevel:2^shearLevel
        #resample wedgeUpsampled as given in equation (22) on page 15.
        wedgeUpsampledSheared = dshear(wedgeUpsampled,k,2);
        #convolve again with flipped lowpass filter, as required by equation (22) on
        #page 15.
        if shearLevel >= 1
          wedgeUpsampledSheared = fftshift(ifft(ifftshift(fftshift(fft(ifftshift(lowpassHelpFlip))).*fftshift(fft(ifftshift(wedgeUpsampledSheared))))));
        end
        #obtain downsampled and renormalized and sheared wedge filter in the
        #frequency domain, according to equation (22) on page 15
        wedge[shearLevel+1][:,:,fix(2^shearLevel)+1-k] = fftshift(fft(ifftshift(2^shearLevel*wedgeUpsampledSheared[:,1:2^shearLevel:(2^shearLevel*cols-1)])));
      end
    end
end

## compute low pass filter of shearlet system
if gpu
  lowpass = fftshift(fft(ifftshift(padarray(transpose(filterLow[1]')*filterLow[1]',[rows,cols],1))));
else
  lowpass = fftshift(fft(ifftshift(padarray(transpose(filterLow[1]')*filterLow[1]',[rows,cols]))));
end

# Generate the final array
return Filterswedgebandlow(wedge,bandpass,lowpass)
end #getwedgebandpasslowpassfilters2D

##############################################################
# Create a type for the Preparedfilters
struct Preparedfilters
  size::Tuple{Int,Int}
  shearLevels::Array{Int,1}
  cone1
  cone2
end

#######################################################################
# Function that prepare the filters
"""
  ...
  SlprepareFilters2D(rows, cols, nScales, shearLevels = ceil.((1:nScales)/2),
      directionalFilter = filt_gen("directional_shearlet"),
      scalingFilter = filt_gen("scaling_shearlet"),
      waveletFilter = mirror(scalingFilter),
      scalingFilter2 = scalingFilter, gpu = false) function that prepare the filters to generate
		   the shearlet system
  ...
  """
function preparefilters2D(rows, cols, nScales, shearLevels = ceil.((1:nScales)/2),
                          directionalFilter = filt_gen("directional_shearlet"),
                          scalingFilter = filt_gen("scaling_shearlet"),
                          waveletFilter = mirror(scalingFilter),
                          scalingFilter2 = scalingFilter, gpu = false)

  #Make sure the shearLevles are integer
  shearLevels = map(Int,shearLevels)
  # check filter sizes
  filters = checkfiltersizes(rows,cols,shearLevels,directionalFilter,scalingFilter,waveletFilter,scalingFilter2);

  # Save the new filters
  directionalFilter = filters.directionalFilter;
  scalingFilter = filters.scalingFilter;
  waveletFilter = filters.waveletFilter;
  scalingFilter2 = filters.scalingFilter2;

  # Define the cones
  cone1 = getwedgebandpasslowpassfilters2D(rows,cols,shearLevels,directionalFilter,scalingFilter,waveletFilter,scalingFilter2,gpu);
  if rows == cols
    cone2 = cone1;
  else
    cone2 = getwedgebandpasslowpassfilters2D(cols,rows,shearLevels,directionalFilter,scalingFilter,waveletFilter,scalingFilter2,gpu);
  end
  return Preparedfilters((rows,cols),shearLevels,cone1,cone2)
end # preparefilters2D

#######################################################################
# Function compute a index set describing a 2D shearlet system
"""
  ...
  getshearletidxs2D(shearLevels, full=0, restriction_type = [],restriction_value = []) Compute a index set describing a 2D shearletsystem, with restriction_type in
  ["cones","scales","shearings"], where shearLevels is a 1D array spcifying the level of shearing on each scale, and full is a boolean that orders to generate the full shearlet indices.
  ...
  """
function getshearletidxs2D(shearLevels, full=0, restriction_type = [],restriction_value = [])
	shearLevels = map(Int,shearLevels)
  # Set default values
  shearletIdxs = [];
  includeLowpass = 1;
  scales = 1:length(shearLevels);
  shearings = -2^(maximum(shearLevels)):2^(maximum(shearLevels));
  cones = 1:2;

  for i = 1:length(restriction_type)
    includeLowpass = 0;
    if restriction_type[i] == "scales"
      scales = restriction_value[i];
    elseif restriction_type[i] == "shearings"
      shearings = restriction_value[i];
    elseif restriction_type[i] == "cones"
      cones = restriction_value[i];
    end
  end

  for cone = intersect(1:2,cones)
    for scale = intersect(1:length(shearLevels),scales)
      for shearing = intersect(-2^shearLevels[scale]:2^shearLevels[scale],shearings)
        if convert(Bool,full) || cone == 1 || abs.(shearing)<2^(shearLevels[scale])
          shearletIdxs=[shearletIdxs;[cone,scale,shearing]];
        end
      end
    end
  end

  shearletIdxs = transpose(reshape(shearletIdxs,3,Int(length(shearletIdxs)/3)))
  if convert(Bool,includeLowpass) || convert(Bool,sum(0 .== scales)) || convert(Bool,sum(0 .== cones))
    shearletIdxs = [shearletIdxs;[0 0 0]];
  end
  return Int.(shearletIdxs)
end # getshearletidxs2D

####################################################################
# Type of shearletsystem in 2D
struct Shearletsystem2D{T<:Number, CT<:Union{Complex{T}, T}} # CT tells us
    # whether we should be using a complex or real fft
    shearlets::Array{Complex{T}, 3}
    size::Tuple{Int, Int}
    shearLevels::Array{Int, 1}
    full::Bool
    nShearlets::Int
    shearletIdxs::Array{Int, 2}
    dualFrameWeights::Array{T, 2}
    RMS::Array{T, 2}
    gpu::Bool
    padded::Bool
    padBy::Tuple{Int, Int}
end

function Base.show(io::IO, l::Shearletsystem2D{T,CT}) where {T,CT}
    print(io, "Shearletsystem2D{$(T),$(CT)}(input = $(l.size), levels = $(l.shearLevels),"*
          " nshearlets = $(l.nShearlets),gpu=$(l.gpu)"*(l.padded ?
                                                        ", padBy = $(l.padBy))" : ""))
end



#######################################################################
# helper methods to reduce code redundandancy

"""
    shearlets, dualFrameWeights = padShearlets(shearlets, dualFrameWeights, typeBecomes)

Add padding in the space domain and convert back. Also, if the type being
transformed is real, only store half the coefficients, as they're otherwise
redundant.
"""
function padShearlets(shearlets, dualFrameWeights, typeBecomes, padBy, upperFrameBound)
    shearlets = real.(fftshift(ifft(ifftshift(shearlets, (1,2)), (1, 2)),
                               (1, 2)))
    shearlets = cat([pad(shearlets[:,:,j], padBy) for
                     j=1:size(shearlets,3)]...; dims = 3)
    if typeBecomes <: Real
        P = plan_rfft(zeros(typeBecomes, size(shearlets)[1:2]), (1,2))
        newSize = (div(size(shearlets,1), 2)+1, size(shearlets)[2:3]...)
    else
        P = plan_fft(zeros(typeBecomes, size(shearlets)[1:2]), (1,2))
        newSize = size(shearlets)
    end
    newShears = zeros(Complex{eltype(shearlets)}, newSize)
    totalMass = sum(abs.(shearlets[:,:,1]))
    # rowMean = sum((1:size(shearlets,1)) .* abs.(shearlets[:,:,1]), dims=(1,2))/totalMass
    # colMean = sum((1:size(shearlets,2))' .* abs.(shearlets[:,:,1]), dims=(1,2))/totalMass
    # println("newish location is $((rowMean, colMean))) out of $(size(shearlets)[1:2])")

    for j=1:size(shearlets, 3)
        newShears[:, :, j] = P * shearlets[:, :, j]
    end
    dualFrameWeights = sum(real.(abs.(newShears).^2), dims=3)[:,:]
    if upperFrameBound > 0
        totalMass = norm(dualFrameWeights, Inf)
        normalize = typeBecomes(upperFrameBound)/totalMass
        newShears = newShears .* normalize
        dualFrameWeights = sum(real.(abs.(newShears).^2), dims=3)[:,:]
    end
    return (newShears, dualFrameWeights)
end

"""
    shearlets, dualFrameWeights, RMS, rows, cols, nShearlets =
        generateShearlets(shearletIdxs, Preparedfilters,typeBecomes)

the core code for generating shearlets
"""
function generateShearlets(shearletIdxs, Preparedfilters, typeBecomes, gpu)
    rows = Preparedfilters.size[1];
    cols = Preparedfilters.size[2];
    nShearlets = size(shearletIdxs,1);
    if gpu
        shearlets = AFArray(zeros(Complex{Float32},rows,cols,nShearlets));
    else
        shearlets = zeros(typeBecomes, rows, cols, nShearlets)+im .*
            zeros(rows, cols, nShearlets);
    end
    # Compute shearlets
    for j = 1:nShearlets
        cone = shearletIdxs[j,1];
        scale = shearletIdxs[j,2];
        shearing = shearletIdxs[j,3];

        if cone == 0
            shearlets[:,:,j] = Preparedfilters.cone1.lowpass;
        elseif cone == 1
            #here, the fft of the digital shearlet filters described in
            #equation (23) on page 15 of "ShearLab 3D: Faithful Digital
            #Shearlet Transforms based on Compactly Supported Shearlets" is computed.
            #for more details on the construction of the wedge and bandpass
            #filters, please refer to getwedgebandpasslowpassfilters2D.
            shearlets[:,:,j] =
                Preparedfilters.cone1.wedge[Preparedfilters.shearLevels[scale]+1][:,:,-shearing+2^Preparedfilters.shearLevels[scale]+1].*conj(Preparedfilters.cone1.bandpass[:,:,scale]);
        else
            shearlets[:,:,j] =
                permutedims(Preparedfilters.cone2.wedge[Preparedfilters.shearLevels[scale]+1][:,:,shearing+2^Preparedfilters.shearLevels[scale]+1].*conj(Preparedfilters.cone2.bandpass[:,:,scale]),[2,1]);
        end
    end

	RMS =  transpose(sum(reshape(sum(real.(abs.(shearlets)).^2, dims=1), size(shearlets, 2), size(shearlets, 3)), dims=1));
    if gpu
        RMS = (sqrt.(RMS)/convert(T, sqrt.(rows*cols)));
        dualFrameWeights = sum(real(abs.(shearlets)).^2,dims=3);
    else
        RMS = (sqrt.(RMS)/sqrt.(rows*cols));
        dualFrameWeights = dropdims(sum(abs.(shearlets).^2,dims=3),dims=3); # need to stream to host before they fix abs of comples AFArray
    end
    return (shearlets, dualFrameWeights, RMS, rows, cols, nShearlets)
end

#######################################################################
# Function that generates the desired shearlet system
"""
    ...
    getshearletsystem2D(rows,cols,nScales,shearLevels=ceil.((1:nScales)/2),full = 0,directionalFilter = filt_gen("directional_shearlet"),quadratureMirrorFilter= filt_gen("scaling_shearlet"), gpu = false)

generates the desired shearlet system
  """
function getshearletsystem2D(rows, cols, nScales,
                             shearLevels=ceil.((1:nScales)/2),
                             full= false,
                             directionalFilter = filt_gen("directional_shearlet"),
                             quadratureMirrorFilter=
                             filt_gen("scaling_shearlet"), gpu=false;
                             typeBecomes=Float64, padded=true, tolerance =
                             1e-10, upperFrameBound = -1)

    # Set default value generates the desired shearlet systems
    shearLevels = Int.(shearLevels)

    #Generate prepared Filters and indices
    Preparedfilters = preparefilters2D(rows, cols, nScales, shearLevels,
                                       directionalFilter,
                                       quadratureMirrorFilter,
                                       mirror(quadratureMirrorFilter),
                                       quadratureMirrorFilter,  gpu);
    shearletIdxs = getshearletidxs2D(shearLevels, full);

    # Generate shearlets, RMS(rootmeansquare), dualFrameWeights
    shearlets, dualFrameWeights, RMS, rows, cols, nShearlets =
        generateShearlets(shearletIdxs, Preparedfilters, typeBecomes, gpu)

    # adjust sizes if we're padding, and/or store using half the coefficients
    # if the data is real
    if padded
        # this is probably inefficient, but it works
        padBy = getPadBy(shearlets, tolerance = tolerance)
        shearlets, dualFrameWeights = padShearlets(shearlets, dualFrameWeights,
                                                   typeBecomes, padBy, upperFrameBound)
    else
        padBy = (0,0)
    end
    
    if typeBecomes <: Real
        T = typeBecomes
        CT = typeBecomes
    else
        T = real(typeBecomes)
        CT = typeBecomes
    end
    # we were doing this a lot. Let's do it once
    shearlets = conj(shearlets)

    #return the system
    return Shearletsystem2D{T, CT}(shearlets, Preparedfilters.size,
                            Preparedfilters.shearLevels, full,
                            size(shearletIdxs, 1), shearletIdxs,
                            dualFrameWeights, RMS, gpu, padded, padBy)
end #getshearletsystem2D


# type for individual shearlets2D
struct Shearlets2D{T<:Number, CT <: Union{Complex{T}, Nothing}}
    shearlets::Array{Complex{T}, 3}
    RMS::Array{T, 2}
    dualFrameWeights::Array{T, 2}
    gpu::Bool
    padded::Bool
    padBy::Tuple{Int, Int}
end

#######################################################################
# Function that generates the desired shearlets
"""
   getshearlets2D(PreparedFilters,shearletIdxs,gpu = false) 

generates the 2D shearlets in the frequency domain
"""
function getshearlets2D(Preparedfilters, shearletIdxs =
                        getshearletidxs2D(Preparedfilters.shearLevels), gpu =
                        false; typeBecomes=Float64, padded=true,
                        upperFrameBound = -1)
  # Generate shearlets, RMS(rootmeansquare), dualFrameWeights
    shearlets, dualFrameWeights, RMS, rows, cols, nShearlets =
        generateShearlets(shearletIdxs, Preparedfilters, typeBecomes, gpu)

    # adjust sizes if we're padding, and/or store using half the coefficients
    # if the data is real
    padBy = getPadBy(shearlets, tolerance = tolerance)
    if padded || typeBecomes <: Real
        # this is probably inefficient, but it works
        shearlets, dualFrameWeights = padShearlets(shearlets, dualFrameWeights,
                                                   typeBecomes, padBy, upperFrameBound)
    end

    if typeBecomes <: Real
        T = typeBecomes
        CT = typeBecomes
    else
        T = real(typeBecomes)
        CT = typeBecomes
    end

	return Shearlets2D{T, CT}(shearlets, RMS, dualFrameWeights, gpu, padded, padBy)
end #getshearlets2D
