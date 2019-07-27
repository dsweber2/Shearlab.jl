# Submodule to compute the coefficients of shearlet recovery and decoding in 2D

##############################################################################
# tools for doing efficient Fourier domain operations

"""
    unshearing!(X, neededShear, P, coeffs, padBy, j) where {N,T}
    unshearing!(X::AbstractArray{Complex{T},2}, neededShear, P,
                coeffs::AbstractArray{T,3}, padBy, j) where {N,T}
given a set of shearlet coefficients `coeffs`, add the inverse of the `j`th one
to X
"""
function unshearing!(X̂, neededShear, P, coeffs, padBy, j)
    for i in eachindex(view(X̂, 1, 1, axes(X̂)[3:end]...))
        cFreq = fftshift( P * (pad(coeffs[:, :, j, i], padBy)))
        X̂[:, :, i] = X̂[:, :, i] + cFreq .* neededShear
    end
end

function unshearing!(X̂::AbstractArray{S,2}, neededShear, P,
                     coeffs::AbstractArray{T,3}, padBy, j) where {T,
                                                                  S <: Complex}
    cFreq = fftshift( P * (pad(coeffs[:, :, j], padBy)))
    X̂[:, :] = X̂[:, :] + cFreq .* neededShear
end

"""
    baseDomain!(X, X̂, duals, P, used1, used2)
Multiply by the dual coordinates and convert to the space domain
"""
function baseDomain!(X::AbstractArray{S}, X̂::AbstractArray{T,N}, duals, P,
                     used1, used2) where {S<:Complex, T, N}
    for i in eachindex(view(X, 1, 1, axes(X)[3:end]...))
        X[:, :, i] = fftshift((P \ (ifftshift(X̂[:,:, i] ./
                                              (duals)))))[used1, used2]
    end
end


function baseDomain!(X::AbstractArray{S}, X̂::AbstractArray{T,2}, duals, P,
                     used1, used2) where {T, S<:Complex}
    X[:, :] = fftshift(P \ (ifftshift(X̂[:,:] ./ (duals))))[used1, used2]
end

function baseDomain!(X::AbstractArray{S}, X̂::AbstractArray{T,N}, duals, P,
                     used1, used2) where {S<:Real, T, N}
    for i in eachindex(view(X, 1, 1, axes(X)[3:end]...))
        X[:, :, i] = real.(fftshift((P \ (ifftshift(X̂[:,:, i] ./
                                              (duals)))))[used1, used2])
    end
end

function baseDomain!(X::AbstractArray{S}, X̂::AbstractArray{T,2}, duals, P,
                     used1, used2) where {T, S<:Real}
    X[:, :] = real.(
        fftshift(P \ (ifftshift(X̂[:,:] ./ (duals))))[used1, used2])
end


"""
    shearing!(X::AbstractArray{T,N}, neededShear, P, coeffs, padBy, used1,
                   used2, j) where {N,T}
    shearing!(X::AbstractArray{T,2}, neededShear, P, coeffs, padBy, used1,
                   used2, j) where T
fill in the `j`th coefficient and fill it into the matrix of coefficients
"""
function shearing!(X::AbstractArray{T,N}, neededShear, P, coeffs, padBy, used1,
                   used2, j) where {N,T}
    for i in eachindex(view(X, 1, 1, axes(X)[3:end]...))
        Xfreq = fftshift( P * ifftshift(pad(X[:, :, i], padBy)))
        tmpCoeff = (P \ ifftshift(Xfreq .* neededShear))[used1, used2]
        if T <: Real
            coeffs[:, :, j, i] = real.(tmpCoeff)
        else
            coeffs[:, :, j, i] = tmpCoeff
        end
    end
end

function shearing!(X::AbstractArray{T,2}, neededShear, P, coeffs, padBy, used1,
                   used2, j) where T
    Xfreq =  fftshift( P * ifftshift(pad(X[:, :], padBy)))
    tmpCoeff = (P \ ifftshift(Xfreq .* neededShear))[used1, used2]
    if T <: Real
        coeffs[:, :, j] = real.(tmpCoeff)
    else
        coeffs[:, :, j] = tmpCoeff
    end
end


function getAPlan(P, zeroMat, CT, padBy)
    if (typeof(P) <: Real) && (CT <: Real)
        P = plan_rfft(pad(zeroMat, padBy), flags = FFTW.PATIENT)
    elseif (typeof(P) <: Real) && (CT <: Complex)
        P = plan_fft!(pad(zeroMat, padBy), flags =
                      FFTW.PATIENT)
    end
    return P
end




##############################################################################
## Function that compute the coefficient matrix of the Shearlet Transform of
## some array
"""
    sheardec2D(X::AbstractArray{CT,N},
                    shearletSystem::Shearlab.Shearletsystem2D{T, CT};
                    P::{plan_(r)fft}=-1) where {T <: Number, CT, N}

Compute the coefficient matrix of the Shearlet transform across the first two
dimensions of X. If `shearletSystem.padded` is true, it pads the shearlet
coefficients by `padBy`, which defaults to the widest spatial component of any
of the shearlets (computed via `getPadBy`). If you're doing a lot of these, you
can precompute the Fourier transform plan and put it in `P` (note that it is
type sensitive), and should have the size of X *after* padding.

"""
function sheardec2D(X::AbstractArray{CT, N},
                    shearletSystem::Shearlab.Shearletsystem2D{T, CT}; P=-1) where {T <: Number, CT, N}
    @assert N >= 2

    padBy = shearletSystem.padBy

    if N>2
        coeffShape = (size(X)[1:2]..., size(shearletSystem.shearlets, 3),
                      size(X)[3:end]...) 
    else
        coeffShape = (size(X)[1:2]..., size(shearletSystem.shearlets, 3))
    end
    
    coeffs = zeros(CT, coeffShape...)
    
    # if you didn't hand it a plan, adjust the kind made based on the inputs
    # data Type
    P = getAPlan(P, zeros(CT, size(X)[1:2]), CT, padBy)
    
    # compute shearlet coefficients at each scale
    # note that pointwise multiplication in the fourier domain equals
    # convolution in the time-domain
    used1 = padBy[1] .+ (1:size(X, 1))
    used2 = padBy[2] .+ (1:size(X, 2))
    for j = 1:shearletSystem.nShearlets
        # The fourier transform of X
        neededShear = conj(shearletSystem.shearlets[:, :, j])
        shearing!(X, neededShear, P,  coeffs, padBy, used1, used2, j)
    end
    return coeffs
end # sheardec2D




##############################################################################
## Function that recovers the image with the Shearlet transform with some shearletSystem
"""
    shearrec2D(coeffs::AbstractArray{complex{T},N},
               shearletSystem::Shearlab.Shearletsystem2D{T}; P=-1,
               padBy::Tuple{Int, Int} = getPadBy(shearletSystem)) where {T<:Real, N, B, C, D} 

function that recovers the image with the Shearlet transform with some
shearletSystem. If you're doing a lot of these, you can precompute the Fourier
transform plan and put it in `P` (note that it is type sensitive), and should
have the size of X *after* padding.
"""

function shearrec2D(coeffs::AbstractArray{CT, N},
                    shearletSystem::Shearlab.Shearletsystem2D{T, CT}; P=-1) where {T<:Real, CT<:Union{T, Complex{T}}, N, B, C, D}

    padBy = shearletSystem.padBy
    # are there meta dimensions? The storage needs to adapt
    if N > 2
        X̂ = zeros(Complex{T}, size(shearletSystem.shearlets)[1:2]..., # this is X\^
                  size(coeffs)[4:end]...)
    else
        X̂ = zeros(Complex{T}, size(shearletSystem.shearlets)[1:2]...)
    end

    P = getAPlan(P, zeros(CT, shearletSystem.size), CT, padBy)
    
    # sum X in the Fourier domain over the coefficients at each scale/shearing
    used1 = padBy[1] .+ (1:shearletSystem.size[1])
    used2 = padBy[2] .+ (1:shearletSystem.size[2])
    for j = 1:shearletSystem.nShearlets
        neededShear = shearletSystem.shearlets[:, :, j]
        unshearing!(X̂, neededShear, P,  coeffs, padBy, j)
    end
    X = zeros(CT, size(coeffs)[1:end-1])
    baseDomain!(X, X̂, shearletSystem.dualFrameWeights, P, used1, used2)
    return X
end


##############################################################################
## Function that compute the array of the adjoint Shearlet Transform of
## some coefficients matrix
"""
...
sheardecadjoint2D(coeffs,shearletSystem) compute the adjoint of the decomposition operator
...
"""
function sheardecadjoint2D(coeffs,shearletSystem)
		# Initialize reconstructed data
    gpu = shearletSystem.gpu;
    if gpu == 1
        X = AFArray(zeros(Complex{Float32},size(coeffs,1),size(coeffs,2)));
    else
        X = zeros(Complex{Float64},size(coeffs,1),size(coeffs,2));
    end
    for j = 1:shearletSystem.nShearlets
        X = X+fftshift(fft(ifftshift(coeffs[:,:,j]))).*conj(shearletSystem.shearlets[:,:,j]);
    end
    return real(fftshift(ifft(ifftshift((1 ./shearletSystem.dualFrameWeights).*X))))

end # sheardecadjoint2D

##########################################################################################
## Function that compute the coefficient matrix of the adjoint inverse Shearlet Transform of
## some array
"""
...
    shearrecadjoint2D(X,shearletSystem) 

compute the coefficient matrix of the adjoint inverse Shearlet transform of the array X
...
"""
function shearrecadjoint2D(X,shearletSystem)
    #Read the GPU info of the system
    gpu = shearletSystem.gpu;
    if gpu == 1
        coeffs = AFArray(zeros(Complex{Float32},size(shearletSystem.shearlets)));
    else
        coeffs = zeros(Complex{Float64},size(shearletSystem.shearlets));
    end
    # The fourier transform of X
    Xfreq = fftshift(fft(ifftshift(X)));
    #compute shearlet coefficients at each scale
    #not that pointwise multiplication in the fourier domain equals convolution
    #in the time-domain
    for j = 1:shearletSystem.nShearlets
        coeffs[:,:,j] = fftshift(ifft(ifftshift(Xfreq.*shearletSystem.shearlets[:,:,j])));
    end
    return coeffs
end # shearrecadjoint2D
